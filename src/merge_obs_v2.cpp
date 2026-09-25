#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

namespace {

constexpr unsigned EVENT_BITS = 20;
constexpr unsigned STATION_PHASE_BITS = 11;
constexpr uint64_t EVENT_MASK = (uint64_t{1} << EVENT_BITS) - 1;
constexpr uint64_t STATION_PHASE_MASK = (uint64_t{1} << STATION_PHASE_BITS) - 1;
constexpr unsigned RADIX_BITS = 13;
constexpr size_t RADIX_SIZE = size_t{1} << RADIX_BITS;
constexpr uint64_t RADIX_MASK = RADIX_SIZE - 1;
constexpr unsigned RADIX_PASSES = 4;

struct Record {
    uint64_t key;
    uint32_t dt;
    uint32_t cc;
};

static_assert(sizeof(Record) == 16, "Record must remain compact");

struct MappedFile {
    std::string path;
    int fd = -1;
    const char* data = nullptr;
    size_t size = 0;

    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept
        : path(std::move(other.path)), fd(other.fd), data(other.data), size(other.size) {
        other.fd = -1;
        other.data = nullptr;
        other.size = 0;
    }

    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close_file();
            path = std::move(other.path);
            fd = other.fd;
            data = other.data;
            size = other.size;
            other.fd = -1;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    ~MappedFile() { close_file(); }

    void close_file() {
        if (data != nullptr) {
            munmap(const_cast<char*>(data), size);
            data = nullptr;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }
};

struct Chunk {
    size_t file_index;
    size_t begin;
    size_t end;
};

struct Options {
    int threads = 30;
    int write_threads = 1;
    int num_sta_thres = 8;
    double memory_limit_gib = 300.0;
    bool binary_input = false;
    std::string station_file;
    std::string output_path;
    std::vector<std::string> input_paths;
};

std::string errno_message(const std::string& action) {
    return action + ": " + std::strerror(errno);
}

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --station-file PATH --output PATH --input FILE [--input FILE ...]\n"
        << "       [--threads 30] [--write-threads 1] [--num-sta-thres 8]\n"
        << "       [--memory-limit-gib 300] [--binary-input]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[i];
        };

        if (arg == "--input") {
            options.input_paths.push_back(require_value(arg));
        } else if (arg == "--station-file") {
            options.station_file = require_value(arg);
        } else if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value(arg));
        } else if (arg == "--write-threads") {
            options.write_threads = std::stoi(require_value(arg));
        } else if (arg == "--num-sta-thres") {
            options.num_sta_thres = std::stoi(require_value(arg));
        } else if (arg == "--memory-limit-gib") {
            options.memory_limit_gib = std::stod(require_value(arg));
        } else if (arg == "--binary-input") {
            options.binary_input = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.input_paths.empty() || options.station_file.empty() || options.output_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--input, --station-file, and --output are required");
    }
    if (options.threads < 1 || options.write_threads < 1 || options.num_sta_thres < 1) {
        throw std::runtime_error("Thread counts and station threshold must be positive");
    }
    return options;
}

uint64_t pack_ascii(const char* begin, const char* end) {
    size_t length = static_cast<size_t>(end - begin);
    if (length == 0 || length > 8) {
        throw std::runtime_error("Station names must contain 1 to 8 ASCII characters");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < length; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(begin[i])) << (56 - 8 * i);
    }
    return value;
}

std::vector<std::string> load_stations(
    const std::string& path,
    std::unordered_map<uint64_t, uint16_t>& station_ids
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open station file: " + path);
    }

    std::vector<std::string> stations;
    std::string line;
    while (std::getline(input, line)) {
        size_t end = line.find_first_of(", \t\r\n");
        std::string station = line.substr(0, end);
        if (!station.empty()) {
            stations.push_back(station);
        }
    }
    std::sort(stations.begin(), stations.end());
    stations.erase(std::unique(stations.begin(), stations.end()), stations.end());
    if (stations.size() > 1024) {
        throw std::runtime_error("At most 1024 stations can be encoded");
    }

    for (size_t i = 0; i < stations.size(); ++i) {
        uint64_t packed = pack_ascii(stations[i].data(), stations[i].data() + stations[i].size());
        station_ids.emplace(packed, static_cast<uint16_t>(i));
    }
    return stations;
}

MappedFile map_file(const std::string& path) {
    MappedFile file;
    file.path = path;
    file.fd = open(path.c_str(), O_RDONLY);
    if (file.fd < 0) {
        throw std::runtime_error(errno_message("Cannot open " + path));
    }
    struct stat info {};
    if (fstat(file.fd, &info) != 0) {
        throw std::runtime_error(errno_message("Cannot stat " + path));
    }
    file.size = static_cast<size_t>(info.st_size);
    if (file.size == 0) {
        return file;
    }
    void* mapping = mmap(nullptr, file.size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(errno_message("Cannot mmap " + path));
    }
    file.data = static_cast<const char*>(mapping);
    madvise(const_cast<char*>(file.data), file.size, MADV_SEQUENTIAL);
    return file;
}

std::vector<int> chunks_per_file(const std::vector<MappedFile>& files, int requested_threads) {
    int total_chunks = std::max<int>(requested_threads, files.size());
    std::vector<int> counts(files.size(), 1);
    std::vector<double> ideal(files.size(), 0.0);
    uint64_t total_size = 0;
    for (const auto& file : files) {
        total_size += file.size;
    }
    if (total_size == 0) {
        return counts;
    }

    int assigned = static_cast<int>(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        ideal[i] = static_cast<double>(files[i].size) * total_chunks / total_size;
        int extra = std::max(0, static_cast<int>(ideal[i]) - 1);
        counts[i] += extra;
        assigned += extra;
    }
    while (assigned < total_chunks) {
        size_t best = 0;
        double best_deficit = -1e100;
        for (size_t i = 0; i < files.size(); ++i) {
            double deficit = ideal[i] - counts[i];
            if (deficit > best_deficit) {
                best_deficit = deficit;
                best = i;
            }
        }
        ++counts[best];
        ++assigned;
    }
    return counts;
}

size_t next_line_start(const MappedFile& file, size_t position) {
    if (position == 0 || position >= file.size) {
        return std::min(position, file.size);
    }
    const void* found = std::memchr(file.data + position, '\n', file.size - position);
    return found == nullptr
        ? file.size
        : static_cast<size_t>(static_cast<const char*>(found) - file.data) + 1;
}

std::vector<Chunk> make_chunks(const std::vector<MappedFile>& files, int threads) {
    std::vector<int> counts = chunks_per_file(files, threads);
    std::vector<Chunk> chunks;
    for (size_t file_index = 0; file_index < files.size(); ++file_index) {
        const auto& file = files[file_index];
        for (int part = 0; part < counts[file_index]; ++part) {
            size_t raw_begin = file.size * static_cast<size_t>(part) / counts[file_index];
            size_t raw_end = file.size * static_cast<size_t>(part + 1) / counts[file_index];
            size_t begin = part == 0 ? 0 : next_line_start(file, raw_begin);
            size_t end = part + 1 == counts[file_index] ? file.size : next_line_start(file, raw_end);
            if (begin < end) {
                chunks.push_back({file_index, begin, end});
            }
        }
    }
    return chunks;
}

inline void skip_spaces(const char*& cursor, const char* end) {
    while (cursor < end && (*cursor == ' ' || *cursor == '\t')) {
        ++cursor;
    }
}

bool parse_uint(const char*& cursor, const char* end, uint32_t& value) {
    skip_spaces(cursor, end);
    if (cursor >= end || *cursor < '0' || *cursor > '9') {
        return false;
    }
    uint64_t result = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        result = result * 10 + static_cast<unsigned>(*cursor - '0');
        ++cursor;
    }
    if (result > EVENT_MASK) {
        return false;
    }
    value = static_cast<uint32_t>(result);
    return true;
}

bool parse_decimal6(const char*& cursor, const char* end, uint32_t& encoded) {
    skip_spaces(cursor, end);
    bool negative = false;
    if (cursor < end && (*cursor == '-' || *cursor == '+')) {
        negative = *cursor == '-';
        ++cursor;
    }
    if (cursor >= end || *cursor < '0' || *cursor > '9') {
        return false;
    }

    uint64_t whole = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        whole = whole * 10 + static_cast<unsigned>(*cursor - '0');
        ++cursor;
    }
    uint32_t fraction = 0;
    unsigned digits = 0;
    if (cursor < end && *cursor == '.') {
        ++cursor;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            if (digits >= 6) {
                return false;
            }
            fraction = fraction * 10 + static_cast<unsigned>(*cursor - '0');
            ++digits;
            ++cursor;
        }
    }
    while (digits++ < 6) {
        fraction *= 10;
    }
    uint64_t magnitude = whole * 1000000 + fraction;
    if (magnitude > 0x7fffffffU) {
        return false;
    }
    encoded = static_cast<uint32_t>(magnitude) | (negative ? 0x80000000U : 0U);
    return true;
}

bool parse_line(
    const char* begin,
    const char* end,
    const std::unordered_map<uint64_t, uint16_t>& station_ids,
    Record& record,
    std::string& error
) {
    const char* cursor = begin;
    uint32_t data_evid = 0;
    uint32_t temp_evid = 0;
    if (!parse_uint(cursor, end, data_evid) || !parse_uint(cursor, end, temp_evid)) {
        error = "invalid event id";
        return false;
    }

    skip_spaces(cursor, end);
    const char* station_begin = cursor;
    while (cursor < end && *cursor != ' ' && *cursor != '\t' && *cursor != '\r') {
        ++cursor;
    }
    uint64_t packed_station = 0;
    try {
        packed_station = pack_ascii(station_begin, cursor);
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
    auto station_it = station_ids.find(packed_station);
    if (station_it == station_ids.end()) {
        error = "station is missing from station file";
        return false;
    }

    skip_spaces(cursor, end);
    if (cursor >= end || (*cursor != 'P' && *cursor != 'S')) {
        error = "invalid phase";
        return false;
    }
    uint64_t phase = *cursor++ == 'S' ? 1 : 0;

    if (!parse_decimal6(cursor, end, record.dt) || !parse_decimal6(cursor, end, record.cc)) {
        error = "invalid dt or cc value";
        return false;
    }
    skip_spaces(cursor, end);
    while (cursor < end && *cursor == '\r') {
        ++cursor;
    }
    if (cursor != end) {
        error = "unexpected trailing field";
        return false;
    }

    uint64_t pair_key = (static_cast<uint64_t>(data_evid) << EVENT_BITS) | temp_evid;
    record.key = (pair_key << STATION_PHASE_BITS)
        | (static_cast<uint64_t>(station_it->second) << 1)
        | phase;
    return true;
}

Record* allocate_records(size_t count) {
    if (count == 0) {
        return nullptr;
    }
    size_t bytes = count * sizeof(Record);
    void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        throw std::runtime_error(errno_message("Cannot allocate record array"));
    }
    madvise(memory, bytes, MADV_HUGEPAGE);
    return static_cast<Record*>(memory);
}

void free_records(Record* records, size_t count) {
    if (records != nullptr) {
        munmap(records, count * sizeof(Record));
    }
}

void stable_parallel_radix_sort(Record*& records, Record*& scratch, size_t count, int threads) {
    const size_t matrix_size = static_cast<size_t>(threads) * RADIX_SIZE;
    std::vector<uint64_t> histograms(matrix_size);
    std::vector<uint64_t> offsets(matrix_size);
    Record* source = records;
    Record* destination = scratch;

    for (unsigned pass = 0; pass < RADIX_PASSES; ++pass) {
        unsigned shift = pass * RADIX_BITS;
        std::fill(histograms.begin(), histograms.end(), 0);

#pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            size_t begin = count * static_cast<size_t>(tid) / threads;
            size_t end = count * static_cast<size_t>(tid + 1) / threads;
            uint64_t* histogram = histograms.data() + static_cast<size_t>(tid) * RADIX_SIZE;
            for (size_t i = begin; i < end; ++i) {
                ++histogram[(source[i].key >> shift) & RADIX_MASK];
            }
        }

        uint64_t running = 0;
        for (size_t bucket = 0; bucket < RADIX_SIZE; ++bucket) {
            uint64_t position = running;
            for (int tid = 0; tid < threads; ++tid) {
                size_t index = static_cast<size_t>(tid) * RADIX_SIZE + bucket;
                offsets[index] = position;
                position += histograms[index];
            }
            running = position;
        }

#pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            size_t begin = count * static_cast<size_t>(tid) / threads;
            size_t end = count * static_cast<size_t>(tid + 1) / threads;
            uint64_t* positions = offsets.data() + static_cast<size_t>(tid) * RADIX_SIZE;
            for (size_t i = begin; i < end; ++i) {
                size_t bucket = (source[i].key >> shift) & RADIX_MASK;
                destination[positions[bucket]++] = source[i];
            }
        }
        std::swap(source, destination);
        std::cout << "Radix pass " << (pass + 1) << "/" << RADIX_PASSES << " complete" << std::endl;
    }

    if (source != records) {
        std::swap(records, scratch);
    }
}

inline uint64_t pair_key(const Record& record) {
    return record.key >> STATION_PHASE_BITS;
}

inline uint16_t station_id(const Record& record) {
    return static_cast<uint16_t>((record.key & STATION_PHASE_MASK) >> 1);
}

inline unsigned phase_id(const Record& record) {
    return static_cast<unsigned>(record.key & 1U);
}

double decode_decimal(uint32_t encoded) {
    double value = static_cast<double>(encoded & 0x7fffffffU) / 1000000.0;
    return (encoded & 0x80000000U) ? -value : value;
}

void append_pair(
    std::string& output,
    const Record* records,
    size_t begin,
    size_t end,
    const std::vector<std::string>& station_codes,
    int num_sta_thres,
    uint64_t& pair_count
) {
    int stations = 0;
    uint16_t previous_station = 0xffffU;
    for (size_t i = begin; i < end; ++i) {
        uint16_t current_station = station_id(records[i]);
        if (current_station != previous_station) {
            ++stations;
            previous_station = current_station;
        }
    }
    if (stations < num_sta_thres) {
        return;
    }

    uint64_t pair = pair_key(records[begin]);
    uint32_t data_evid = static_cast<uint32_t>(pair >> EVENT_BITS);
    uint32_t temp_evid = static_cast<uint32_t>(pair & EVENT_MASK);
    char buffer[160];
    int length = std::snprintf(buffer, sizeof(buffer), "# %9u %9u 0.0\n", data_evid, temp_evid);
    output.append(buffer, static_cast<size_t>(length));

    size_t i = begin;
    while (i < end) {
        uint16_t station = station_id(records[i]);
        size_t station_end = i + 1;
        while (station_end < end && station_id(records[station_end]) == station) {
            ++station_end;
        }

        size_t phase_begin = i;
        while (phase_begin < station_end) {
            unsigned phase = phase_id(records[phase_begin]);
            size_t phase_end = phase_begin + 1;
            while (phase_end < station_end && phase_id(records[phase_end]) == phase) {
                ++phase_end;
            }
            const Record& chosen = records[phase_end - 1];
            length = std::snprintf(
                buffer,
                sizeof(buffer),
                "%-7s %8.5f %.4f %c\n",
                station_codes[station].c_str(),
                decode_decimal(chosen.dt),
                decode_decimal(chosen.cc),
                phase == 0 ? 'P' : 'S'
            );
            output.append(buffer, static_cast<size_t>(length));
            phase_begin = phase_end;
        }
        i = station_end;
    }
    ++pair_count;
}

std::string station_code(const std::string& station) {
    size_t dot = station.rfind('.');
    return dot == std::string::npos ? station : station.substr(dot + 1);
}

void write_all_at(int fd, const char* data, size_t size, off_t offset) {
    size_t written = 0;
    while (written < size) {
        ssize_t result = pwrite(fd, data + written, size - written, offset + static_cast<off_t>(written));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(errno_message("pwrite failed"));
        }
        written += static_cast<size_t>(result);
    }
}

std::string output_directory(const std::string& path) {
    size_t slash = path.rfind('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string output_filename(const std::string& path) {
    size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);

        std::unordered_map<uint64_t, uint16_t> station_ids;
        std::vector<std::string> stations = load_stations(options.station_file, station_ids);
        std::vector<std::string> station_codes;
        station_codes.reserve(stations.size());
        std::unordered_set<std::string> seen_station_codes;
        for (const auto& station : stations) {
            std::string code = station_code(station);
            if (!seen_station_codes.insert(code).second) {
                throw std::runtime_error("Duplicate station code after removing network prefix: " + code);
            }
            station_codes.push_back(std::move(code));
        }

        std::vector<MappedFile> files;
        files.reserve(options.input_paths.size());
        uint64_t input_bytes = 0;
        for (const auto& path : options.input_paths) {
            files.push_back(map_file(path));
            input_bytes += files.back().size;
        }
        std::cout << "Inputs: " << files.size() << " files, "
                  << (input_bytes / 1024.0 / 1024.0 / 1024.0) << " GiB; "
                  << "stations=" << stations.size()
                  << ", format=" << (options.binary_input ? "native-binary" : "text") << std::endl;

        uint64_t record_count = 0;
        Record* records = nullptr;
        if (options.binary_input) {
            std::vector<uint64_t> record_offsets(files.size() + 1, 0);
            for (size_t i = 0; i < files.size(); ++i) {
                if (files[i].size % sizeof(Record) != 0) {
                    throw std::runtime_error(
                        "Native observation file size is not a multiple of 16 bytes: " + files[i].path
                    );
                }
                record_offsets[i] = record_count;
                record_count += files[i].size / sizeof(Record);
            }
            record_offsets.back() = record_count;
            double sort_peak_gib = 2.0 * record_count * sizeof(Record)
                / 1024.0 / 1024.0 / 1024.0;
            std::cout << "Loaded " << record_count << " native rows; compact data="
                      << (record_count * sizeof(Record) / 1024.0 / 1024.0 / 1024.0)
                      << " GiB; estimated sort peak=" << sort_peak_gib << " GiB" << std::endl;
            if (sort_peak_gib > options.memory_limit_gib) {
                throw std::runtime_error("Native record sort exceeds --memory-limit-gib.");
            }
            records = allocate_records(record_count);
#pragma omp parallel for schedule(static) num_threads(options.threads)
            for (size_t i = 0; i < files.size(); ++i) {
                if (files[i].size > 0) {
                    std::memcpy(records + record_offsets[i], files[i].data, files[i].size);
                }
            }
        } else {
            std::vector<Chunk> chunks = make_chunks(files, options.threads);
            std::cout << "Text parse chunks: " << chunks.size() << std::endl;
            std::vector<std::vector<Record>> local_records(chunks.size());
            std::atomic<bool> parse_failed(false);
            std::mutex error_mutex;
            std::string parse_error;

#pragma omp parallel for schedule(static) num_threads(options.threads)
            for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
                const Chunk& chunk = chunks[chunk_index];
                const MappedFile& file = files[chunk.file_index];
                auto& output = local_records[chunk_index];
                output.reserve((chunk.end - chunk.begin) / 32 + 1);
                const char* cursor = file.data + chunk.begin;
                const char* chunk_end = file.data + chunk.end;
                while (cursor < chunk_end && !parse_failed.load(std::memory_order_relaxed)) {
                    const char* newline = static_cast<const char*>(std::memchr(cursor, '\n', chunk_end - cursor));
                    const char* line_end = newline == nullptr ? chunk_end : newline;
                    if (line_end > cursor) {
                        Record record {};
                        std::string error;
                        if (!parse_line(cursor, line_end, station_ids, record, error)) {
                            parse_failed.store(true, std::memory_order_relaxed);
                            std::lock_guard<std::mutex> lock(error_mutex);
                            if (parse_error.empty()) {
                                parse_error = file.path + " near byte "
                                    + std::to_string(static_cast<size_t>(cursor - file.data)) + ": " + error;
                            }
                            break;
                        }
                        output.push_back(record);
                    }
                    cursor = newline == nullptr ? chunk_end : newline + 1;
                }
            }
            if (parse_failed.load()) {
                throw std::runtime_error("Observation parse failed: " + parse_error);
            }

            uint64_t local_capacity = 0;
            std::vector<uint64_t> record_offsets(local_records.size() + 1, 0);
            for (size_t i = 0; i < local_records.size(); ++i) {
                record_offsets[i] = record_count;
                record_count += local_records[i].size();
                local_capacity += local_records[i].capacity();
            }
            record_offsets.back() = record_count;
            double parse_peak_gib = (local_capacity + record_count) * sizeof(Record)
                / 1024.0 / 1024.0 / 1024.0;
            double sort_peak_gib = 2.0 * record_count * sizeof(Record)
                / 1024.0 / 1024.0 / 1024.0;
            double estimated_peak = std::max(parse_peak_gib, sort_peak_gib);
            std::cout << "Parsed " << record_count << " rows; compact data="
                      << (record_count * sizeof(Record) / 1024.0 / 1024.0 / 1024.0)
                      << " GiB; estimated pre-output peak=" << estimated_peak << " GiB" << std::endl;
            if (estimated_peak > options.memory_limit_gib) {
                throw std::runtime_error(
                    "Estimated memory exceeds --memory-limit-gib; raise the limit or use fewer/larger input partitions"
                );
            }

            records = allocate_records(record_count);
#pragma omp parallel for schedule(static) num_threads(options.threads)
            for (size_t i = 0; i < local_records.size(); ++i) {
                std::memcpy(
                    records + record_offsets[i],
                    local_records[i].data(),
                    local_records[i].size() * sizeof(Record)
                );
            }
        }

        files.clear();
        files.shrink_to_fit();
        Record* scratch = allocate_records(record_count);
        stable_parallel_radix_sort(records, scratch, record_count, options.threads);
        free_records(scratch, record_count);
        scratch = nullptr;

        int format_threads = std::min<int>(options.threads, record_count == 0 ? 1 : record_count);
        std::vector<size_t> boundaries(static_cast<size_t>(format_threads) + 1, 0);
        boundaries.back() = record_count;
        for (int thread = 1; thread < format_threads; ++thread) {
            size_t position = record_count * static_cast<size_t>(thread) / format_threads;
            while (position < record_count && position > 0
                   && pair_key(records[position]) == pair_key(records[position - 1])) {
                ++position;
            }
            boundaries[thread] = std::max(position, boundaries[thread - 1]);
        }

        std::vector<std::string> output_chunks(static_cast<size_t>(format_threads));
        std::vector<uint64_t> output_pair_counts(static_cast<size_t>(format_threads), 0);
#pragma omp parallel for schedule(static) num_threads(format_threads)
        for (int thread = 0; thread < format_threads; ++thread) {
            size_t begin = boundaries[thread];
            size_t end = boundaries[thread + 1];
            std::string& output = output_chunks[thread];
            output.reserve((end - begin) * 32);
            size_t position = begin;
            while (position < end) {
                size_t pair_end = position + 1;
                uint64_t current_pair = pair_key(records[position]);
                while (pair_end < end && pair_key(records[pair_end]) == current_pair) {
                    ++pair_end;
                }
                append_pair(
                    output,
                    records,
                    position,
                    pair_end,
                    station_codes,
                    options.num_sta_thres,
                    output_pair_counts[thread]
                );
                position = pair_end;
            }
        }
        free_records(records, record_count);
        records = nullptr;

        std::vector<off_t> output_offsets(output_chunks.size() + 1, 0);
        uint64_t total_pairs = 0;
        for (size_t i = 0; i < output_chunks.size(); ++i) {
            output_offsets[i + 1] = output_offsets[i] + static_cast<off_t>(output_chunks[i].size());
            total_pairs += output_pair_counts[i];
        }

        std::string directory = output_directory(options.output_path);
        std::string pending_template = directory + "/." + output_filename(options.output_path) + ".XXXXXX";
        std::vector<char> pending_buffer(pending_template.begin(), pending_template.end());
        pending_buffer.push_back('\0');
        int output_fd = mkstemp(pending_buffer.data());
        if (output_fd < 0) {
            throw std::runtime_error(errno_message("Cannot create pending output"));
        }
        std::string pending_path = pending_buffer.data();
        try {
            if (ftruncate(output_fd, output_offsets.back()) != 0) {
                throw std::runtime_error(errno_message("Cannot size pending output"));
            }
            std::atomic<bool> write_failed(false);
            std::mutex write_error_mutex;
            std::string write_error;
#pragma omp parallel for schedule(static) num_threads(options.write_threads)
            for (size_t i = 0; i < output_chunks.size(); ++i) {
                if (write_failed.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    write_all_at(output_fd, output_chunks[i].data(), output_chunks[i].size(), output_offsets[i]);
                } catch (const std::exception& exc) {
                    write_failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(write_error_mutex);
                    if (write_error.empty()) {
                        write_error = exc.what();
                    }
                }
            }
            if (write_failed.load()) {
                throw std::runtime_error(write_error);
            }
            if (close(output_fd) != 0) {
                output_fd = -1;
                throw std::runtime_error(errno_message("Cannot close pending output"));
            }
            output_fd = -1;
            if (rename(pending_path.c_str(), options.output_path.c_str()) != 0) {
                throw std::runtime_error(errno_message("Cannot replace output"));
            }
        } catch (...) {
            if (output_fd >= 0) {
                close(output_fd);
            }
            unlink(pending_path.c_str());
            throw;
        }

        std::cout << "Wrote " << total_pairs << " event pairs, "
                  << (output_offsets.back() / 1024.0 / 1024.0 / 1024.0)
                  << " GiB, to " << options.output_path << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << std::endl;
        return 1;
    }
}
