"""Utilities for GPU-oriented waveform cutting and dt.cc calculation."""

from collections import defaultdict
import json
import multiprocessing as mp
import os

import numpy as np
from numcodecs import Blosc
from obspy import UTCDateTime, read
from obspy.signal.filter import bandpass, highpass, lowpass
from scipy.interpolate import InterpolatedUnivariateSpline
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

from reader import dtime2str, get_data_dict, iter_fpha, read_fsta
from signal_lib import preprocess

zarr.storage.default_format = 2

_STATION_ASSIGN_RECORDS = None
_STATION_ASSIGN_EVENT_MAP = None
_STATION_ASSIGN_STA_DICT = None
_STATION_ASSIGN_CFG = None
_STATION_ASSIGN_PAIR_ARRAY = None
_STATION_ASSIGN_TASK_DIR = None
_PAIR_BUILD_OT = None
_PAIR_BUILD_LAT = None
_PAIR_BUILD_LON = None
_PAIR_BUILD_DEP = None
_PAIR_BUILD_IS_TEMP = None
_PAIR_BUILD_STATION_SETS = None
_PAIR_BUILD_PARAMS = None

_NATIVE_EVENT_BITS = 20
_NATIVE_STATION_PHASE_BITS = 11
_NATIVE_RECORD_DTYPE = np.dtype(
    [("key", "<u8"), ("dt", "<u4"), ("cc", "<u4")],
    align=False,
)


def calc_dist_km(lat, lon):
    cos_lat = np.cos(np.mean(lat) * np.pi / 180.0)
    dx = cos_lat * (lon[1] - lon[0])
    dy = lat[1] - lat[0]
    return 111.0 * (dx**2 + dy**2) ** 0.5


def get_phase_specs(cfg):
    specs = {}
    phase_cfg = {
        "P": (cfg.chn_p, cfg.win_temp_p, cfg.dt_thres[0]),
        "S": (cfg.chn_s, cfg.win_temp_s, cfg.dt_thres[1]),
    }
    for phase_name, (channels, temp_win, dt_limit) in phase_cfg.items():
        data_win = [temp_win[0] + dt_limit, temp_win[1] + dt_limit]
        temp_npts = int(round(sum(temp_win) * cfg.samp_rate))
        data_npts = int(round(sum(data_win) * cfg.samp_rate))
        temp_start = int(round((data_win[0] - temp_win[0]) * cfg.samp_rate))
        specs[phase_name] = {
            "channels": list(channels),
            "temp_win": list(temp_win),
            "data_win": list(data_win),
            "temp_npts": temp_npts,
            "data_npts": data_npts,
            "temp_start": temp_start,
            "temp_end": temp_start + temp_npts,
            "dt_limit": float(dt_limit),
            "tt_shift": float(temp_win[0] - data_win[0]),
        }
    return specs


def build_compressor(cfg):
    compressor_name = str(getattr(cfg, "zarr_compressor", "lz4")).lower()
    if compressor_name in {"", "none", "null"}:
        return None
    shuffle_name = str(getattr(cfg, "zarr_shuffle", "bitshuffle")).lower()
    shuffle_map = {
        "noshuffle": Blosc.NOSHUFFLE,
        "shuffle": Blosc.SHUFFLE,
        "bitshuffle": Blosc.BITSHUFFLE,
    }
    return Blosc(
        cname=compressor_name,
        clevel=int(getattr(cfg, "zarr_compression_level", 1)),
        shuffle=shuffle_map.get(shuffle_name, Blosc.BITSHUFFLE),
    )


def build_station_day_tasks(event_list, data_dir):
    day_cache = {}
    task_map = {}

    for _, _, event_loc, pick_dict in event_list:
        ot = event_loc[0]
        event_id = dtime2str(ot)
        date_code = f"{ot.year:04d}{ot.month:02d}{ot.day:02d}"
        if date_code not in day_cache:
            day_cache[date_code] = get_data_dict(ot, data_dir)
        data_dict = day_cache[date_code]

        for station, (tp, ts) in pick_dict.items():
            if station not in data_dict:
                continue
            key = (date_code, station)
            if key not in task_map:
                task_map[key] = {
                    "date_code": date_code,
                    "station": station,
                    "stream_paths": data_dict[station],
                    "requests": [],
                }
            task_map[key]["requests"].append(
                {
                    "event_id": event_id,
                    "ot": ot,
                    "tp": tp,
                    "ts": ts,
                }
            )

    tasks = list(task_map.values())
    tasks.sort(key=lambda item: (item["date_code"], item["station"]))
    for task in tasks:
        task["requests"].sort(key=lambda item: (float(item["tp"] - item["ot"]) if item["tp"] != -1 else 1e12))
    return tasks


def _extract_phase_window(traces, starttime, endtime, phase_time, spec, samp_rate, native_rate, freq_band):
    n_channels = len(spec["channels"])
    window = np.zeros((n_channels, spec["data_npts"]), dtype=np.float32)
    if phase_time == -1:
        return window, False

    t0 = phase_time - spec["data_win"][0]
    t1 = phase_time + spec["data_win"][1]
    if t0 < starttime or t1 > endtime:
        return window, False

    start_idx = int(round((t0 - starttime) * samp_rate))
    end_idx = start_idx + spec["data_npts"]
    if start_idx < 0 or end_idx > int((endtime - starttime) * samp_rate) + 1:
        return window, False

    # Match the original full-day filter with a warm-up interval, but only
    # interpolate and filter the samples needed for this phase window.
    filter_start_idx = max(0, start_idx - int(20 * samp_rate))
    target_indices = np.arange(filter_start_idx, end_idx, dtype=np.float64)
    if native_rate == samp_rate:
        segment = traces[spec["channels"], filter_start_idx:end_idx]
    else:
        source_indices = target_indices * (native_rate / samp_rate)
        source_start = max(0, int(np.floor(source_indices[0])) - 8)
        source_end = min(traces.shape[1], int(np.ceil(source_indices[-1])) + 9)
        source_x = np.arange(source_start, source_end, dtype=np.float64)
        segment = np.stack([
            InterpolatedUnivariateSpline(source_x, traces[channel, source_start:source_end], k=3)(source_indices)
            for channel in spec["channels"]
        ])

    freq_min, freq_max = freq_band
    if freq_min and freq_max:
        segment = bandpass(segment, freqmin=freq_min, freqmax=freq_max, df=samp_rate)
    elif freq_min:
        segment = highpass(segment, freq=freq_min, df=samp_rate)
    elif freq_max:
        segment = lowpass(segment, freq=freq_max, df=samp_rate)

    window[:] = segment[:, start_idx - filter_start_idx:]
    return window, True


def process_station_day_task(task, cfg, phase_specs):
    try:
        stream = read(task["stream_paths"][0])
        stream += read(task["stream_paths"][1])
        stream += read(task["stream_paths"][2])
    except Exception:
        return None

    if len(stream) != 3:
        return None

    try:
        native_rate = stream[0].stats.sampling_rate
        stream = preprocess(stream, native_rate, (None, None))
    except Exception:
        return None

    if len(stream) != 3:
        return None

    npts = min(tr.stats.npts for tr in stream)
    traces = np.stack([
        tr.data[:npts].astype(np.float32, copy=False)
        for tr in stream
    ])
    starttime = stream[0].stats.starttime
    endtime = stream[0].stats.endtime

    event_ids = []
    tp_rel = []
    ts_rel = []
    has_p = []
    has_s = []
    p_data = []
    s_data = []

    for request in task["requests"]:
        p_window, valid_p = _extract_phase_window(
            traces=traces,
            starttime=starttime,
            endtime=endtime,
            phase_time=request["tp"],
            spec=phase_specs["P"],
            samp_rate=cfg.samp_rate,
            native_rate=native_rate,
            freq_band=cfg.freq_band,
        )
        s_window, valid_s = _extract_phase_window(
            traces=traces,
            starttime=starttime,
            endtime=endtime,
            phase_time=request["ts"],
            spec=phase_specs["S"],
            samp_rate=cfg.samp_rate,
            native_rate=native_rate,
            freq_band=cfg.freq_band,
        )
        if not (valid_p or valid_s):
            continue

        event_ids.append(request["event_id"])
        tp_rel.append(float(request["tp"] - request["ot"]) if request["tp"] != -1 else -1.0)
        ts_rel.append(float(request["ts"] - request["ot"]) if request["ts"] != -1 else -1.0)
        has_p.append(valid_p)
        has_s.append(valid_s)
        p_data.append(p_window)
        s_data.append(s_window)

    if not event_ids:
        return None

    return {
        "station": task["station"],
        "event_id": np.asarray(event_ids, dtype="U32"),
        "tp_rel": np.asarray(tp_rel, dtype=np.float32),
        "ts_rel": np.asarray(ts_rel, dtype=np.float32),
        "has_p": np.asarray(has_p, dtype=np.uint8),
        "has_s": np.asarray(has_s, dtype=np.uint8),
        "p_data": np.stack(p_data).astype(np.float32, copy=False),
        "s_data": np.stack(s_data).astype(np.float32, copy=False),
    }


def station_day_checkpoint_paths(checkpoint_dir, task_index, task):
    day_dir = os.path.join(checkpoint_dir, task["date_code"])
    stem = f"{int(task_index):08d}"
    return os.path.join(day_dir, f"{stem}.npz"), os.path.join(day_dir, f"{stem}.json")


def checkpoint_is_complete(checkpoint_dir, task_index, task):
    data_path, marker_path = station_day_checkpoint_paths(checkpoint_dir, task_index, task)
    try:
        with open(marker_path, encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("task_index") != int(task_index) or marker.get("station") != task["station"]:
            return False
        if marker.get("status") == "empty":
            return True
        return (
            marker.get("status") == "written"
            and os.path.getsize(data_path) == int(marker.get("size_bytes", -1))
            and int(marker.get("rows", 0)) > 0
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _atomic_write_json(path, value):
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def process_station_day_to_checkpoint(task_item, cfg, phase_specs, checkpoint_dir):
    task_index, task = task_item
    data_path, marker_path = station_day_checkpoint_paths(checkpoint_dir, task_index, task)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    station_batch = process_station_day_task(task, cfg, phase_specs)
    if station_batch is None:
        _atomic_write_json(
            marker_path,
            {"task_index": int(task_index), "station": task["station"], "status": "empty", "rows": 0},
        )
        return int(task_index), "empty", 0

    temp_path = f"{data_path}.{os.getpid()}.tmp"
    with open(temp_path, "wb") as handle:
        np.savez(handle, **station_batch)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, data_path)
    rows = len(station_batch["event_id"])
    _atomic_write_json(
        marker_path,
        {
            "task_index": int(task_index),
            "station": task["station"],
            "status": "written",
            "rows": rows,
            "size_bytes": os.path.getsize(data_path),
        },
    )
    return int(task_index), "written", rows


def load_station_day_checkpoint(path):
    with np.load(path, allow_pickle=False) as stored:
        return {
            "station": str(stored["station"].item()),
            "event_id": stored["event_id"],
            "tp_rel": stored["tp_rel"],
            "ts_rel": stored["ts_rel"],
            "has_p": stored["has_p"],
            "has_s": stored["has_s"],
            "p_data": stored["p_data"],
            "s_data": stored["s_data"],
        }


class StationPhaseWriter:
    def __init__(self, zarr_path, cfg, phase_specs, run_fingerprint=None):
        self.cfg = cfg
        self.phase_specs = phase_specs
        self.chunk_rows = max(1, int(getattr(cfg, "zarr_chunk_rows", 128)))
        self.root = zarr.open_group(zarr_path, mode="w")
        self.root.attrs.update(
            {
                "layout": "per_station_phase_window_v2",
                "sampling_rate": float(cfg.samp_rate),
                "freq_band": list(cfg.freq_band),
                "win_temp_p": list(cfg.win_temp_p),
                "win_temp_s": list(cfg.win_temp_s),
                "dt_thres": list(cfg.dt_thres),
                "chn_p": list(cfg.chn_p),
                "chn_s": list(cfg.chn_s),
                "run_fingerprint": run_fingerprint or "",
            }
        )
        self.stations = self.root.require_group("stations")
        self.compressor = build_compressor(cfg)

    def append_station_batch(self, station_batch):
        station = station_batch["station"]
        grp = self.stations.require_group(station)
        n_add = len(station_batch["event_id"])

        if "event_id" not in grp:
            grp.create_dataset("event_id", shape=(0,), chunks=(self.chunk_rows,), dtype="U32")
            grp.create_dataset("tp_rel", shape=(0,), chunks=(self.chunk_rows,), dtype="f4")
            grp.create_dataset("ts_rel", shape=(0,), chunks=(self.chunk_rows,), dtype="f4")
            grp.create_dataset("has_p", shape=(0,), chunks=(self.chunk_rows,), dtype="u1")
            grp.create_dataset("has_s", shape=(0,), chunks=(self.chunk_rows,), dtype="u1")
            grp.create_dataset(
                "p_data",
                shape=(0, len(self.phase_specs["P"]["channels"]), self.phase_specs["P"]["data_npts"]),
                chunks=(self.chunk_rows, len(self.phase_specs["P"]["channels"]), self.phase_specs["P"]["data_npts"]),
                dtype="f4",
                compressor=self.compressor,
            )
            grp.create_dataset(
                "s_data",
                shape=(0, len(self.phase_specs["S"]["channels"]), self.phase_specs["S"]["data_npts"]),
                chunks=(self.chunk_rows, len(self.phase_specs["S"]["channels"]), self.phase_specs["S"]["data_npts"]),
                dtype="f4",
                compressor=self.compressor,
            )

        n_old = grp["event_id"].shape[0]
        n_new = n_old + n_add
        grp["event_id"].resize((n_new,))
        grp["tp_rel"].resize((n_new,))
        grp["ts_rel"].resize((n_new,))
        grp["has_p"].resize((n_new,))
        grp["has_s"].resize((n_new,))
        grp["p_data"].resize((n_new, len(self.phase_specs["P"]["channels"]), self.phase_specs["P"]["data_npts"]))
        grp["s_data"].resize((n_new, len(self.phase_specs["S"]["channels"]), self.phase_specs["S"]["data_npts"]))

        grp["event_id"][n_old:n_new] = station_batch["event_id"]
        grp["tp_rel"][n_old:n_new] = station_batch["tp_rel"]
        grp["ts_rel"][n_old:n_new] = station_batch["ts_rel"]
        grp["has_p"][n_old:n_new] = station_batch["has_p"]
        grp["has_s"][n_old:n_new] = station_batch["has_s"]
        grp["p_data"][n_old:n_new] = station_batch["p_data"]
        grp["s_data"][n_old:n_new] = station_batch["s_data"]

    def write_station_checkpoints(self, station, checkpoints, total_rows):
        grp = self.stations.require_group(station)
        p_channels = len(self.phase_specs["P"]["channels"])
        s_channels = len(self.phase_specs["S"]["channels"])
        p_npts = self.phase_specs["P"]["data_npts"]
        s_npts = self.phase_specs["S"]["data_npts"]
        grp.create_dataset("event_id", shape=(total_rows,), chunks=(self.chunk_rows,), dtype="U32")
        grp.create_dataset("tp_rel", shape=(total_rows,), chunks=(self.chunk_rows,), dtype="f4")
        grp.create_dataset("ts_rel", shape=(total_rows,), chunks=(self.chunk_rows,), dtype="f4")
        grp.create_dataset("has_p", shape=(total_rows,), chunks=(self.chunk_rows,), dtype="u1")
        grp.create_dataset("has_s", shape=(total_rows,), chunks=(self.chunk_rows,), dtype="u1")
        grp.create_dataset(
            "p_data",
            shape=(total_rows, p_channels, p_npts),
            chunks=(self.chunk_rows, p_channels, p_npts),
            dtype="f4",
            compressor=self.compressor,
        )
        grp.create_dataset(
            "s_data",
            shape=(total_rows, s_channels, s_npts),
            chunks=(self.chunk_rows, s_channels, s_npts),
            dtype="f4",
            compressor=self.compressor,
        )

        buffer = {
            "event_id": np.empty(self.chunk_rows, dtype="U32"),
            "tp_rel": np.empty(self.chunk_rows, dtype=np.float32),
            "ts_rel": np.empty(self.chunk_rows, dtype=np.float32),
            "has_p": np.empty(self.chunk_rows, dtype=np.uint8),
            "has_s": np.empty(self.chunk_rows, dtype=np.uint8),
            "p_data": np.empty((self.chunk_rows, p_channels, p_npts), dtype=np.float32),
            "s_data": np.empty((self.chunk_rows, s_channels, s_npts), dtype=np.float32),
        }
        output_offset = 0
        buffered_rows = 0

        def flush_buffer(n_rows):
            nonlocal output_offset
            end = output_offset + n_rows
            for key in ("event_id", "tp_rel", "ts_rel", "has_p", "has_s", "p_data", "s_data"):
                grp[key][output_offset:end] = buffer[key][:n_rows]
            output_offset = end

        for path, expected_rows in checkpoints:
            batch = load_station_day_checkpoint(path)
            if batch["station"] != station or len(batch["event_id"]) != expected_rows:
                raise RuntimeError(f"Invalid station-day checkpoint for {station}: {path}")
            source_offset = 0
            while source_offset < expected_rows:
                take = min(self.chunk_rows - buffered_rows, expected_rows - source_offset)
                source_slice = slice(source_offset, source_offset + take)
                target_slice = slice(buffered_rows, buffered_rows + take)
                for key in buffer:
                    buffer[key][target_slice] = batch[key][source_slice]
                source_offset += take
                buffered_rows += take
                if buffered_rows == self.chunk_rows:
                    flush_buffer(buffered_rows)
                    buffered_rows = 0

        if buffered_rows:
            flush_buffer(buffered_rows)
        if output_offset != total_rows:
            raise RuntimeError(
                f"Station {station} wrote {output_offset} rows; expected {total_rows}."
            )


def build_station_event_index(root):
    stations_root = root["stations"]
    event_station_map = {}
    for station in tqdm(sorted(stations_root.keys()), desc="Indexing Zarr"):
        grp = stations_root[station]
        event_ids = grp["event_id"][:]
        for row_idx, event_id in enumerate(event_ids):
            event_id = str(event_id)
            if event_id not in event_station_map:
                event_station_map[event_id] = {}
            event_station_map[event_id][station] = row_idx
    return event_station_map


def _utc_timestamp(value):
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return float(timestamp())
    if timestamp is not None:
        return float(timestamp)
    return float(value)


def build_event_records(phase_file, event_station_map, cfg):
    records = []
    for evid, event_name, event_loc, pha_dict_pick in tqdm(iter_fpha(phase_file), desc="Reading phase file"):
        event_rows = event_station_map.get(event_name, {})
        available_stations = [
            station for station in pha_dict_pick.keys() if station in event_rows
        ]
        if len(available_stations) < cfg.num_sta_thres:
            continue
        ot, lat, lon, dep, mag = event_loc
        station_list = sorted(available_stations)
        records.append(
            {
                "evid": str(evid),
                "waveform_id": str(event_name),
                "ot_ts": _utc_timestamp(ot),
                "lat": float(lat),
                "lon": float(lon),
                "dep": float(dep),
                "mag": float(mag),
                "stations": station_list,
                "station_set": set(station_list),
                "station_rows": {station: int(event_rows[station]) for station in station_list},
                "is_temp": int(mag >= cfg.temp_mag and len(station_list) >= cfg.temp_sta),
            }
        )
    return records


def build_candidate_pairs(records, cfg):
    if not records:
        return np.empty((0, 2), dtype=np.int32)

    ot = np.asarray([record["ot_ts"] for record in records], dtype=np.float64)
    lat = np.asarray([record["lat"] for record in records], dtype=np.float64)
    lon = np.asarray([record["lon"] for record in records], dtype=np.float64)
    dep = np.asarray([record["dep"] for record in records], dtype=np.float64)
    is_temp = np.asarray([record["is_temp"] for record in records], dtype=np.uint8)
    station_sets = [record["station_set"] for record in records]

    num_events = len(records)
    chunk_events = max(1, int(getattr(cfg, "pair_chunk_events", 128)))
    index_chunks = [(idx, min(idx + chunk_events, num_events)) for idx in range(0, num_events, chunk_events)]
    num_workers = max(0, int(getattr(cfg, "pair_num_workers", 0)))
    params = {
        "loc_dev_thres": float(cfg.loc_dev_thres),
        "dep_dev_thres": float(cfg.dep_dev_thres),
        "num_sta_thres": int(cfg.num_sta_thres),
        "max_nbr": int(cfg.max_nbr),
        "cal_win": float(cfg.cal_win),
    }

    init_args = (ot, lat, lon, dep, is_temp, station_sets, params)
    pair_chunks = []
    if num_workers <= 1 or len(index_chunks) == 1:
        _init_pair_build_worker(*init_args)
        iterator = map(_build_candidate_pairs_chunk, index_chunks)
        for pair_arr in tqdm(iterator, total=len(index_chunks), desc="Building pairs", unit="chunk"):
            if pair_arr.size:
                pair_chunks.append(pair_arr)
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(processes=num_workers, initializer=_init_pair_build_worker, initargs=init_args) as pool:
            iterator = pool.imap_unordered(_build_candidate_pairs_chunk, index_chunks, chunksize=1)
            for pair_arr in tqdm(iterator, total=len(index_chunks), desc="Building pairs", unit="chunk"):
                if pair_arr.size:
                    pair_chunks.append(pair_arr)

    if not pair_chunks:
        return np.empty((0, 2), dtype=np.int32)

    pair_array = np.concatenate(pair_chunks, axis=0)
    if len(pair_array) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(pair_array, axis=0)


def _init_pair_build_worker(ot, lat, lon, dep, is_temp, station_sets, params):
    global _PAIR_BUILD_OT
    global _PAIR_BUILD_LAT
    global _PAIR_BUILD_LON
    global _PAIR_BUILD_DEP
    global _PAIR_BUILD_IS_TEMP
    global _PAIR_BUILD_STATION_SETS
    global _PAIR_BUILD_PARAMS

    _PAIR_BUILD_OT = ot
    _PAIR_BUILD_LAT = lat
    _PAIR_BUILD_LON = lon
    _PAIR_BUILD_DEP = dep
    _PAIR_BUILD_IS_TEMP = is_temp
    _PAIR_BUILD_STATION_SETS = station_sets
    _PAIR_BUILD_PARAMS = params


def _build_candidate_pairs_chunk(index_range):
    start, stop = index_range
    pair_chunks = []
    for index in range(start, stop):
        pair_arr = _build_candidate_pairs_for_event(index)
        if pair_arr.size:
            pair_chunks.append(pair_arr)
    if not pair_chunks:
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(np.concatenate(pair_chunks, axis=0), axis=0)


def _build_candidate_pairs_for_event(index):
    params = _PAIR_BUILD_PARAMS
    lat = _PAIR_BUILD_LAT[index]
    lon = _PAIR_BUILD_LON[index]
    dep = _PAIR_BUILD_DEP[index]
    ot = _PAIR_BUILD_OT[index]

    cos_lat = np.cos(lat * np.pi / 180.0)
    cond_lat = 111.0 * np.abs(_PAIR_BUILD_LAT - lat) < params["loc_dev_thres"]
    cond_lon = 111.0 * np.abs(_PAIR_BUILD_LON - lon) * cos_lat < params["loc_dev_thres"]
    cond_dep = np.abs(_PAIR_BUILD_DEP - dep) < params["dep_dev_thres"]
    cond_time = np.abs(_PAIR_BUILD_OT - ot) / 86400.0 <= params["cal_win"]
    cond_loc = cond_lat & cond_lon & cond_dep & cond_time
    if _PAIR_BUILD_IS_TEMP[index] != 1:
        cond_loc = cond_loc & (_PAIR_BUILD_IS_TEMP == 1)

    candidate_indices = np.flatnonzero(cond_loc)
    if len(candidate_indices) == 0:
        return np.empty((0, 2), dtype=np.int32)

    sta_ref = _PAIR_BUILD_STATION_SETS[index]
    min_station_count = params["num_sta_thres"]
    valid_indices = [
        int(candidate_idx)
        for candidate_idx in candidate_indices
        if _has_min_common_stations(_PAIR_BUILD_STATION_SETS[candidate_idx], sta_ref, min_station_count)
    ]
    if not valid_indices:
        return np.empty((0, 2), dtype=np.int32)

    valid_indices = np.asarray(valid_indices, dtype=np.int32)
    dist_lat = 111.0 * np.abs(_PAIR_BUILD_LAT[valid_indices] - lat)
    dist_lon = 111.0 * np.abs(_PAIR_BUILD_LON[valid_indices] - lon) * cos_lat
    dist_dep = np.abs(_PAIR_BUILD_DEP[valid_indices] - dep)
    dist_list = np.sqrt(dist_lat**2 + dist_lon**2 + dist_dep**2)
    if len(dist_list) > params["max_nbr"] + 1:
        dist_cut = np.partition(dist_list, params["max_nbr"])[params["max_nbr"]]
    else:
        dist_cut = np.max(dist_list)

    neighbor_indices = valid_indices[dist_list <= dist_cut]
    pairs = np.empty((len(neighbor_indices), 2), dtype=np.int32)
    pair_count = 0
    for event_idx in neighbor_indices:
        event_idx = int(event_idx)
        if event_idx == index:
            continue
        if event_idx < index:
            pairs[pair_count] = (event_idx, index)
        else:
            pairs[pair_count] = (index, event_idx)
        pair_count += 1
    return pairs[:pair_count]


def _has_min_common_stations(stations_a, stations_b, min_count):
    if len(stations_a) > len(stations_b):
        stations_a, stations_b = stations_b, stations_a
    count = 0
    for station in stations_a:
        if station in stations_b:
            count += 1
            if count >= min_count:
                return True
    return False


def _as_unicode_array(values):
    if len(values) == 0:
        return np.asarray([], dtype="U1")
    max_len = max(1, max(len(str(value)) for value in values))
    return np.asarray([str(value) for value in values], dtype=f"U{max_len}")


def build_station_pair_tasks(records, pair_list, event_station_map, sta_dict, cfg):
    pair_array = np.asarray(pair_list, dtype=np.int32)
    if pair_array.size == 0:
        return {}

    num_workers = max(1, int(getattr(cfg, "station_assign_num_workers", 1)))
    chunk_size = max(1, int(getattr(cfg, "station_assign_chunk_pairs", 20000)))
    pair_ranges = [(idx, min(idx + chunk_size, len(pair_array))) for idx in range(0, len(pair_array), chunk_size)]

    station_pairs = defaultdict(list)
    if num_workers == 1 or len(pair_ranges) == 1:
        _init_station_assign_worker(records, event_station_map, sta_dict, cfg, pair_array)
        iterator = map(_build_station_pair_tasks_chunk, pair_ranges)
        for partial_pairs in tqdm(iterator, total=len(pair_ranges), desc="Assigning station pairs"):
            _merge_station_pair_dicts(station_pairs, partial_pairs)
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_station_assign_worker,
            initargs=(records, event_station_map, sta_dict, cfg, pair_array),
        ) as pool:
            iterator = pool.imap_unordered(_build_station_pair_tasks_chunk, pair_ranges, chunksize=1)
            for partial_pairs in tqdm(iterator, total=len(pair_ranges), desc="Assigning station pairs"):
                _merge_station_pair_dicts(station_pairs, partial_pairs)

    station_tasks = {}
    for station, entries in station_pairs.items():
        data_evid = [records[entry[2]]["evid"] for entry in entries]
        temp_evid = [records[entry[3]]["evid"] for entry in entries]
        station_tasks[station] = {
            "row_a": np.asarray([entry[0] for entry in entries], dtype=np.int32),
            "row_b": np.asarray([entry[1] for entry in entries], dtype=np.int32),
            "data_evid": _as_unicode_array(data_evid),
            "temp_evid": _as_unicode_array(temp_evid),
        }
    return station_tasks


def build_and_save_station_pair_tasks(records, pair_list, event_station_map, sta_dict, cfg, task_dir, metadata=None):
    pair_array = np.asarray(pair_list, dtype=np.int32)
    metadata = dict(metadata or {})
    metadata.setdefault("event_ids", [record["evid"] for record in records])
    writer = StationTaskShardWriter(task_dir, metadata=metadata)
    if pair_array.size == 0:
        return writer.finish()

    num_workers = max(1, int(getattr(cfg, "station_assign_num_workers", 1)))
    chunk_size = max(1, int(getattr(cfg, "station_assign_chunk_pairs", 20000)))
    pair_ranges = [(idx, min(idx + chunk_size, len(pair_array))) for idx in range(0, len(pair_array), chunk_size)]

    if num_workers == 1 or len(pair_ranges) == 1:
        _init_station_assign_worker(records, event_station_map, sta_dict, cfg, pair_array, task_dir)
        iterator = map(_build_and_write_station_pair_tasks_chunk, pair_ranges)
        for shard_summary in tqdm(iterator, total=len(pair_ranges), desc="Assigning station pairs"):
            writer.merge_summary(shard_summary)
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_station_assign_worker,
            initargs=(records, event_station_map, sta_dict, cfg, pair_array, task_dir),
        ) as pool:
            iterator = pool.imap_unordered(_build_and_write_station_pair_tasks_chunk, pair_ranges, chunksize=1)
            for shard_summary in tqdm(iterator, total=len(pair_ranges), desc="Assigning station pairs"):
                writer.merge_summary(shard_summary)

    return writer.finish()


def _init_station_assign_worker(records, event_station_map, sta_dict, cfg, pair_array=None, task_dir=None):
    global _STATION_ASSIGN_RECORDS
    global _STATION_ASSIGN_EVENT_MAP
    global _STATION_ASSIGN_STA_DICT
    global _STATION_ASSIGN_CFG
    global _STATION_ASSIGN_PAIR_ARRAY
    global _STATION_ASSIGN_TASK_DIR

    _STATION_ASSIGN_RECORDS = records
    _STATION_ASSIGN_EVENT_MAP = event_station_map
    _STATION_ASSIGN_STA_DICT = sta_dict
    _STATION_ASSIGN_CFG = cfg
    _STATION_ASSIGN_PAIR_ARRAY = pair_array
    _STATION_ASSIGN_TASK_DIR = task_dir


def _build_station_pair_tasks_chunk(pair_chunk):
    station_pairs = defaultdict(list)
    min_station_count = _STATION_ASSIGN_CFG.num_sta_thres
    if isinstance(pair_chunk, tuple) and len(pair_chunk) == 2:
        start, stop = pair_chunk
        pair_iter = _STATION_ASSIGN_PAIR_ARRAY[start:stop]
    else:
        pair_iter = pair_chunk

    for idx_a, idx_b in pair_iter:
        rec_a = _STATION_ASSIGN_RECORDS[idx_a]
        rec_b = _STATION_ASSIGN_RECORDS[idx_b]
        common_stations = [station for station in rec_a["stations"] if station in rec_b["station_set"]]
        if len(common_stations) < min_station_count:
            continue

        valid_entries = []
        for station in common_stations:
            sta_lat, sta_lon = _STATION_ASSIGN_STA_DICT[station][:2]
            dist_a = calc_dist_km([sta_lat, rec_a["lat"]], [sta_lon, rec_a["lon"]])
            dist_b = calc_dist_km([sta_lat, rec_b["lat"]], [sta_lon, rec_b["lon"]])
            if min(dist_a, dist_b) > _STATION_ASSIGN_CFG.dist_thres:
                continue
            if max(dist_a, dist_b) < 1.5:
                continue
            valid_entries.append(
                (
                    station,
                    _station_row_index(rec_a, station),
                    _station_row_index(rec_b, station),
                    int(idx_a),
                    int(idx_b),
                )
            )

        if len(valid_entries) < min_station_count:
            continue

        for station, row_a, row_b, data_evid, temp_evid in valid_entries:
            station_pairs[station].append((row_a, row_b, data_evid, temp_evid))

    return dict(station_pairs)


def _station_row_index(record, station):
    if "station_rows" in record:
        return record["station_rows"][station]
    return _STATION_ASSIGN_EVENT_MAP[record["waveform_id"]][station]


def _build_and_write_station_pair_tasks_chunk(pair_chunk):
    partial_pairs = _build_station_pair_tasks_chunk(pair_chunk)
    if isinstance(pair_chunk, tuple) and len(pair_chunk) == 2:
        shard_tag = f"{pair_chunk[0]:012d}_{pair_chunk[1]:012d}"
    else:
        shard_tag = "legacy_chunk"
    return _write_station_pair_shards(_STATION_ASSIGN_TASK_DIR, partial_pairs, shard_tag)


def _merge_station_pair_dicts(target, source):
    for station, entries in source.items():
        target[station].extend(entries)


def split_station_tasks_for_gpus(station_tasks, gpu_ids):
    if not gpu_ids:
        return {}
    gpu_loads = {gpu_id: 0 for gpu_id in gpu_ids}
    gpu_assignment = {gpu_id: [] for gpu_id in gpu_ids}

    stations = sorted(station_tasks.keys(), key=lambda station: len(station_tasks[station]["row_a"]), reverse=True)
    for station in stations:
        gpu_id = min(gpu_ids, key=lambda item: gpu_loads[item])
        gpu_assignment[gpu_id].append(station)
        gpu_loads[gpu_id] += len(station_tasks[station]["row_a"])
    return gpu_assignment


def split_station_counts_for_gpus(station_counts, gpu_ids):
    if not gpu_ids:
        return {}
    gpu_loads = {gpu_id: 0 for gpu_id in gpu_ids}
    gpu_assignment = {gpu_id: [] for gpu_id in gpu_ids}
    stations = sorted(station_counts.keys(), key=lambda station: station_counts[station], reverse=True)
    for station in stations:
        gpu_id = min(gpu_ids, key=lambda item: gpu_loads[item])
        gpu_assignment[gpu_id].append(station)
        gpu_loads[gpu_id] += int(station_counts[station])
    return gpu_assignment


def _task_manifest_path(task_dir):
    return os.path.join(task_dir, "manifest.json")


def _write_station_pair_shards(task_dir, partial_pairs, shard_tag):
    summary = {
        "stations": {},
        "total_station_pairs": 0,
    }
    if not partial_pairs:
        return summary

    for station_idx, (station, entries) in enumerate(sorted(partial_pairs.items())):
        if not entries:
            continue
        row_a = np.asarray([entry[0] for entry in entries], dtype=np.int32)
        row_b = np.asarray([entry[1] for entry in entries], dtype=np.int32)
        event_idx_a = np.asarray([entry[2] for entry in entries], dtype=np.int32)
        event_idx_b = np.asarray([entry[3] for entry in entries], dtype=np.int32)
        file_name = f"task_{os.getpid()}_{shard_tag}_{station_idx:05d}.npz"
        np.savez(
            os.path.join(task_dir, file_name),
            row_a=row_a,
            row_b=row_b,
            event_idx_a=event_idx_a,
            event_idx_b=event_idx_b,
        )
        num_pairs = int(len(row_a))
        summary["stations"][station] = {
            "files": [{"file": file_name, "num_pairs": num_pairs}],
            "num_pairs": num_pairs,
        }
        summary["total_station_pairs"] += num_pairs

    return summary


class StationTaskShardWriter:
    def __init__(self, task_dir, metadata=None):
        self.task_dir = task_dir
        self.metadata = dict(metadata or {})
        self.station_entries = {}
        self.total_pairs = 0
        self.shard_index = 0
        os.makedirs(task_dir, exist_ok=True)

    def write_partial(self, partial_pairs):
        shard_tag = f"main_{self.shard_index:08d}"
        self.shard_index += 1
        self.merge_summary(_write_station_pair_shards(self.task_dir, partial_pairs, shard_tag))

    def merge_summary(self, shard_summary):
        for station, source_entry in shard_summary.get("stations", {}).items():
            station_entry = self.station_entries.setdefault(station, {"files": [], "num_pairs": 0})
            station_entry["files"].extend(source_entry.get("files", []))
            station_entry["num_pairs"] += int(source_entry.get("num_pairs", 0))
        self.total_pairs += int(shard_summary.get("total_station_pairs", 0))

    def finish(self):
        manifest = dict(self.metadata)
        manifest.update(
            {
                "format": "gpu_ph2cc_station_tasks_v3",
                "num_stations": len(self.station_entries),
                "total_station_pairs": self.total_pairs,
                "stations": self.station_entries,
            }
        )
        manifest_path = _task_manifest_path(self.task_dir)
        tmp_path = f"{manifest_path}.tmp"
        with open(tmp_path, "w") as fout:
            json.dump(manifest, fout, indent=2, sort_keys=True)
        os.replace(tmp_path, manifest_path)
        return manifest


def save_gpu_input_tasks(task_dir, station_tasks, metadata=None):
    os.makedirs(task_dir, exist_ok=True)
    station_entries = {}
    total_pairs = 0

    for station_idx, station in enumerate(tqdm(sorted(station_tasks.keys()), desc="Writing GPU input")):
        task = station_tasks[station]
        row_a = np.asarray(task["row_a"], dtype=np.int32)
        row_b = np.asarray(task["row_b"], dtype=np.int32)
        data_evid = _as_unicode_array(task["data_evid"])
        temp_evid = _as_unicode_array(task["temp_evid"])
        file_name = f"station_{station_idx:05d}.npz"
        np.savez(
            os.path.join(task_dir, file_name),
            row_a=row_a,
            row_b=row_b,
            data_evid=data_evid,
            temp_evid=temp_evid,
        )
        station_entries[station] = {
            "file": file_name,
            "num_pairs": int(len(row_a)),
        }
        total_pairs += int(len(row_a))

    manifest = dict(metadata or {})
    manifest.update(
        {
            "format": "gpu_ph2cc_station_tasks_v1",
            "num_stations": len(station_entries),
            "total_station_pairs": total_pairs,
            "stations": station_entries,
        }
    )
    with open(_task_manifest_path(task_dir), "w") as fout:
        json.dump(manifest, fout, indent=2, sort_keys=True)
    return manifest


def load_gpu_input_manifest(task_dir):
    with open(_task_manifest_path(task_dir)) as fin:
        manifest = json.load(fin)
    supported_formats = {
        "gpu_ph2cc_station_tasks_v1",
        "gpu_ph2cc_station_tasks_v2",
        "gpu_ph2cc_station_tasks_v3",
    }
    if manifest.get("format") not in supported_formats:
        raise RuntimeError(f"Unexpected GPU input manifest format in {_task_manifest_path(task_dir)}")
    return manifest


def _load_station_pair_task_file(task_path):
    with np.load(task_path, allow_pickle=False) as data:
        task = {
            "row_a": data["row_a"].astype(np.int32, copy=False),
            "row_b": data["row_b"].astype(np.int32, copy=False),
        }
        if "event_idx_a" in data:
            task["event_idx_a"] = data["event_idx_a"].astype(np.int32, copy=False)
            task["event_idx_b"] = data["event_idx_b"].astype(np.int32, copy=False)
        else:
            task["data_evid"] = data["data_evid"]
            task["temp_evid"] = data["temp_evid"]
        return task


def iter_station_pair_tasks(task_dir, station, manifest=None):
    if manifest is None:
        manifest = load_gpu_input_manifest(task_dir)
    try:
        station_info = manifest["stations"][station]
    except KeyError as exc:
        raise KeyError(f"Station {station} is missing from GPU input manifest {task_dir}") from exc

    if "files" in station_info:
        for file_info in station_info["files"]:
            yield _load_station_pair_task_file(os.path.join(task_dir, file_info["file"]))
    else:
        yield _load_station_pair_task_file(os.path.join(task_dir, station_info["file"]))


def load_station_pair_task(task_dir, station, manifest=None):
    task_chunks = list(iter_station_pair_tasks(task_dir, station, manifest))
    if len(task_chunks) == 1:
        return task_chunks[0]
    if not task_chunks:
        return {
            "row_a": np.asarray([], dtype=np.int32),
            "row_b": np.asarray([], dtype=np.int32),
            "data_evid": np.asarray([], dtype="U1"),
            "temp_evid": np.asarray([], dtype="U1"),
        }
    return {
        "row_a": np.concatenate([task["row_a"] for task in task_chunks]),
        "row_b": np.concatenate([task["row_b"] for task in task_chunks]),
        **_concat_pair_identity_arrays(task_chunks),
    }


def _concat_pair_identity_arrays(task_chunks):
    if "event_idx_a" in task_chunks[0]:
        return {
            "event_idx_a": np.concatenate([task["event_idx_a"] for task in task_chunks]),
            "event_idx_b": np.concatenate([task["event_idx_b"] for task in task_chunks]),
        }
    return {
        "data_evid": np.concatenate([task["data_evid"] for task in task_chunks]),
        "temp_evid": np.concatenate([task["temp_evid"] for task in task_chunks]),
    }


def get_station_pair_counts(manifest):
    return {station: int(info["num_pairs"]) for station, info in manifest.get("stations", {}).items()}


def _station_arrays_from_group(station_group):
    return {
        "tp_rel": station_group["tp_rel"][:].astype(np.float32, copy=False),
        "ts_rel": station_group["ts_rel"][:].astype(np.float32, copy=False),
        "has_p": station_group["has_p"][:].astype(bool, copy=False),
        "has_s": station_group["has_s"][:].astype(bool, copy=False),
        "p_data": station_group["p_data"][:].astype(np.float32, copy=False),
        "s_data": station_group["s_data"][:].astype(np.float32, copy=False),
    }


def _device_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _to_device(array, device, dtype):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, dtype=dtype, non_blocking=(device.type == "cuda"))


def _to_device_once(array, device, dtype):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(device=device, dtype=dtype)


def _station_arrays_to_device(station_arrays, device, cfg):
    compute_dtype = _device_dtype(getattr(cfg, "gpu_compute_dtype", "float32"))
    try:
        return {
            "_device_preloaded": True,
            "tp_rel": _to_device_once(station_arrays["tp_rel"], device=device, dtype=torch.float32),
            "ts_rel": _to_device_once(station_arrays["ts_rel"], device=device, dtype=torch.float32),
            "has_p": _to_device_once(station_arrays["has_p"], device=device, dtype=torch.bool),
            "has_s": _to_device_once(station_arrays["has_s"], device=device, dtype=torch.bool),
            "p_data": _to_device_once(station_arrays["p_data"], device=device, dtype=compute_dtype),
            "s_data": _to_device_once(station_arrays["s_data"], device=device, dtype=compute_dtype),
        }
    except Exception:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise


def _is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _normalized_xcorr(data_windows, temp_windows):
    batch_size, channels, temp_len = temp_windows.shape
    if data_windows.dtype == torch.float16:
        # Keep half-precision reductions and convolution below fp16's range.
        scale = torch.maximum(
            data_windows.abs().amax(dim=-1, keepdim=True),
            temp_windows.abs().amax(dim=-1, keepdim=True),
        ).clamp_min(torch.finfo(torch.float16).tiny)
        data_windows = data_windows / scale
        temp_windows = temp_windows / scale
    data_windows = data_windows.contiguous()
    temp_windows = temp_windows.contiguous()

    # ===============================
    # 1️⃣ 计算 sliding mean（data）
    # ===============================
    data_cum = F.pad(torch.cumsum(data_windows, dim=-1), (1, 0))
    data_mean = (data_cum[..., temp_len:] - data_cum[..., :-temp_len]) / temp_len

    # ===============================
    # 2️⃣ template 去均值
    # ===============================
    temp_mean = temp_windows.mean(dim=-1, keepdim=True)
    temp_zero = temp_windows - temp_mean

    # ===============================
    # 3️⃣ 原始互相关（用去均值 template）
    # ===============================
    raw_cc = F.conv1d(
        data_windows.reshape(1, batch_size * channels, data_windows.shape[-1]),
        temp_zero.reshape(batch_size * channels, 1, temp_len),
        groups=batch_size * channels,
    ).reshape(batch_size, channels, -1)

    # 去掉 conv1d 对齐产生的第一个点（和你原来一致）
    raw_cc = raw_cc[..., 1:]

    # ===============================
    # 4️⃣ 修正 numerator（减去 data_mean * sum(temp)）
    # ===============================
    # 因为：
    # sum((x - μx)(t - μt)) =
    # sum(x * t_zero) - μx * sum(t_zero)
    temp_zero_sum = temp_zero.sum(dim=-1, keepdim=True)
    raw_cc = raw_cc - data_mean[..., 1:] * temp_zero_sum

    # ===============================
    # 5️⃣ data 方差（滑动）
    # ===============================
    data_sq_cum = F.pad(torch.cumsum(data_windows * data_windows, dim=-1), (1, 0))
    data_sq = data_sq_cum[..., temp_len:] - data_sq_cum[..., :-temp_len]

    # Var(x) = E[x^2] - μ^2
    data_var = data_sq / temp_len - data_mean**2
    data_std = torch.sqrt(torch.clamp(data_var, min=0.0))

    # 对齐
    data_std = data_std[..., 1:]

    # ===============================
    # 6️⃣ template 方差
    # ===============================
    temp_var = (temp_zero * temp_zero).sum(dim=-1, keepdim=True) / temp_len
    temp_std = torch.sqrt(torch.clamp(temp_var, min=0.0))

    # ===============================
    # 7️⃣ ZNCC
    # ===============================
    cc = raw_cc / (data_std * temp_std * temp_len)

    cc[~torch.isfinite(cc)] = 0.0

    return cc.mean(dim=1)


def _scc_metrics(cc, cfg):
    """Return the absolute primary peak and the enabled CC/SCC quality mask."""
    abs_cc = torch.abs(cc)
    cc_max, lag_idx = torch.max(abs_cc, dim=-1)
    quality_pass = torch.ones_like(cc_max, dtype=torch.bool)

    cc_thres = getattr(cfg, "cc_thres", None)
    if cc_thres is not None:
        quality_pass &= cc_max >= float(cc_thres)

    scc_thres = getattr(cfg, "scc_thres", None)
    if scc_thres is not None:
        separation_samples = max(
            1,
            int(round(float(cfg.secondary_peak_separation) * float(cfg.samp_rate))),
        )
        lag_axis = torch.arange(cc.shape[-1], device=cc.device)[None, :]
        background_mask = torch.abs(lag_axis - lag_idx[:, None]) > separation_samples
        background = abs_cc.masked_fill(~background_mask, torch.nan)
        background_median = torch.nanmedian(background, dim=-1).values
        background_mad = torch.nanmedian(
            torch.abs(background - background_median[:, None]), dim=-1
        ).values
        background_mad = torch.clamp(
            background_mad, min=torch.finfo(background_mad.dtype).eps
        )
        scc = (cc_max - background_median) / background_mad
        quality_pass &= scc >= float(scc_thres)

    return cc_max, lag_idx, quality_pass


def compute_phase_batch(station_arrays, pair_task, batch_slice, phase_name, phase_specs, cfg, device):
    if station_arrays.get("_device_preloaded", False):
        return compute_phase_batch_preloaded(station_arrays, pair_task, batch_slice, phase_name, phase_specs, cfg, device)
    return compute_phase_batch_cpu(station_arrays, pair_task, batch_slice, phase_name, phase_specs, cfg, device)


def compute_phase_batch_preloaded(station_arrays, pair_task, batch_slice, phase_name, phase_specs, cfg, device):
    phase_spec = phase_specs[phase_name]
    row_a = torch.as_tensor(pair_task["row_a"][batch_slice], dtype=torch.long, device=device)
    row_b = torch.as_tensor(pair_task["row_b"][batch_slice], dtype=torch.long, device=device)
    if phase_name == "P":
        valid = station_arrays["has_p"].index_select(0, row_a) & station_arrays["has_p"].index_select(0, row_b)
        phase_rel = station_arrays["tp_rel"]
        phase_store = station_arrays["p_data"]
    else:
        valid = station_arrays["has_s"].index_select(0, row_a) & station_arrays["has_s"].index_select(0, row_b)
        phase_rel = station_arrays["ts_rel"]
        phase_store = station_arrays["s_data"]

    if not torch.any(valid):
        return None

    batch_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    row_a_valid = row_a.index_select(0, batch_indices)
    row_b_valid = row_b.index_select(0, batch_indices)

    data_windows = phase_store.index_select(0, row_a_valid)
    temp_source = phase_store.index_select(0, row_b_valid)
    temp_windows = temp_source[..., phase_spec["temp_start"] : phase_spec["temp_end"]]

    cc = _normalized_xcorr(data_windows, temp_windows)
    cc_max, lag_idx, quality_pass = _scc_metrics(cc, cfg)

    dt = (
        phase_rel.index_select(0, row_a_valid)
        + phase_spec["tt_shift"]
        + lag_idx.to(torch.float32) / cfg.samp_rate
        - phase_rel.index_select(0, row_b_valid)
    )

    keep = quality_pass & (torch.abs(dt) <= phase_spec["dt_limit"])
    if not torch.any(keep):
        return None

    return {
        "indices": batch_indices[keep].detach().cpu().numpy(),
        "dt": dt[keep].detach().cpu().numpy(),
        "cc": cc_max[keep].to(torch.float32).detach().cpu().numpy(),
    }


def compute_phase_batch_cpu(station_arrays, pair_task, batch_slice, phase_name, phase_specs, cfg, device):
    phase_spec = phase_specs[phase_name]
    row_a = pair_task["row_a"][batch_slice]
    row_b = pair_task["row_b"][batch_slice]
    if phase_name == "P":
        valid = station_arrays["has_p"][row_a] & station_arrays["has_p"][row_b]
        phase_rel = station_arrays["tp_rel"]
        phase_store = station_arrays["p_data"]
    else:
        valid = station_arrays["has_s"][row_a] & station_arrays["has_s"][row_b]
        phase_rel = station_arrays["ts_rel"]
        phase_store = station_arrays["s_data"]

    if not np.any(valid):
        return None

    batch_indices = np.nonzero(valid)[0]
    row_a_valid = row_a[valid]
    row_b_valid = row_b[valid]
    compute_dtype = _device_dtype(getattr(cfg, "gpu_compute_dtype", "float32"))

    data_windows = _to_device(phase_store[row_a_valid], device=device, dtype=compute_dtype)
    temp_source = _to_device(phase_store[row_b_valid], device=device, dtype=compute_dtype)
    temp_windows = temp_source[..., phase_spec["temp_start"] : phase_spec["temp_end"]]

    cc = _normalized_xcorr(data_windows, temp_windows)
    cc_max, lag_idx, quality_pass = _scc_metrics(cc, cfg)

    dt = (
        _to_device(phase_rel[row_a_valid], device=device, dtype=torch.float32)
        + phase_spec["tt_shift"]
        + lag_idx.to(torch.float32) / cfg.samp_rate
        - _to_device(phase_rel[row_b_valid], device=device, dtype=torch.float32)
    )

    keep = quality_pass & (torch.abs(dt) <= phase_spec["dt_limit"])
    if not torch.any(keep):
        return None

    keep_indices = batch_indices[keep.detach().cpu().numpy()]
    return {
        "indices": keep_indices,
        "dt": dt[keep].detach().cpu().numpy(),
        "cc": cc_max[keep].to(torch.float32).detach().cpu().numpy(),
    }


def _encode_decimal6(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Observation values must be finite.")
    magnitude = np.rint(np.abs(values) * 1_000_000.0)
    if np.any(magnitude > np.iinfo(np.int32).max):
        raise ValueError("Observation value exceeds the native record range.")
    encoded = magnitude.astype(np.uint32)
    encoded |= np.signbit(values).astype(np.uint32) << np.uint32(31)
    return encoded


def _write_native_phase_records(
    fout,
    pair_task,
    start,
    result,
    event_ids_numeric,
    station_id,
    phase_id,
):
    if result is None:
        return 0

    pair_indices = start + np.asarray(result["indices"], dtype=np.int64)
    if "event_idx_a" in pair_task:
        if event_ids_numeric is None:
            raise RuntimeError("Prepared task uses event indices, but manifest is missing numeric event_ids.")
        event_a = event_ids_numeric[pair_task["event_idx_a"][pair_indices]]
        event_b = event_ids_numeric[pair_task["event_idx_b"][pair_indices]]
    else:
        event_a = np.asarray(pair_task["data_evid"][pair_indices], dtype=np.uint64)
        event_b = np.asarray(pair_task["temp_evid"][pair_indices], dtype=np.uint64)

    event_limit = np.uint64(1 << _NATIVE_EVENT_BITS)
    if np.any(event_a >= event_limit) or np.any(event_b >= event_limit):
        raise ValueError(f"Event ids must be non-negative integers below {int(event_limit)}.")

    records = np.empty(len(pair_indices), dtype=_NATIVE_RECORD_DTYPE)
    pair_key = (event_a << np.uint64(_NATIVE_EVENT_BITS)) | event_b
    records["key"] = (
        (pair_key << np.uint64(_NATIVE_STATION_PHASE_BITS))
        | np.uint64(station_id << 1)
        | np.uint64(phase_id)
    )
    records["dt"] = _encode_decimal6(result["dt"])
    records["cc"] = _encode_decimal6(result["cc"])
    records.tofile(fout)
    return len(records)


def run_station_gpu_worker(gpu_id, assigned_stations, station_tasks, zarr_path, output_prefix, cfg):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    batch_size = max(1, int(getattr(cfg, "gpu_station_pair_batch_size", 32768)))
    log_every_n_stations = max(0, int(getattr(cfg, "gpu_log_every_n_stations", 0)))
    preload_station_data = bool(getattr(cfg, "gpu_preload_station_data", True))
    preload_fallback_to_cpu = bool(getattr(cfg, "gpu_preload_fallback_to_cpu", True))
    phase_specs = get_phase_specs(cfg)
    task_manifest = None
    event_ids = None
    if isinstance(station_tasks, (str, os.PathLike)):
        task_manifest = load_gpu_input_manifest(station_tasks)
        event_ids = task_manifest.get("event_ids")
    event_ids_numeric = None
    if event_ids is not None:
        try:
            event_ids_numeric = np.asarray(event_ids, dtype=np.uint64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Native observation output requires numeric event ids.") from exc

    station_file = cfg.fsta
    if not os.path.isabs(station_file):
        station_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), station_file)
    station_names = sorted(read_fsta(station_file))
    if len(station_names) > (1 << (_NATIVE_STATION_PHASE_BITS - 1)):
        raise ValueError("Native observation output supports at most 1024 stations.")
    station_to_id = {station: idx for idx, station in enumerate(station_names)}

    zroot = zarr.open_group(zarr_path, mode="r")
    stations_root = zroot["stations"]
    part_path = f"{output_prefix}.gpu{gpu_id}.obsbin"
    observation_count = 0

    with open(part_path, "wb") as fout:
        iterator = tqdm(assigned_stations, position=gpu_id, desc=f"GPU {gpu_id}", leave=False)
        for station_idx, station in enumerate(iterator, start=1):
            try:
                station_id = station_to_id[station]
            except KeyError as exc:
                raise KeyError(f"Station {station} is missing from {station_file}.") from exc
            station_arrays = _station_arrays_from_group(stations_root[station])
            station_data_on_gpu = False
            if preload_station_data:
                try:
                    station_arrays_gpu = _station_arrays_to_device(station_arrays, device=device, cfg=cfg)
                    del station_arrays
                    station_arrays = station_arrays_gpu
                    station_data_on_gpu = True
                except Exception as exc:
                    if not (_is_cuda_oom(exc) and preload_fallback_to_cpu):
                        raise
                    torch.cuda.empty_cache()
                    print(
                        f"[GPU {gpu_id}] station {station} did not fit in GPU memory; "
                        "falling back to CPU batch transfers."
                    )

            if task_manifest is None:
                pair_task_iter = (station_tasks[station],)
                station_pair_count = len(station_tasks[station]["row_a"])
            else:
                pair_task_iter = iter_station_pair_tasks(station_tasks, station, task_manifest)
                station_pair_count = int(task_manifest["stations"][station]["num_pairs"])
            if log_every_n_stations and station_idx % log_every_n_stations == 0:
                print(
                    f"[GPU {gpu_id}] station {station_idx}/{len(assigned_stations)} "
                    f"{station}: {station_pair_count} pairs, "
                    f"station_data={'gpu' if station_data_on_gpu else 'cpu'}"
                )

            for pair_task in pair_task_iter:
                num_pairs = len(pair_task["row_a"])
                for start in range(0, num_pairs, batch_size):
                    stop = min(start + batch_size, num_pairs)
                    batch_slice = slice(start, stop)

                    p_result = compute_phase_batch(
                        station_arrays=station_arrays,
                        pair_task=pair_task,
                        batch_slice=batch_slice,
                        phase_name="P",
                        phase_specs=phase_specs,
                        cfg=cfg,
                        device=device,
                    )
                    observation_count += _write_native_phase_records(
                        fout=fout,
                        pair_task=pair_task,
                        start=start,
                        result=p_result,
                        event_ids_numeric=event_ids_numeric,
                        station_id=station_id,
                        phase_id=0,
                    )

                    s_result = compute_phase_batch(
                        station_arrays=station_arrays,
                        pair_task=pair_task,
                        batch_slice=batch_slice,
                        phase_name="S",
                        phase_specs=phase_specs,
                        cfg=cfg,
                        device=device,
                    )
                    observation_count += _write_native_phase_records(
                        fout=fout,
                        pair_task=pair_task,
                        start=start,
                        result=s_result,
                        event_ids_numeric=event_ids_numeric,
                        station_id=station_id,
                        phase_id=1,
                    )

            del station_arrays
    print(
        f"[GPU {gpu_id}] wrote {observation_count} native observations "
        f"({os.path.getsize(part_path) / 1024**3:.2f} GiB) to {part_path}"
    )


def validate_phase_store(root):
    layout = str(root.attrs.get("layout", ""))
    if layout == "per_station_phase_window_v2":
        return

    if "stations" in root:
        station_names = sorted(root["stations"].keys())
        if station_names:
            first_station = root["stations"][station_names[0]]
            keys = set(first_station.keys())
            required = {"event_id", "tp_rel", "ts_rel", "has_p", "has_s", "p_data", "s_data"}
            if required.issubset(keys):
                root.attrs["layout"] = "per_station_phase_window_v2"
                return
            detected_keys = sorted(keys)
        else:
            detected_keys = []
    else:
        detected_keys = []

    raise RuntimeError(
        "Unexpected Zarr layout: "
        f"{layout or '<missing>'}. "
        f"Detected station datasets: {detected_keys or '<none>'}. "
        "This gpu_ph2cc.py expects the new phase-window store "
        "with layout='per_station_phase_window_v2'. "
        "Please rerun gpu_cut_events.py to rebuild the same --zarr-path "
        "that gpu_ph2cc.py is reading."
    )
