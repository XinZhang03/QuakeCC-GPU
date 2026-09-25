#!/usr/bin/env python3
"""Run multi-GPU dt.cc calculation with compact native observation merging."""

import argparse
import os
import sys
import subprocess
import warnings

import torch
import torch.multiprocessing as mp
import zarr

import config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from gpu_utils import (
    get_station_pair_counts,
    load_gpu_input_manifest,
    run_station_gpu_worker,
    split_station_counts_for_gpus,
    validate_phase_store,
)

warnings.filterwarnings("ignore")

cfg = config.Config()
torch.multiprocessing.set_sharing_strategy("file_system")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NATIVE_MERGE_SOURCE = os.path.join(SRC_DIR, "merge_obs_v2.cpp")
NATIVE_MERGE_BINARY = os.path.join(SCRIPT_DIR, "bin", "merge_obs")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU dt.cc from prepared GPU input tasks.")
    parser.add_argument("--task-dir", default=cfg.gpu_input_dir, help="Directory created by 2_prepare_gpu_input.py.")
    parser.add_argument("--fsta", default=None, help="Station file; defaults to the path in the task manifest.")
    parser.add_argument(
        "--zarr-path",
        default=None,
        help="Override the Zarr path saved in the task manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg.output_dir,
        help="Directory for dt_all.cc and compact per-GPU observation files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Full output dt.cc path. Overrides --output-dir when provided.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids, e.g. 0,1,2. Default: all visible GPUs.",
    )
    parser.add_argument("--merge-threads", type=int, default=30, help="CPU threads for native merge.")
    parser.add_argument(
        "--merge-write-threads",
        type=int,
        default=1,
        help="Final output writer threads; one is best for sequential disks.",
    )
    parser.add_argument(
        "--merge-memory-limit-gib",
        type=float,
        default=300.0,
        help="Safety limit for native merge memory estimates.",
    )
    return parser.parse_args()


def resolve_output_path(args):
    if args.output:
        return args.output
    return os.path.join(args.output_dir, "dt_all.cc")


def parse_gpu_ids(gpu_text):
    if gpu_text:
        return [int(item) for item in gpu_text.split(",") if item.strip()]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    return list(range(torch.cuda.device_count()))


def build_native_merger():
    needs_build = not os.path.exists(NATIVE_MERGE_BINARY)
    if not needs_build:
        needs_build = os.path.getmtime(NATIVE_MERGE_BINARY) < os.path.getmtime(NATIVE_MERGE_SOURCE)
    if needs_build:
        print("Compiling native observation merger")
        subprocess.run(
            [
                "g++",
                "-O3",
                "-march=native",
                "-std=c++17",
                "-fopenmp",
                "-DNDEBUG",
                NATIVE_MERGE_SOURCE,
                "-o",
                NATIVE_MERGE_BINARY,
            ],
            check=True,
        )
    return NATIVE_MERGE_BINARY


def merge_native_observation_files(output_path, gpu_ids, args):
    part_paths = [f"{output_path}.gpu{gpu_id}.obsbin" for gpu_id in gpu_ids]
    missing = [path for path in part_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing native GPU observation files: {missing}")
    invalid = [path for path in part_paths if os.path.getsize(path) % 16 != 0]
    if invalid:
        raise RuntimeError(f"Invalid native observation file sizes: {invalid}")

    station_file = cfg.fsta
    if not os.path.isabs(station_file):
        station_file = os.path.join(SCRIPT_DIR, station_file)
    command = [
        build_native_merger(),
        "--binary-input",
        "--station-file",
        station_file,
        "--output",
        output_path,
        "--threads",
        str(max(1, args.merge_threads)),
        "--write-threads",
        str(max(1, args.merge_write_threads)),
        "--num-sta-thres",
        str(cfg.num_sta_thres),
        "--memory-limit-gib",
        str(args.merge_memory_limit_gib),
    ]
    for path in part_paths:
        command.extend(["--input", path])

    total_bytes = sum(os.path.getsize(path) for path in part_paths)
    print(
        f"Merging {total_bytes / 1024**3:.2f} GiB of native observations "
        f"with {max(1, args.merge_threads)} CPU threads"
    )
    env = os.environ.copy()
    env.setdefault("OMP_PLACES", "cores")
    env.setdefault("OMP_PROC_BIND", "spread")
    subprocess.run(command, check=True, env=env)


def main():
    args = parse_args()
    manifest = load_gpu_input_manifest(args.task_dir)
    zarr_path = args.zarr_path or manifest.get("zarr_path") or cfg.zarr_root
    cfg.fsta = args.fsta or manifest.get("fsta") or cfg.fsta
    output_path = resolve_output_path(args)
    station_counts = get_station_pair_counts(manifest)

    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("No GPU ids were provided.")

    gpu_assignment = split_station_counts_for_gpus(station_counts, gpu_ids)

    print(f"Prepared input: {args.task_dir}")
    print(f"Opening Zarr store: {zarr_path}")
    print(f"Output path: {output_path}")
    root = zarr.open_group(zarr_path, mode="r")
    validate_phase_store(root)
    print(f"Usable events: {manifest.get('num_usable_records', '<unknown>')}")
    print(f"Candidate pairs: {manifest.get('num_candidate_pairs', '<unknown>')}")
    print(f"Active stations: {manifest.get('num_stations', len(station_counts))}")
    print(f"Station-pair tasks: {manifest.get('total_station_pairs', sum(station_counts.values()))}")
    cc_label = f"CC >= {cfg.cc_thres:g}" if cfg.cc_thres is not None else "CC threshold disabled"
    scc_label = f"SCC >= {cfg.scc_thres:g}" if cfg.scc_thres is not None else "SCC threshold disabled"
    print(
        f"Phase QC: {cc_label}; {scc_label}; "
        f"background peak exclusion = +/-{cfg.secondary_peak_separation:g} s"
    )
    print(f"Using GPUs: {gpu_ids}")
    for gpu_id in gpu_ids:
        name = torch.cuda.get_device_name(gpu_id)
        print(f"GPU {gpu_id}: {name}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not station_counts:
        open(output_path, "w").close()
        print(f"No station tasks were generated. Wrote empty file to {output_path}")
        return

    build_native_merger()

    mp.set_start_method("spawn", force=True)
    workers = []
    active_gpu_ids = []
    for gpu_id in gpu_ids:
        assigned_stations = gpu_assignment.get(gpu_id, [])
        if not assigned_stations:
            continue
        process = mp.Process(
            target=run_station_gpu_worker,
            args=(
                gpu_id,
                assigned_stations,
                args.task_dir,
                zarr_path,
                output_path,
                cfg,
            ),
        )
        process.start()
        workers.append(process)
        active_gpu_ids.append(gpu_id)

    for process in workers:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"GPU worker failed with exit code {process.exitcode}.")

    merge_native_observation_files(output_path, active_gpu_ids, args)
    print(f"Finished writing {output_path}")


if __name__ == "__main__":
    main()
