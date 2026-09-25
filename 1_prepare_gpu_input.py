#!/usr/bin/env python3
"""Build reusable GPU input tasks from the Zarr store and phase file."""

import argparse
from datetime import datetime, timezone
import gc
import os
import sys
import shutil
import warnings

import zarr

import config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from gpu_utils import (
    build_and_save_station_pair_tasks,
    build_candidate_pairs,
    build_event_records,
    build_station_event_index,
    validate_phase_store,
)
from reader import read_fsta

warnings.filterwarnings("ignore")

cfg = config.Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Index Zarr, build event pairs, and write reusable GPU input.")
    parser.add_argument("--phase-file", default=cfg.fpha_name, help="Input phase file.")
    parser.add_argument("--fsta", default=cfg.fsta, help="Station file path.")
    parser.add_argument("--zarr-path", default=cfg.zarr_root, help="Phase-window Zarr store.")
    parser.add_argument("--task-dir", default=cfg.gpu_input_dir, help="Directory for prepared GPU input tasks.")
    parser.add_argument(
        "--clean-task-dir",
        action="store_true",
        help="Remove an existing task directory before writing new prepared GPU input.",
    )
    parser.add_argument(
        "--pair-workers",
        type=int,
        default=4,
        help="Number of workers for candidate-pair building.",
    )
    parser.add_argument(
        "--pair-chunk-events",
        type=int,
        default=128,
        help="Number of events processed per pair-building task chunk.",
    )
    parser.add_argument(
        "--station-workers",
        type=int,
        default=4,
        help="Number of workers for assigning pairs to stations.",
    )
    parser.add_argument(
        "--station-chunk-pairs",
        type=int,
        default=100000,
        help="Number of candidate pairs per station-assignment chunk.",
    )
    return parser.parse_args()


def config_snapshot(cfg):
    keys = [
        "cc_thres",
        "scc_thres",
        "secondary_peak_separation",
        "loc_dev_thres",
        "dep_dev_thres",
        "dist_thres",
        "num_sta_thres",
        "max_nbr",
        "temp_mag",
        "temp_sta",
        "cal_win",
        "win_temp_p",
        "win_temp_s",
        "freq_band",
        "dt_thres",
        "chn_p",
        "chn_s",
        "samp_rate",
        "gpu_station_pair_batch_size",
        "gpu_compute_dtype",
    ]
    return {key: getattr(cfg, key) for key in keys}


def main():
    args = parse_args()
    manifest_path = os.path.join(args.task_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.clean_task_dir:
        raise FileExistsError(f"Prepared tasks exist: {manifest_path}; pass --clean-task-dir to replace them")
    cfg.pair_num_workers = args.pair_workers
    cfg.pair_chunk_events = args.pair_chunk_events
    cfg.station_assign_num_workers = args.station_workers
    cfg.station_assign_chunk_pairs = args.station_chunk_pairs

    task_parent = os.path.dirname(args.task_dir)
    if task_parent:
        os.makedirs(task_parent, exist_ok=True)
    if args.clean_task_dir and os.path.exists(args.task_dir):
        shutil.rmtree(args.task_dir)

    print(f"Opening Zarr store: {args.zarr_path}")
    root = zarr.open_group(args.zarr_path, mode="r")
    validate_phase_store(root)

    event_station_map = build_station_event_index(root)
    records = build_event_records(args.phase_file, event_station_map, cfg)
    pair_list = build_candidate_pairs(records, cfg)
    candidate_pair_count = int(len(pair_list))
    sta_dict = read_fsta(args.fsta)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_file": args.phase_file,
        "fsta": args.fsta,
        "zarr_path": args.zarr_path,
        "num_indexed_events": len(event_station_map),
        "num_usable_records": len(records),
        "num_candidate_pairs": candidate_pair_count,
        "config": config_snapshot(cfg),
    }

    del event_station_map
    gc.collect()

    manifest = build_and_save_station_pair_tasks(
        records=records,
        pair_list=pair_list,
        event_station_map=None,
        sta_dict=sta_dict,
        cfg=cfg,
        task_dir=args.task_dir,
        metadata=metadata,
    )

    del pair_list
    del records
    del sta_dict
    gc.collect()

    print(f"Usable events: {metadata['num_usable_records']}")
    print(f"Candidate pairs: {metadata['num_candidate_pairs']}")
    print(f"Active stations: {manifest['num_stations']}")
    print(f"Station-pair tasks: {manifest['total_station_pairs']}")
    print(f"Wrote GPU input to {args.task_dir}")


if __name__ == "__main__":
    main()
