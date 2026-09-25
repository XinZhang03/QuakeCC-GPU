#!/usr/bin/env python3
"""Cut phase-centered waveform windows once and store them in Zarr."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
import sys
import shutil
import warnings
from functools import partial
import multiprocessing as mp

from tqdm import tqdm

import config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from gpu_utils import (
    StationPhaseWriter,
    build_station_day_tasks,
    checkpoint_is_complete,
    get_phase_specs,
    process_station_day_to_checkpoint,
    station_day_checkpoint_paths,
)
from reader import read_fpha

warnings.filterwarnings("ignore")

cfg = config.Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Cut phase windows into a fast Zarr store.")
    parser.add_argument("--fpha", default=cfg.fpha_name, help="Input phase file.")
    parser.add_argument("--data-dir", default=cfg.data_dir, help="Continuous waveform root.")
    parser.add_argument("--zarr-path", default=cfg.zarr_root, help="Output Zarr path.")
    parser.add_argument(
        "--checkpoint-dir",
        default=f"{cfg.zarr_root}.checkpoints",
        help="Directory for resumable station-day output.",
    )
    parser.add_argument("--restart", action="store_true", help="Discard checkpoints and recut every task.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing Zarr store.")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep task files after a successful merge.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of station-day workers.",
    )
    return parser.parse_args()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cut_fingerprint(args, tasks):
    fields = (
        "samp_rate", "freq_band", "chn_p", "chn_s", "win_temp_p", "win_temp_s",
        "dt_thres",
    )
    waveform_paths = sorted({path for task in tasks for path in task["stream_paths"]})
    waveforms = []
    for path in waveform_paths:
        stat = os.stat(path)
        waveforms.append((os.path.realpath(path), stat.st_size, stat.st_mtime_ns))
    signature = {
        "phase_file": os.path.realpath(args.fpha),
        "phase_sha256": _sha256_file(args.fpha),
        "data_dir": os.path.realpath(args.data_dir),
        "waveforms": waveforms,
        "config": {key: getattr(cfg, key) for key in fields},
        "tasks": [(task["date_code"], task["station"], len(task["requests"])) for task in tasks],
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_run_manifest(path, manifest):
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _read_run_manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _store_fingerprint(zarr_path):
    if not os.path.exists(zarr_path):
        return None
    try:
        import zarr

        return zarr.open_group(zarr_path, mode="r").attrs.get("run_fingerprint")
    except Exception:
        return None


def _cleanup_checkpoints(checkpoint_dir, keep_checkpoints):
    if keep_checkpoints:
        return
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)


def _merge_checkpoints(checkpoint_dir, tasks, phase_specs, zarr_path, fingerprint):
    files_by_station = defaultdict(list)
    total_rows = 0
    for task_index, task in enumerate(tasks):
        data_path, marker_path = station_day_checkpoint_paths(checkpoint_dir, task_index, task)
        with open(marker_path, encoding="utf-8") as handle:
            marker = json.load(handle)
        total_rows += int(marker.get("rows", 0))
        if marker.get("status") == "written":
            files_by_station[task["station"]].append((data_path, int(marker["rows"])))

    writer = StationPhaseWriter(zarr_path, cfg, phase_specs, run_fingerprint=fingerprint)
    for station, paths in tqdm(sorted(files_by_station.items()), desc="Writing station Zarr groups"):
        station_rows = sum(row_count for _, row_count in paths)
        writer.write_station_checkpoints(station, paths, station_rows)
    return writer, total_rows


def main():
    args = parse_args()
    if not os.path.isfile(args.fpha):
        raise FileNotFoundError(args.fpha)
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(args.data_dir)
    parent_dir = os.path.dirname(args.zarr_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_dir)), exist_ok=True)

    print(f"Reading phase picks from {args.fpha}")
    event_list = read_fpha(args.fpha)
    tasks = build_station_day_tasks(event_list, args.data_dir)
    phase_specs = get_phase_specs(cfg)
    fingerprint = _cut_fingerprint(args, tasks)
    run_manifest_path = os.path.join(args.checkpoint_dir, "run.json")

    if args.restart and os.path.exists(args.zarr_path) and not args.overwrite:
        raise FileExistsError("The Zarr store exists; pass --overwrite together with --restart to replace it.")
    if args.restart and os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_manifest = _read_run_manifest(run_manifest_path)
    if run_manifest is None:
        if os.listdir(args.checkpoint_dir):
            raise RuntimeError(f"Invalid checkpoint directory {args.checkpoint_dir}; use --restart to clear it.")
        if os.path.exists(args.zarr_path) and not args.overwrite:
            raise FileExistsError(f"Zarr store exists: {args.zarr_path}; pass --overwrite to replace it.")
        run_manifest = {
            "format": "quakeweave_cut_checkpoint_v1",
            "fingerprint": fingerprint,
            "num_tasks": len(tasks),
            "status": "cutting",
        }
        _write_run_manifest(run_manifest_path, run_manifest)
    elif run_manifest.get("fingerprint") != fingerprint or run_manifest.get("num_tasks") != len(tasks):
        raise RuntimeError(
            "Checkpoint inputs or settings differ from this run; use matching inputs or pass --restart."
        )

    if run_manifest.get("status") == "complete" and _store_fingerprint(args.zarr_path) == fingerprint:
        _cleanup_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
        print(f"Zarr store already complete: {args.zarr_path}")
        return
    if os.path.exists(args.zarr_path) and _store_fingerprint(args.zarr_path) != fingerprint and not args.overwrite:
        raise FileExistsError(
            f"Zarr store exists and does not match this checkpoint: {args.zarr_path}; "
            "pass --overwrite to replace it after processing."
        )

    pending = [
        (task_index, task)
        for task_index, task in enumerate(tasks)
        if not checkpoint_is_complete(args.checkpoint_dir, task_index, task)
    ]
    completed = len(tasks) - len(pending)
    print(f"Resuming cut stage: {completed}/{len(tasks)} station-day tasks already checkpointed")
    checkpoint_worker = partial(
        process_station_day_to_checkpoint,
        cfg=cfg,
        phase_specs=phase_specs,
        checkpoint_dir=args.checkpoint_dir,
    )
    with mp.Pool(processes=max(1, int(args.num_workers))) as pool:
        iterator = pool.imap_unordered(checkpoint_worker, pending, chunksize=1)
        for _ in tqdm(iterator, total=len(pending), desc="Checkpointing station-days"):
            pass

    run_manifest["status"] = "cut_complete"
    _write_run_manifest(run_manifest_path, run_manifest)
    writer, total_rows = _merge_checkpoints(
        args.checkpoint_dir, tasks, phase_specs, args.zarr_path, fingerprint
    )
    run_manifest["status"] = "complete"
    run_manifest["total_rows"] = total_rows
    _write_run_manifest(run_manifest_path, run_manifest)
    _cleanup_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
    print(f"Wrote {total_rows} station-event phase windows to {args.zarr_path}")
    print(f"Zarr layout: {writer.root.attrs.get('layout', '<missing>')}")


if __name__ == "__main__":
    main()
