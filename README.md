# QuakeCC-GPU

GPU-accelerated waveform cross-correlation for earthquake relocation. The package cuts and filters phase-centered windows on CPU, prepares event pairs, computes normalized cross-correlation on NVIDIA GPUs, and writes differential-time measurements in `dt.cc` format.

## Layout

```text
QuakeCC-GPU/
├── input/                 # Phase catalog, station metadata, downloaded waveforms
├── src/                   # Readers, signal helpers, GPU code, merger source
├── bin/merge_obs          # Native merger (built automatically when needed)
├── output/
│   ├── zarr_root/         # Cut waveform windows
│   ├── gpu_input_dir/     # Prepared station-pair tasks
│   └── output_dir/        # dt.cc and temporary records
├── 0_gpu_cut_events.py
├── 1_prepare_gpu_input.py
├── 2_gpu_ph2cc.py
└── example_data_download.py
```

## Install

Use Python 3.12, a CUDA 13.0-capable NVIDIA driver, and a C++17 compiler with OpenMP. The versions in `requirements.txt` match the tested `gpu_5090` `gputorch` environment. Install the CUDA-enabled PyTorch wheel first, then the pinned dependencies:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

For a different CUDA platform, install its compatible PyTorch build and adjust the `torch` pin accordingly.

The merger source is compiled to `bin/merge_obs` automatically when required.

## Run the Cahuilla example

The compact example uses 37 events from 20 minutes on **2018-08-15**, the busiest day in the Cahuilla catalog, and eight stations. Download its 24-minute, three-component waveform snippets, then run the stages from the project root:

```bash
python example_data_download.py
python 0_gpu_cut_events.py
python 1_prepare_gpu_input.py
python 2_gpu_ph2cc.py
```

The result is `output/output_dir/dt_all.cc`. Step 0 writes atomic station-day checkpoints to `<zarr_root>.checkpoints/`; after interruption, repeat the same command with the same inputs and settings to skip completed cuts. Use `--restart --overwrite` to discard checkpoints and rebuild the Zarr store. Step 1 stops if prepared tasks already exist; pass `--clean-task-dir` to replace them.

## Configuration

Edit `config.py` for input paths, waveform processing, event-pair criteria, quality thresholds, and GPU execution. `cc_thres` filters by absolute CC peak and `scc_thres` filters by peak contrast against the correlation background. Either can be set to `None` to disable that filter.

## Tips

### Waveform directory and files

Set `data_dir` to a directory with one `YYYYMMDD/` subdirectory per day. For every station-day, provide exactly three separate miniSEED files named with the `NET.STA.` prefix, for example:

```text
input/waveforms/
└── 20180815/
    ├── AZ.BZN..HHE.mseed
    ├── AZ.BZN..HHN.mseed
    └── AZ.BZN..HHZ.mseed
```

The files must cover all picks and configured windows, with enough lead-in for filter warm-up. Files are sorted by name and their component order must match `chn_p`/`chn_s`; the example assumes E, N, Z. Stations with missing components or a different number of miniSEED files are skipped.

### Put Zarr on a large-volume disk

For large catalogs, the cut-window Zarr store can grow substantially. Set `zarr_root` in `config.py` to an absolute path on a large, fast data volume instead of storing it under the repository or a small root filesystem. Keep the prepared-task and final-output directories on storage with enough free space as well.

### PALM / AI-PAL interoperability

[PALM](https://github.com/YijianZhou/PALM) and [AI-PAL](https://github.com/YijianZhou/AI-PAL) can supply associated, located events and phase picks for QuakeCC-GPU. Convert those products to QuakeCC's `.pha` layout: event rows contain `origin_time,latitude,longitude,depth_km,magnitude,event_id`; pick rows contain `NET.STA,P_time,S_time`, using `-1` for a missing pick. Use the matching station file and continuous waveforms.

The native output requires unique, nonnegative integer event IDs below `2^20`; map PAL/AI-PAL IDs if needed and retain that mapping. QuakeCC's `dt.cc` can then be paired with an `event.dat` built from the same event-ID mapping for the PALM/HypoDD relocation workflow. PALM documents `event.dat` and `dt.cc` as the HypoDD inputs; verify the output and catalog IDs together before relocation.

## Input formats

Station rows are `NET.STA,latitude,longitude,elevation`. Phase picks are absolute timestamps. The channel indices in `config.py` are zero-based positions in the sorted E/N/Z waveform components. The example downloader skips files already present; use `--overwrite` to fetch them again.
