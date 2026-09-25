# QuakeCC-GPU

GPU-accelerated waveform cross-correlation for earthquake relocation. The package cuts and filters phase-centered windows on CPU, prepares event pairs, computes normalized cross-correlation on NVIDIA GPUs, and writes differential-time measurements in `dt.cc` format.

## Layout

```text
QuakeCC-GPU/
├── input/                 # Phase catalog, station metadata, bundled example waveforms
├── src/                   # Readers, signal helpers, GPU code, merger source
├── bin/merge_obs          # Native merger (built automatically when needed)
├── output/
│   ├── zarr_root/         # Cut waveform windows
│   ├── gpu_input_dir/     # Prepared station-pair tasks
│   └── output_dir/        # dt.cc and temporary records
├── 0_gpu_cut_events.py
├── 1_prepare_gpu_input.py
├── 2_gpu_ph2cc.py
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

The compact example uses 37 events from 20 minutes on **2018-08-15**, the busiest day in the Cahuilla catalog, and eight stations. Its 24-minute, three-component waveform snippets are included in `input/waveforms/20180815/`, so no download step is needed. Run the stages from the project root:

```bash
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

The files must cover all picks and configured windows, with enough lead-in for filter warm-up. Files are sorted by name and their component order must match `chn_p`/`chn_s`; the example assumes E, N, Z. Stations with missing components or a different number of miniSEED files are skipped. The bundled sample is from the AZ ANZA Regional Network and CI Southern California Seismic Network, accessed through EarthScope and SCEDC. Cite the networks ([AZ DOI](https://doi.org/10.7914/SN/AZ), [CI DOI](https://doi.org/10.7914/SN/CI)) and the [SCEDC dataset](https://doi.org/10.7909/C3WD3xH1) when using these waveforms. The waveform data are separate from the repository's MIT-licensed software. The data centers provide [EarthScope citation guidance](https://www.earthscope.org/terms-of-service/) and [SCEDC citation guidance](https://scedc.caltech.edu/about/citation.html).

### Put Zarr on a large-volume disk

For large catalogs, the cut-window Zarr store can grow substantially. Set `zarr_root` in `config.py` to an absolute path on a large, fast data volume instead of storing it under the repository or a small root filesystem. Keep the prepared-task and final-output directories on storage with enough free space as well.

### PALM / AI-PAL interoperability

[PALM](https://github.com/YijianZhou/PALM) and [AI-PAL](https://github.com/YijianZhou/AI-PAL) can supply associated, located events and phase picks for QuakeCC-GPU. Convert those products to QuakeCC's `.pha` layout: event rows contain `origin_time,latitude,longitude,depth_km,magnitude,event_id`; pick rows contain `NET.STA,P_time,S_time`, using `-1` for a missing pick. Use the matching station file and continuous waveforms.

