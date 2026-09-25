"""Default settings for the Cahuilla peak-day example."""

from pathlib import Path




class Config(object):
    def __init__(self):
        # Inputs and output locations
        self.fpha_name = "input/Cahuilla_2018-08-15.pha"               # Phase catalog
        self.fsta =  "input/Cahuilla.sta"                 # NET.STA and coordinates
        self.data_dir =  "input/waveforms"                 # Contains YYYYMMDD/ waveform dirs
        self.zarr_root = "output/zarr_root"               # Cut phase windows
        self.gpu_input_dir = "output/gpu_input_dir"       # Prepared station-pair tasks
        self.output_dir =  "output/output_dir"             # Final dt.cc and temporary records

        # Waveform processing; times are seconds relative to the phase pick
        self.samp_rate = 1000             # Target rate (Hz); cubic interpolation if resampling
        self.freq_band = [1.0, 16.0]      # Bandpass corners (Hz)
        self.chn_p = [2]                  # 0-based E,N,Z channel index: Z
        self.chn_s = [0, 1]               # 0-based E,N,Z channel indices: E and N
        self.win_temp_p = [0.2, 1.0]      # P template: 0.2 s before to 1.0 s after pick
        self.win_temp_s = [0.2, 2.5]      # S template: 0.2 s before to 2.5 s after pick
        self.dt_thres = [0.5, 0.8]        # Maximum |dt| for P and S (s)

        # Event-pair selection and phase quality control
        self.cc_thres = 0.3               # Minimum absolute CC peak; None disables CC filtering
        self.scc_thres = 7.0              # Minimum (peak - background median)/MAD; None disables
        # Excluded half-width around the CC peak when estimating SCC background (s)
        self.secondary_peak_separation = 0.03
        self.loc_dev_thres = 3.0          # Pair search half-width per horizontal axis (km)
        self.dep_dev_thres = 4.0          # Event-pair depth difference limit (km)
        self.dist_thres = 100.0             # Station must be within this distance of either event (km)
        self.num_sta_thres = 6             # Minimum qualifying common stations per pair
        self.max_nbr = 2000                # Maximum nearby event neighbors per event
        self.temp_mag = 0.0                # Minimum magnitude for a template event
        self.temp_sta = 6                  # Minimum stations for a template event
        self.cal_win = 365 * 2             # Maximum event-pair time separation (days)

        # GPU execution
        self.gpu_station_pair_batch_size = 8192  # Station-pair correlations per GPU batch
        self.gpu_compute_dtype = "float32"      # CC compute precision
        self.gpu_log_every_n_stations = 0        # Log interval; 0 disables per-station logs
        self.gpu_preload_station_data = True    # Keep each station's waveform arrays on GPU
        self.gpu_preload_fallback_to_cpu = True # On GPU OOM, retry with per-batch transfers
