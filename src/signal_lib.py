"""Waveform preprocessing helpers."""

import numpy as np


def preprocess(stream, samp_rate, freq_band):
    start_time = max(trace.stats.starttime for trace in stream)
    end_time = min(trace.stats.endtime for trace in stream)
    if start_time > end_time:
        return []

    st = stream.slice(start_time, end_time)
    st = st.detrend("demean").detrend("linear").taper(max_percentage=0.05, max_length=10.0)

    if any(trace.stats.sampling_rate != samp_rate for trace in st):
        st.interpolate(sampling_rate=samp_rate, method="cubic")

    for trace in st:
        trace.data[np.isnan(trace.data)] = 0
        trace.data[np.isinf(trace.data)] = 0

    freq_min, freq_max = freq_band
    if freq_min and freq_max:
        return st.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
    if freq_min and not freq_max:
        return st.filter("highpass", freq=freq_min)
    if freq_max and not freq_min:
        return st.filter("lowpass", freq=freq_max)
    return st
