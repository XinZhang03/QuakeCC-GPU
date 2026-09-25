"""Input readers for phase files, station files, and waveform paths."""

import glob
import gzip
import os

from obspy import UTCDateTime


def dtime2str(dtime):
    date = "".join(str(dtime).split("T")[0].split("-"))
    time = "".join(str(dtime).split("T")[1].split(":"))[0:9]
    return date + time


def _read_lines(path):
    with open(path) as fin:
        return fin.readlines()


def _parse_pick_time(value):
    value = value.strip()
    if value == "-1":
        return -1
    return UTCDateTime(value)


def read_fpha(fpha):
    """Read the original phase file used for cutting waveforms."""
    return list(iter_fpha(fpha))


def iter_fpha(fpha):
    """Stream the original phase file event by event."""
    current_event = None
    opener = gzip.open if str(fpha).endswith(".gz") else open
    with opener(fpha, "rt") as fin:
        for line in fin:
            if not line.strip():
                continue
            codes = [code.strip() for code in line.split(",")]
            if len(codes[0]) > 10:
                if current_event is not None:
                    yield current_event
                ot = UTCDateTime(codes[0])
                lat, lon, dep, mag = [float(code) for code in codes[1:5]]
                evid = codes[-1]
                event_name = dtime2str(ot)
                current_event = [evid, event_name, [ot, lat, lon, dep, mag], {}]
            elif current_event is not None:
                net_sta = codes[0]
                tp = _parse_pick_time(codes[1])
                ts = _parse_pick_time(codes[2])
                current_event[-1][net_sta] = [tp, ts]
    if current_event is not None:
        yield current_event


def iter_fpha_temp(fpha, ot_min=None, ot_max=None):
    """Stream the phase.temp file used for pair generation and dt.cc output."""
    if ot_min is None:
        ot_min = UTCDateTime("19000101")
    if ot_max is None:
        ot_max = UTCDateTime("21000101")

    current_event = None
    to_add = False
    with open(fpha) as fin:
        for line in fin:
            if not line.strip():
                continue
            codes = [code.strip() for code in line.split(",")]
            if len(codes[0]) >= 10:
                if current_event is not None:
                    yield current_event
                current_event = None
                evid, event_name = codes[0].split("_", 1)
                ot = UTCDateTime(codes[1])
                if ot_min < ot < ot_max:
                    lat, lon, dep, mag = [float(code) for code in codes[2:6]]
                    current_event = [evid, event_name, [ot, lat, lon, dep, mag], {}]
                    to_add = True
                else:
                    to_add = False
            else:
                if not to_add or current_event is None:
                    continue
                net_sta = codes[0]
                tp = _parse_pick_time(codes[1])
                ts = _parse_pick_time(codes[2])
                current_event[-1][net_sta] = [tp, ts]
    if current_event is not None:
        yield current_event


def read_fpha_temp(fpha, ot_min=None, ot_max=None):
    """Read the phase.temp file used for pair generation and dt.cc output."""
    return list(iter_fpha_temp(fpha, ot_min=ot_min, ot_max=ot_max))


def read_fsta(fsta):
    """Read station coordinates."""
    sta_dict = {}
    for line in _read_lines(fsta):
        codes = [code.strip() for code in line.split(",")]
        net_sta = codes[0]
        lat, lon, ele = [float(code) for code in codes[1:4]]
        sta_dict[net_sta] = [lat, lon, ele]
    return sta_dict


def get_data_dict(date, data_dir):
    """Collect 3-component waveform paths for one day."""
    date_code = f"{date.year:04d}{date.month:02d}{date.day:02d}"
    data_dict = {}
    for st_path in sorted(glob.glob(os.path.join(data_dir, date_code, "*"))):
        if not os.path.isfile(st_path) or not st_path.lower().endswith(".mseed"):
            continue
        fname = os.path.basename(st_path)
        net_sta = ".".join(fname.split(".")[0:2])
        data_dict.setdefault(net_sta, []).append(st_path)

    return {net_sta: paths for net_sta, paths in data_dict.items() if len(paths) == 3}
