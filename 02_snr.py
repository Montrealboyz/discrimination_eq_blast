import os
import glob
import math
import numpy as np
import pandas as pd
from obspy import read, read_inventory, UTCDateTime
from obspy.core.stream import Stream
import seisbench.models as sbm

# ─────────── CONFIG ───────────
PROJECT_ROOT  = "/media/justin/Data2/Discrime_paper_example"
CAT_PATH      = os.path.join(PROJECT_ROOT, "earthquake_catalog.csv")
STATION_ROOT  = os.path.join(PROJECT_ROOT, "earthquake_station")
WAVEFORM_ROOT = os.path.join(PROJECT_ROOT, "earthquake_waveform")
OUT_CSV       = os.path.join(PROJECT_ROOT, "earthquake_SNR_test.csv")
# ───────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great‐circle distance between two points on Earth (km).
    """
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * math.asin(math.sqrt(a)) * 6371.0


def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Time'] = df['Time'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return df.set_index('Time')


def load_denoiser(model_name: str = "original"):
    return sbm.DeepDenoiser.from_pretrained(model_name)


def preprocess_stream(st: Stream, inv, fs: int) -> Stream:
    """
    Detrend, taper, highpass, and remove instrument response.
    """
    st = st.copy()
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("highpass", freq=1.0)
    # choose pre_filt based on sampling rate
    for tr in st:
        sr = round(tr.stats.sampling_rate)
        if sr == 100:
            pre_filt = [0.001, 0.002, 45, 50]
        elif sr == 40:
            pre_filt = [0.001, 0.002, 18, 20]
        else:
            continue
        tr.remove_response(
            inventory=inv, pre_filt=pre_filt,
            output="DISP", water_level=60,
            taper=True, taper_fraction=1e-5
        )
    return st


def compute_snr(traces: Stream, fs: int, t1: float, t2: float, ddof: int = 1) -> float:
    """
    Compute average SNR (in dB) across all traces in a 3‐component set.
    """
    snrs = []
    for tr in traces:
        data = tr.data
        # sample indices
        start = int(round(t1 * fs))
        end   = int(round(t2 * fs))
        signal = data[start:end]
        noise  = np.concatenate([data[:start], data[end:]])
        if len(signal) < ddof or len(noise) < ddof:
            continue
        p_sig = np.var(signal, ddof=ddof)
        p_noi = np.var(noise,  ddof=ddof)
        snr   = (p_sig / p_noi) if (p_sig and p_noi) else 1e-8
        snrs.append(10 * np.log10(snr))
    return float(np.mean(snrs)) if snrs else np.nan


def process_event(event_time: str,
                  df_cat: pd.DataFrame,
                  station_root: str,
                  waveform_root: str,
                  denoiser,
                  ddof: int = 1):
    """
    For a single event, read inventory + waveforms, compute original & denoised SNRs.
    You can modify time window by adjusting thoritical P and S wave speed and + - seconds 
    Returns a list of dicts, one per station.
    """
    ts = pd.to_datetime(event_time, format='%Y%m%d%H%M%S', utc=True)
    key = ts.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    row = df_cat.loc[key]    # now matches the index
    ev_lat, ev_lon = row['Latitude'], row['Longitude']

    # read station inventory
    os.chdir(os.path.join(station_root, event_time))
    inv = read_inventory('*')

    # read raw waveforms
    st0 = read(os.path.join(waveform_root, event_time, '*'))
    # select only channels of interest
    st0 = Stream([tr for tr in st0
                  if (round(tr.stats.sampling_rate) in (40,100))
                  and tr.stats.npts > 10 * round(tr.stats.sampling_rate)])
    st0 = preprocess_stream(st0, inv, fs=None)  # will pick pre_filt in function

    # run denoiser annotations (always at 100 Hz)
    annotations = denoiser.annotate(st0)

    results = []
    for sta in set(tr.stats.station for tr in st0):
        st_raw = st0.select(station=sta)
        st_den = annotations.select(station=sta)
        # only compute if we have 3-components
        if len(st_raw) == 3:
            # compute distance & time window
            coords = inv.get_coordinates(st_raw[0].stats.network+'.' + st_raw[0].stats.station + '.' + st_raw[0].stats.location + '.' + st_raw[0].stats.channel, UTCDateTime(event_time))
            dist = haversine(coords['latitude'], coords['longitude'], ev_lat, ev_lon)
            t1 = dist/6.5 - 1 + 30 # +30 because we download the waveform from -30 secs compare to event origin time
            t2 = dist/3.7 + 5 + 30

            # sampling rates
            fs_raw = round(st_raw[0].stats.sampling_rate)
            snr_raw = compute_snr(st_raw, fs_raw, t1, t2, ddof)

            fs_den = 100
            snr_den = compute_snr(st_den, fs_den, t1, t2, ddof)

            results.append({
                'Event': event_time,
                'Station': sta,
                'Distance_km': dist,
                'SNR_raw_dB': snr_raw,
                'SNR_denoise_dB': snr_den
            })
    return results


def main():
    cat_path      = CAT_PATH
    station_root  = STATION_ROOT
    waveform_root = WAVEFORM_ROOT
    out_csv       = OUT_CSV

    df_cat    = load_catalog(cat_path)
    denoiser  = load_denoiser("original")

    all_events = sorted(os.listdir(station_root))
    all_results = []
    for ev in all_events:
        print(f"Processing {ev} …")
        try:
            res = process_event(ev, df_cat, station_root, waveform_root, denoiser)
            all_results.extend(res)
        except Exception as e:
            print(f"  → Skipped {ev}: {e}")

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(out_csv, index=False)
    print(f"Done! Results written to {out_csv}")


if __name__ == "__main__":
    main()
