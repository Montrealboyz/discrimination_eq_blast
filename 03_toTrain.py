import os
import glob
import numpy as np
import pandas as pd
import torch
from scipy import signal
from obspy import read, read_inventory, UTCDateTime
from obspy.core.stream import Stream
import seisbench.models as sbm

# ─────────── CONFIG ───────────
PROJECT_ROOT       = "/media/justin/Data2/Discrime_paper_example"
REGIONS            = {
    "test": {
        "df_csv": os.path.join(PROJECT_ROOT, "earthquake_SNR_test.csv"),
        "station_dir": os.path.join(PROJECT_ROOT, "earthquake_station"),
        "waveform_dir": os.path.join(PROJECT_ROOT, "earthquake_waveform")
    }}
"""
    "ADD more region if you want": {
        "df_csv": os.path.join(PROJECT_ROOT, "XXXXXXXXX.csv"),
        "station_dir": os.path.join(PROJECT_ROOT, "XXXXXXXXXXX"),
        "waveform_dir": os.path.join(PROJECT_ROOT, "XXXXXXXXXX")
    }
"""
OUT_TENSOR    = os.path.join(PROJECT_ROOT, "earthquake_tensor_SNR18.pt")
OUT_HOUR_NPY  = os.path.join(PROJECT_ROOT, "earthquake_tensor_SNR18_hour.npy")
SNR_THRESHOLD = 1.8
FS            = 100
NPERSEG       = 256
NOVERLAP      = 128
EPSILON       = 1e-40
# ───────────────────────────────


def load_and_filter(df_csv: str, threshold: float):
    df = pd.read_csv(df_csv,dtype={"Event": str})
    return df[df["SNR_denoise_dB"] >= threshold]

def process_region(df: pd.DataFrame, station_root: str, waveform_root: str, model):
    """
    For each (time, station) in df, load inventory + waveforms,
    compute a normalized 3‐component spectrogram, return list of arrays & hours.
    """
    specs = []
    hours = []

    # enumerate so we can report every 100 rows
    for idx, (time_str, sta) in enumerate(zip(df["Event"], df["Station"]), start=1):
        # print progress every 100 items
        if idx % 3 == 0:
            print(f"  → process_region: handled {idx} rows so far…")

        evt = UTCDateTime(time_str)
        evdir = evt.strftime("%Y%m%d%H%M%S")

        # 1) inventory
        inv_files = glob.glob(os.path.join(station_root, evdir, f"*{sta}*"))
        if not inv_files:
            continue
        inv = read_inventory(inv_files[0])

        # 2) raw waveforms
        wf_files = glob.glob(os.path.join(waveform_root, evdir, f"*{sta}*"))
        if not wf_files:
            continue
        st = Stream()
        for f in wf_files:
            st += read(f)

        # 3) remove unwanted stations
        for drop_sta in ("M53A","O53A","P52A"):
            for tr in st.select(station=drop_sta):
                st.remove(tr)
        if len(st) == 0:
            continue

        # 4) preprocess & denoise
        for tr in st:
            sr = round(tr.stats.sampling_rate)
            pre_filt = [0.001,0.002,45,50] if sr == 100 else [0.001,0.002,18,20]
            tr.remove_response(
                inventory=inv,
                pre_filt=pre_filt,
                output="DISP",
                water_level=60,
                taper=True,
                taper_fraction=1e-5
            )
        st.detrend("linear")
        st.taper(max_percentage=0.05, type="hann")
        st.filter("highpass", freq=1.0)

        denoised = model.annotate(st)

        # 5) spectrogram (only if 3‐component and long enough)
        mats = []
        for tr in denoised:
            if tr.stats.npts > 11990:
                _, _, Sxx = signal.spectrogram(
                    tr.data,
                    fs=FS,
                    nperseg=NPERSEG,
                    noverlap=NOVERLAP
                )
                mats.append(Sxx)
        if len(mats) != 3:
            continue

        # 6) stack & normalize
        arr = np.stack(mats, axis=-1)
        loga = np.log10(arr + EPSILON)
        s0, s1 = loga.min(), loga.max()
        norm = (loga - s0) / (s1 - s0)

        specs.append(norm)
        hours.append(evt.hour)

    return specs, hours


def main():
    denoiser = sbm.DeepDenoiser.from_pretrained("original")
    all_specs = []
    all_hours = []


    for region, cfg in REGIONS.items():
        df_filt = load_and_filter(cfg["df_csv"], SNR_THRESHOLD)
        specs, hrs = process_region(
            df_filt,
            cfg["station_dir"],
            cfg["waveform_dir"],
            denoiser
        )

        # extend lists and update counter
        all_specs.extend(specs)
        all_hours.extend(hrs)

    # stack into single array: (freq x time x comp x sample)
    X = np.stack(all_specs, axis=-1)
    y = np.array(all_hours, dtype=np.int64)

    # convert & permute to (N x C x F x T)
    X_torch = torch.from_numpy(X).permute(3, 2, 0, 1).to(torch.float32)

    # save
    torch.save(X_torch, OUT_TENSOR)
    np.save(OUT_HOUR_NPY, y)

    print(f"\n🎉  Done! ")
    print(f"Saved tensor dataset: {OUT_TENSOR}")
    print(f"Saved hour labels:    {OUT_HOUR_NPY}")

if __name__ == "__main__":
    main()

