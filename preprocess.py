"""
==============================================================
  ECG Signal Preprocessing Pipeline
  MIT-BIH Arrhythmia Database
==============================================================
  This script:
    1. Downloads ECG records from MIT-BIH via the `wfdb` library
    2. Applies a bandpass filter to remove noise
    3. Normalises each beat
    4. Segments individual heartbeats around R-peaks
    5. Labels beats as Normal (0) or Arrhythmia (1)
    6. Saves the dataset as NumPy arrays ready for training

  Run:  python scripts/preprocess.py
==============================================================
"""

import os
import numpy as np
import wfdb                          # reads PhysioNet/MIT-BIH records
from scipy.signal import butter, filtfilt
from tqdm import tqdm                # progress bar (pip install tqdm)

# ----------------------------------------------
# CONFIGURATION  (tweak these freely)
# ----------------------------------------------
DATA_DIR   = "data"          # where raw records are cached
OUT_DIR    = "data"          # where preprocessed arrays are saved
WINDOW     = 180             # samples on each side of R-peak  → beat = 360 pts
FS         = 360             # MIT-BIH sampling frequency (Hz)
LOW_CUT    = 0.5             # bandpass lower edge (Hz)
HIGH_CUT   = 45.0            # bandpass upper edge (Hz)
FILTER_ORD = 4               # Butterworth filter order

# MIT-BIH record numbers (48 total; we use a subset for speed)
RECORDS = [
    "100","101","103","105","106","108","109","111","112","113",
    "114","115","116","117","118","119","121","122","123","124",
    "200","201","202","203","205","207","208","209","210","212",
    "213","214","215","217","219","220","221","222","223","228",
    "230","231","232","233","234"
]

# AAMI beat-class mapping used throughout the literature
#   N  → Normal / non-ectopic  → label 0
#   other → Arrhythmia          → label 1
NORMAL_SYMBOLS   = {"N", "L", "R", "e", "j"}   # AAMI Class N
ARRHYTHMIA_SYMBOLS = {"V", "A", "F", "f", "/", "!", "E", "J", "a"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)


# ----------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: int, low: float, high: float, order: int) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Zero-phase (filtfilt) means no time-delay distortion,
    which matters when we are aligning peaks precisely.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def normalise(signal: np.ndarray) -> np.ndarray:
    """
    Min-max normalise a 1-D signal to the range [0, 1].

    Avoids dividing by zero when the beat is flat (edge case).
    """
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max - sig_min < 1e-8:
        return np.zeros_like(signal)
    return (signal - sig_min) / (sig_max - sig_min)


def extract_beats(record_name: str):
    """
    Download (or load cached) one MIT-BIH record, filter it,
    and return (beats, labels) arrays.

    Parameters
    ----------
    record_name : str
        e.g. "100"

    Returns
    -------
    beats  : np.ndarray  shape (N, 2*WINDOW)
    labels : np.ndarray  shape (N,)   dtype int   0=Normal 1=Arrhythmia
    """
    # wfdb downloads to pn_dir automatically and caches locally
    try:
        record = wfdb.rdrecord(record_name, pn_dir="mitdb", sampfrom=0)
        ann    = wfdb.rdann(record_name,   "atr", pn_dir="mitdb")
    except Exception as exc:
        print(f"  [WARN] Could not load record {record_name}: {exc}")
        return np.array([]), np.array([])

    # Use the first channel (MLII lead) only
    raw_signal = record.p_signal[:, 0]

    # --- Step 1: Noise removal ---
    filtered = bandpass_filter(raw_signal, FS, LOW_CUT, HIGH_CUT, FILTER_ORD)

    beats, labels = [], []

    for idx, symbol in zip(ann.sample, ann.symbol):
        # Keep only annotated beat types we recognise
        if symbol not in NORMAL_SYMBOLS and symbol not in ARRHYTHMIA_SYMBOLS:
            continue

        # --- Step 2: Segmentation ---
        start = idx - WINDOW
        end   = idx + WINDOW
        if start < 0 or end > len(filtered):
            continue          # skip incomplete windows near record edges

        beat = filtered[start:end]

        # --- Step 3: Normalisation ---
        beat = normalise(beat)

        label = 0 if symbol in NORMAL_SYMBOLS else 1

        beats.append(beat)
        labels.append(label)

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int32)


# ----------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------

def build_dataset():
    all_beats, all_labels = [], []

    print(f"\n[INFO] Processing {len(RECORDS)} MIT-BIH records …\n")
    for rec in tqdm(RECORDS, desc="Records"):
        beats, labels = extract_beats(rec)
        if len(beats):
            all_beats.append(beats)
            all_labels.append(labels)

    X = np.concatenate(all_beats,  axis=0)   # (total_beats, 360)
    y = np.concatenate(all_labels, axis=0)   # (total_beats,)

    # ----------------------------------------------
    # Handle missing / corrupted data
    # ----------------------------------------------
    # After bandpass + normalise, NaN/Inf can still appear on very
    # noisy channels.  Remove any such beats.
    valid_mask = np.isfinite(X).all(axis=1)
    n_dropped  = (~valid_mask).sum()
    if n_dropped:
        print(f"\n[WARN] Dropping {n_dropped} corrupted beats.")
    X, y = X[valid_mask], y[valid_mask]

    # ----------------------------------------------
    # Class balance report
    # ----------------------------------------------
    n_normal = (y == 0).sum()
    n_arrhy  = (y == 1).sum()
    print(f"\n[INFO] Dataset summary:")
    print(f"       Total beats : {len(y):,}")
    print(f"       Normal (0)  : {n_normal:,}  ({100*n_normal/len(y):.1f}%)")
    print(f"       Arrhythmia  : {n_arrhy:,}   ({100*n_arrhy/len(y):.1f}%)")

    # ----------------------------------------------
    # Save data
    # ----------------------------------------------
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    print(f"\n[INFO] Saved  data/X.npy  {X.shape}")
    print(f"[INFO] Saved  data/y.npy  {y.shape}")
    return X, y


if __name__ == "__main__":
    build_dataset()
