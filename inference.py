"""
==============================================================
  Inference Script – ECG CNN Arrhythmia Classifier
==============================================================
  Accepts a CSV file of ECG beats and prints:
    • Predicted class (Normal / Arrhythmia) for each beat
    • Confidence score (%)

  CSV FORMAT:
    • Each ROW = one heartbeat
    • Each COLUMN = one time-sample  (360 values per beat)
    • No header row required (but it is tolerated)
    • Example:  0.12, 0.34, 0.56, … (360 values)

  Usage:
    python scripts/inference.py --csv path/to/ecg_beats.csv
    python scripts/inference.py --csv path/to/ecg_beats.csv --plot

  Optional flags:
    --threshold FLOAT   Probability cutoff (default 0.5)
    --plot              Save a PNG of each predicted beat
==============================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


# ----------------------------------------------
# CONSTANTS  (must match preprocessing)
# ----------------------------------------------
MODEL_PATH   = os.path.join("models", "ecg_cnn.keras")
EXPECTED_LEN = 360
FS           = 360
LOW_CUT      = 0.5
HIGH_CUT     = 45.0
CLASS_NAMES  = {0: "Normal", 1: "Arrhythmia"}


# ----------------------------------------------
# HELPERS
# ----------------------------------------------

def bandpass(signal: np.ndarray) -> np.ndarray:
    """Apply the same bandpass filter used during preprocessing."""
    nyq = FS / 2.0
    b, a = butter(4, [LOW_CUT / nyq, HIGH_CUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def normalise(signal: np.ndarray) -> np.ndarray:
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max - sig_min < 1e-8:
        return np.zeros_like(signal)
    return (signal - sig_min) / (sig_max - sig_min)


def preprocess_beat(raw_beat: np.ndarray) -> np.ndarray:
    """Filter + normalise a single raw beat array."""
    filtered = bandpass(raw_beat)
    normed   = normalise(filtered)
    return normed


# ----------------------------------------------
# LOAD CSV
# ----------------------------------------------

def load_csv(csv_path: str, threshold: float = 0.5):
    """
    Read the CSV.  Each row must have exactly 360 numeric values.
    Rows with a different length or non-numeric values are skipped.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"[ERROR] Could not parse CSV: {e}")
        sys.exit(1)

    # If first row is a text header, drop it
    if df.iloc[0].astype(str).str.match(r"[a-zA-Z]").any():
        df = df.iloc[1:].reset_index(drop=True)

    # Convert to float, coerce errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Keep only rows of the correct length with no NaN
    valid_mask = (df.notna().all(axis=1)) & (df.shape[1] == EXPECTED_LEN)
    n_dropped  = (~valid_mask).sum()
    if n_dropped:
        print(f"[WARN] Dropped {n_dropped} rows (wrong length or non-numeric).")

    beats = df[valid_mask].values.astype(np.float32)

    if len(beats) == 0:
        print("[ERROR] No valid beats found in the CSV.  "
              "Make sure each row has exactly 360 numeric values.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(beats)} beats from  {csv_path}")
    return beats


# ----------------------------------------------
# PREPROCESS
# ----------------------------------------------

def preprocess_all(raw_beats: np.ndarray) -> np.ndarray:
    processed = np.array([preprocess_beat(b) for b in raw_beats], dtype=np.float32)
    processed = processed[..., np.newaxis]   # → (N, 360, 1)
    return processed


# ----------------------------------------------
# PREDICT
# ----------------------------------------------

def run_inference(model, X: np.ndarray, threshold: float):
    y_prob = model.predict(X, batch_size=256, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


# ----------------------------------------------
# PRINT RESULTS TABLE
# ----------------------------------------------

def print_results(y_prob, y_pred):
    print("\n" + "="*55)
    print(f"  {'Beat':>5}  {'Prediction':>12}  {'Confidence':>12}")
    print("="*55)
    for i, (prob, pred) in enumerate(zip(y_prob, y_pred)):
        conf = prob if pred == 1 else 1 - prob
        print(f"  {i+1:>5}  {CLASS_NAMES[pred]:>12}  {conf*100:>11.2f}%")
    print("="*55)

    # Summary
    n_normal = (y_pred == 0).sum()
    n_arrhy  = (y_pred == 1).sum()
    print(f"\n  Total beats    : {len(y_pred)}")
    print(f"  Normal         : {n_normal}")
    print(f"  Arrhythmia     : {n_arrhy}")
    print()


# ----------------------------------------------
# OPTIONAL PLOT
# ----------------------------------------------

def save_beat_plots(X, y_prob, y_pred, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    n  = min(len(X), 9)   # plot up to 9 beats
    cols, rows = 3, (n + 2) // 3
    fig, axes  = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes       = axes.ravel()
    time_axis  = np.linspace(0, 1000, EXPECTED_LEN)

    for i in range(n):
        beat  = X[i].ravel()
        pred  = y_pred[i]
        conf  = y_prob[i] if pred == 1 else 1 - y_prob[i]
        color = "crimson" if pred == 1 else "steelblue"
        axes[i].plot(time_axis, beat, color=color, linewidth=1.5)
        axes[i].set_title(f"Beat {i+1}: {CLASS_NAMES[pred]} ({conf*100:.1f}%)",
                          fontsize=10, color=color)
        axes[i].set_xlabel("Time (ms)"); axes[i].set_ylabel("Amplitude")
        axes[i].grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Inference Results – Predicted ECG Beats", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(out_dir, "inference_beats.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Beat plot saved → {path}")


# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ECG Arrhythmia Classifier – Inference")
    parser.add_argument("--csv",       required=True,  help="Path to CSV file (rows = beats, cols = 360 samples)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (default 0.5)")
    parser.add_argument("--plot",      action="store_true",     help="Save beat plots to results/")
    args = parser.parse_args()

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}  — run train.py first.")
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from  {MODEL_PATH}")

    # Load + preprocess CSV
    raw_beats = load_csv(args.csv, args.threshold)
    X         = preprocess_all(raw_beats)

    # Predict
    y_prob, y_pred = run_inference(model, X, args.threshold)

    # Report
    print_results(y_prob, y_pred)

    # Optional plot
    if args.plot:
        save_beat_plots(X, y_prob, y_pred)


if __name__ == "__main__":
    main()
