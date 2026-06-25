"""
==============================================================
  QUICK-START DEMO  –  ECG CNN Classifier (Synthetic Data)
==============================================================
  Run this FIRST if you want to test the full pipeline
  without downloading the MIT-BIH database.

  It:
    1. Generates realistic-looking synthetic ECG beats
    2. Runs the same preprocessing pipeline
    3. Trains the CNN for a few epochs
    4. Evaluates and saves all plots to  results/

  Usage:
    python scripts/demo.py

  After this works end-to-end, swap in real MIT-BIH data by
  running  preprocess.py  →  train.py  →  evaluate.py  instead.
==============================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
BEAT_LEN   = 360
N_NORMAL   = 4000    # synthetic Normal beats
N_ARRHYTHMIA = 1200  # synthetic Arrhythmia beats  (imbalanced on purpose)
EPOCHS     = 15
BATCH_SIZE = 64
RESULT_DIR = "results"
MODEL_DIR  = "models"
SEED       = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ----------------------------------------------
# 1.  SYNTHETIC DATA GENERATOR
# ----------------------------------------------

def make_ecg_beat(arrhythmia: bool = False) -> np.ndarray:
    """
    Simulate a single ECG beat using Gaussian bumps for P, QRS, T waves.

    Normal beat  : standard PQRST morphology
    Arrhythmia   : wider/shifted R-peak, inverted T-wave, irregular baseline
    """
    t   = np.linspace(0, 1, BEAT_LEN)
    ecg = np.zeros(BEAT_LEN)

    if not arrhythmia:
        # P-wave
        ecg += 0.25 * np.exp(-((t - 0.20) ** 2) / (2 * 0.015**2))
        # Q dip
        ecg -= 0.10 * np.exp(-((t - 0.35) ** 2) / (2 * 0.008**2))
        # R spike
        ecg += 1.00 * np.exp(-((t - 0.40) ** 2) / (2 * 0.010**2))
        # S dip
        ecg -= 0.15 * np.exp(-((t - 0.45) ** 2) / (2 * 0.008**2))
        # T-wave
        ecg += 0.35 * np.exp(-((t - 0.65) ** 2) / (2 * 0.030**2))
    else:
        # Wider, taller R (PVC-like)
        r_pos = np.random.uniform(0.35, 0.50)
        ecg += 1.20 * np.exp(-((t - r_pos) ** 2) / (2 * 0.022**2))
        # Inverted/slurred T
        ecg -= 0.40 * np.exp(-((t - 0.70) ** 2) / (2 * 0.040**2))
        # Baseline wander
        ecg += 0.12 * np.sin(2 * np.pi * 1.5 * t + np.random.uniform(0, np.pi))

    # Add realistic noise
    ecg += np.random.normal(0, 0.03, BEAT_LEN)

    # Normalise to [0, 1]
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-8)
    return ecg.astype(np.float32)


def generate_dataset():
    print("[INFO] Generating synthetic ECG beats …")
    X_n = np.array([make_ecg_beat(arrhythmia=False) for _ in range(N_NORMAL)])
    X_a = np.array([make_ecg_beat(arrhythmia=True)  for _ in range(N_ARRHYTHMIA)])

    X = np.concatenate([X_n, X_a], axis=0)
    y = np.concatenate([np.zeros(N_NORMAL, dtype=np.int32),
                        np.ones(N_ARRHYTHMIA,  dtype=np.int32)])

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    print(f"[INFO] Dataset: {len(y)} beats  ({N_NORMAL} Normal, {N_ARRHYTHMIA} Arrhythmia)")
    return X, y


# ----------------------------------------------
# 2.  VISUALISE SAMPLE BEATS  (before training)
# ----------------------------------------------

def plot_sample_beats(X, y):
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    t = np.linspace(0, 1000, BEAT_LEN)   # ms
    labels = ["Normal", "Arrhythmia"]
    colors = ["steelblue", "crimson"]

    for row in range(2):
        idxs = np.where(y == row)[0][:3]
        for col, idx in enumerate(idxs):
            ax = axes[row][col]
            ax.plot(t, X[idx], color=colors[row], linewidth=1.5)
            ax.set_title(f"{labels[row]} Beat #{col+1}", fontsize=10)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (normalised)")
            ax.grid(alpha=0.3)

    plt.suptitle("Synthetic ECG Beats – Sample Visualisation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "sample_ecg_plots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved  {path}")


# ----------------------------------------------
# 3.  TRAIN
# ----------------------------------------------

def train(X_train, y_train, X_val, y_val):
    # Add channel dim for Conv1D
    Xt = X_train[..., np.newaxis]
    Xv = X_val[..., np.newaxis]

    # Class weights
    cw_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(cw_arr))

    model = build_cnn(input_length=BEAT_LEN)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=3, verbose=1)
    ]

    print(f"\n[INFO] Training for up to {EPOCHS} epochs …\n")
    history = model.fit(
        Xt, y_train,
        validation_data=(Xv, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


# ----------------------------------------------
# 4.  PLOT TRAINING CURVES
# ----------------------------------------------

def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(history.history["loss"],     label="Train", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val",   linewidth=2, linestyle="--")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved  {path}")


# ----------------------------------------------
# 5.  EVALUATE
# ----------------------------------------------

def evaluate(model, X_test, y_test):
    Xt = X_test[..., np.newaxis]
    y_prob = model.predict(Xt, batch_size=256, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print("\n" + "="*50)
    print("  EVALUATION RESULTS  (test set)")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("="*50)
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Arrhythmia"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Arrhythmia"],
                yticklabels=["Normal", "Arrhythmia"], ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved  {path}")

    return y_prob, y_pred


# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    print("\n" + "="*60)
    print("  ECG ARRHYTHMIA CLASSIFIER  –  DEMO (Synthetic Data)")
    print("="*60 + "\n")

    X, y = generate_dataset()
    plot_sample_beats(X, y)

    # 70 / 15 / 15 split
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15,
                                                    stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15/0.85,
                                                        stratify=y_tv, random_state=SEED)

    model, history = train(X_train, y_train, X_val, y_val)
    plot_curves(history)
    evaluate(model, X_test, y_test)

    # Save model
    model_path = os.path.join(MODEL_DIR, "ecg_cnn_demo.keras")
    model.save(model_path)
    print(f"\n[INFO] Demo model saved →  {model_path}")
    print("[INFO] All plots saved  →  results/")
    print("\n[DONE] ✓  Full pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
