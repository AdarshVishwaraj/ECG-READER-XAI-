"""
==============================================================
  Evaluation Script – ECG CNN Arrhythmia Classifier
==============================================================
  Loads the best saved model and the held-out test set,
  then reports:
    • Accuracy, Precision, Recall, F1-Score
    • Confusion Matrix  (saved as PNG)
    • ROC-AUC Curve     (saved as PNG)
    • Sample ECG plots  (saved as PNG)

  Run:  python scripts/evaluate.py
==============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# ---------------------------------------------
# PATHS
# ---------------------------------------------
DATA_DIR   = "data"
MODEL_PATH = os.path.join("models", "ecg_cnn.keras")
RESULT_DIR = "results"
THRESHOLD  = 0.5       # probability cutoff for class 1 (Arrhythmia)

os.makedirs(RESULT_DIR, exist_ok=True)


# ----------------------------------------------
# 1. LOAD MODEL & TEST DATA
# ----------------------------------------------

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    model  = tf.keras.models.load_model(MODEL_PATH)
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"[INFO] Model   : {MODEL_PATH}")
    print(f"[INFO] X_test  : {X_test.shape}")
    print(f"[INFO] y_test  : {y_test.shape}")
    return model, X_test, y_test


# ----------------------------------------------
# 2. PREDICT
# ----------------------------------------------

def predict(model, X_test):
    # model outputs P(Arrhythmia) ∈ [0, 1]
    y_prob = model.predict(X_test, batch_size=256, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return y_prob, y_pred


# ----------------------------------------------
# 3. METRICS REPORT
# ----------------------------------------------

def print_metrics(y_test, y_pred, y_prob):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    print("\n" + "="*50)
    print("  TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}   (of predicted Arrhy, how many correct?)")
    print(f"  Recall    : {rec:.4f}   (of actual Arrhy, how many detected?)")
    print(f"  F1-Score  : {f1:.4f}   (harmonic mean of Prec & Recall)")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("="*50)

    print("\nFull classification report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Arrhythmia"]))
    return fpr, tpr, roc_auc


# ----------------------------------------------
# 4. CONFUSION MATRIX
# ----------------------------------------------

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Arrhythmia"],
        yticklabels=["Normal", "Arrhythmia"],
        ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved  {path}")


# ----------------------------------------------
# 5. ROC CURVE
# ----------------------------------------------

def plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved  {path}")


# ----------------------------------------------
# 6. SAMPLE ECG PLOTS
# ----------------------------------------------

def plot_sample_ecg(X_test, y_test, y_pred, y_prob, n_samples=6):
    """
    Plot a grid of sample ECG beats with their true label,
    predicted label, and confidence score.
    """
    # Pick 3 Normal and 3 Arrhythmia samples
    norm_idx  = np.where(y_test == 0)[0][:3]
    arrhy_idx = np.where(y_test == 1)[0][:3]
    indices   = np.concatenate([norm_idx, arrhy_idx])

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    axes = axes.ravel()

    time_axis = np.linspace(0, 1000, X_test.shape[1])   # ~1 second window in ms

    class_names  = ["Normal", "Arrhythmia"]
    class_colors = ["steelblue", "crimson"]

    for ax, idx in zip(axes, indices):
        beat       = X_test[idx].ravel()
        true_cls   = y_test[idx]
        pred_cls   = y_pred[idx]
        confidence = y_prob[idx] if pred_cls == 1 else 1 - y_prob[idx]

        color = class_colors[pred_cls]
        ax.plot(time_axis, beat, color=color, linewidth=1.5)
        ax.set_title(
            f"True: {class_names[true_cls]}  |  "
            f"Pred: {class_names[pred_cls]}  ({confidence*100:.1f}%)",
            fontsize=9,
            color="green" if true_cls == pred_cls else "red"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (normalised)")
        ax.grid(alpha=0.3)

    plt.suptitle("Sample ECG Beat Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "sample_ecg_plots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved  {path}")


# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    model, X_test, y_test = load_artifacts()
    y_prob, y_pred         = predict(model, X_test)
    fpr, tpr, roc_auc      = print_metrics(y_test, y_pred, y_prob)

    plot_confusion_matrix(X_test, y_pred) if False else \
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(fpr, tpr, roc_auc)
    plot_sample_ecg(X_test, y_test, y_pred, y_prob)

    print("\n[INFO] All evaluation plots saved to  results/")


if __name__ == "__main__":
    main()
