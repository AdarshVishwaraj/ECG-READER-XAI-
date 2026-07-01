"""
==============================================================
  Training Script – ECG CNN Arrhythmia Classifier
==============================================================
  What this script does:
    1. Loads preprocessed data (data/X.npy, data/y.npy)
    2. Handles class imbalance with class weights
    3. Splits → Train / Validation / Test
    4. Builds the CNN (from model.py)
    5. Trains with early stopping + learning-rate scheduling
    6. Saves the best model to  models/ecg_cnn.keras
    7. Plots training curves and saves them to  results/

  Run:  python scripts/train.py
  (Run preprocess.py first if data/ arrays don't exist yet)
==============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")                  # headless (no display needed)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Import our model builder
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import build_cnn

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------
DATA_DIR   = "data"
MODEL_DIR  = "models"
RESULT_DIR = "results"

EPOCHS     = 30          # increase to 50-100 for a final model
BATCH_SIZE = 128
LR         = 1e-3        # initial learning rate
VAL_SPLIT  = 0.15        # 15 % of train used for validation
TEST_SPLIT  = 0.15        # 15 % of whole dataset held out

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ----------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------

def load_data():
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    if not os.path.exists(X_path):
        raise FileNotFoundError(
            "data/X.npy not found.  Run  python scripts/preprocess.py  first."
        )

    X = np.load(X_path)   # shape: (N, 360)
    y = np.load(y_path)   # shape: (N,)

    # CNN expects shape (N, 360, 1)  — add channel dimension
    X = X[..., np.newaxis]

    print(f"[INFO] Loaded X {X.shape}  y {y.shape}")
    return X, y


# ----------------------------------------------
# 2. SPLIT
# ----------------------------------------------

def split_data(X, y):
    # First cut: separate test set (stratified so class ratios are preserved)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    # Second cut: train vs validation
    val_frac_of_trainval = VAL_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac_of_trainval, random_state=SEED, stratify=y_train_val
    )

    print(f"\n[INFO] Split sizes:")
    print(f"       Train : {len(y_train):,}")
    print(f"       Val   : {len(y_val):,}")
    print(f"       Test  : {len(y_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------------------------
# 3. CLASS WEIGHTS (handles imbalance)
# ----------------------------------------------

def get_class_weights(y_train):
    """
    MIT-BIH is heavily imbalanced: ~75 % Normal, ~25 % Arrhythmia.
    class_weight tells Keras to penalise errors on the minority class more.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    print(f"\n[INFO] Class weights: {cw}")
    return cw


# ----------------------------------------------
# 4. BUILD & COMPILE MODEL
# ----------------------------------------------

def compile_model():
    model = build_cnn(input_length=360)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",   # binary → 2 classes (Normal vs Arrhythmia)
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    model.summary()
    return model


# ----------------------------------------------
# 5. CALLBACKS
# ----------------------------------------------

def get_callbacks():
    # ModelCheckpoint: save only the best epoch (by val_loss)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "ecg_cnn.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # EarlyStopping: stop if val_loss does not improve for `patience` epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    # ReduceLROnPlateau: halve LR when val_loss stalls
    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    return [ckpt, early_stop, lr_sched]


# ----------------------------------------------
# 6. PLOT TRAINING CURVES
# ----------------------------------------------

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Loss -------------------------------
    axes[0].plot(history.history["loss"],     label="Train Loss",      linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    axes[0].set_title("Training vs Validation Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ----- Accuracy -------------------------------
    axes[1].plot(history.history["accuracy"],     label="Train Accuracy",      linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    axes[1].set_title("Training vs Validation Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved  {path}")


# ----------------------------------------------
# MAIN
# ----------------------------------------------

def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    class_weights = get_class_weights(y_train)
    model = compile_model()
    callbacks = get_callbacks()

    print(f"\n[INFO] Starting training …  (epochs={EPOCHS}, batch={BATCH_SIZE})\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    plot_training_curves(history)

    # ── Save test split so evaluate.py can load the same split ───────
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    print("\n[INFO] Saved test split → data/X_test.npy  data/y_test.npy")
    print("\n[INFO] Training complete.  Best model saved to  models/ecg_cnn.keras")


if __name__ == "__main__":
    main()
