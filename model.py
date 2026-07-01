"""
==============================================================
  CNN Model Definition – ECG Arrhythmia Classifier
==============================================================
  Architecture:
    Input (360,1)
      │
      ├─ Conv1D(32)  + BN + ReLU + MaxPool + Dropout
      ├─ Conv1D(64)  + BN + ReLU + MaxPool + Dropout
      ├─ Conv1D(128) + BN + ReLU + MaxPool + Dropout
      │
      ├─ GlobalAveragePooling1D
      │
      ├─ Dense(128) + ReLU + Dropout
      └─ Dense(1)   + Sigmoid   →  P(Arrhythmia)

  Why 1D-CNN for ECG?
    ECG is a time-series signal.  A 1-D convolution slides a
    small "window" along the time axis and learns local waveform
    patterns (Q, R, S morphology, ST shape, etc.) automatically
    — no hand-crafted feature engineering needed.

  Import this module in train.py and inference.py.
==============================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn(input_length: int = 360, l2: float = 1e-4) -> tf.keras.Model:
    """
    Build and return an uncompiled Keras CNN model.

    Parameters
    ----------
    input_length : int
        Number of time-steps per beat (default 360 = 2 × WINDOW).
    l2 : float
        L2 regularisation strength on Conv and Dense kernels.

    Returns
    -------
    model : tf.keras.Model
    """

    inputs = tf.keras.Input(shape=(input_length, 1), name="ecg_input")

    # --- Block 1 ----------------------------------------------------
    # 32 filters, each of width 11 samples (~30 ms at 360 Hz).
    # BatchNorm speeds up training by keeping activations well-scaled.
    x = layers.Conv1D(32, kernel_size=11, padding="same",
                      kernel_regularizer=regularizers.l2(l2), name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # --- Block 2 ----------------------------------------------------
    x = layers.Conv1D(64, kernel_size=7, padding="same",
                      kernel_regularizer=regularizers.l2(l2), name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
    x = layers.Dropout(0.2, name="drop2")(x)

    # --- Block 3 ----------------------------------------------------
    x = layers.Conv1D(128, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(l2), name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)
    x = layers.Dropout(0.3, name="drop3")(x)

    # --- Global pooling replaces Flatten ----------------------------------
    # GlobalAveragePooling averages each feature map over time.
    # It reduces parameters and improves generalisation vs Flatten.
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # --- Dense head -----------------------------------------------------
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2), name="dense1")(x)
    x = layers.Dropout(0.4, name="drop4")(x)

    # Sigmoid output → probability of class 1 (Arrhythmia)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="ECG_CNN")
    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_cnn()
    m.summary()
