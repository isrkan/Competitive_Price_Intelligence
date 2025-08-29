import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .model_utils import symmetric_mean_absolute_percentage_error, r2_score

def train_gru(
    X_train, y_train, X_val, y_val,
    **kwargs
):
    """
    Build, train, and return a GRU model for price forecasting.

    Parameters:
    - X_train (np.ndarray): Training input windows, shape (n_samples, lookback).
    - y_train (np.ndarray): Training targets, shape (n_samples, horizon).
    - X_val (np.ndarray): Validation input windows.
    - y_val (np.ndarray): Validation targets.

    Hyperparameters (kwargs):
    - forecasting_model_params (dict): Dictionary of hyperparameters passed from config.
        - gru_units (int): Number of GRU units.
        - dense_units (int): Units in dense layer.
        - dropout_rate (float): Dropout rate.
        - epochs (int): Training epochs.
        - batch_size (int): Batch size.
        - learning_rate (float): Adam learning rate.
        - callbacks: Optional list of callbacks.

    Returns:
    - model (tf.keras.Model): Trained GRU model.
    - history: Training history (loss, val_loss, etc.)
    """
    # Validate inputs
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D: (n_samples, lookback).")
    if y_train.ndim != 2:
        raise ValueError("y_train must be 2D: (n_samples, horizon).")

    # ----------------------------------------------------
    # Extract hyperparameters with defaults
    # ----------------------------------------------------
    gru_units = kwargs.get("gru_units", 64)
    dense_units = kwargs.get("dense_units", 64)
    dropout_rate = kwargs.get("dropout_rate", 0.3)
    epochs = kwargs.get("epochs", 50)
    batch_size = kwargs.get("batch_size", 32)
    learning_rate = kwargs.get("learning_rate", 1e-3)
    callbacks = kwargs.get("callbacks", None)

    # Sequence lengths
    lookback = X_train.shape[1]  # timesteps in input window
    horizon = y_train.shape[1]  # timesteps to forecast

    # ---------------------------
    # Reshape inputs
    # ---------------------------
    # Reshape inputs for GRU → expects 3D (n, timesteps, features)
    # We only have 1 feature (the price), so expand last dim. Shape becomes (n_samples, lookback, 1)
    X_train = X_train[..., np.newaxis]  # shape (n, lookback, 1)
    X_val = X_val[..., np.newaxis]      # shape (n, lookback, 1)

    # ----------------------------------------------------
    # Build the GRU model architecture
    # ----------------------------------------------------
    # Sequence input (sliding window of prices)
    sequence_input = Input(shape=(lookback, 1), name="sequence_input")

    # GRU encoder: captures temporal dependencies
    # return_sequences=False → only final hidden state is used (suitable for forecasting horizon)
    x = GRU(gru_units, return_sequences=False)(sequence_input)
    x = Dropout(dropout_rate)(x)  # Regularize to avoid overfitting

    # Fully connected layers to transform hidden state into forecast
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)

    # Output layer: predict horizon steps ahead (vector output)
    output = Dense(horizon, activation="linear", name="forecast_output")(x)

    # Define model
    model = Model(inputs=sequence_input, outputs=output)

    # -----------------------
    # Compile model
    # -----------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            "mae",
            symmetric_mean_absolute_percentage_error,  # SMAPE (%)
            r2_score  # R^2 (goodness of fit)
        ]
    )

    # -----------------------
    # Callbacks
    # -----------------------
    if callbacks is None:
        callbacks = [
            # Stop early when val loss hasn't improved for some epochs (prevents overfitting)
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            # Reduce learning-rate when val loss plateaus (helps escape shallow minima)
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, verbose=1)
        ]

    # ----------------------------------------------------
    # Train the Model
    # ----------------------------------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history