import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Bidirectional, GRU, Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from .imputation_model_utils import masked_mse, masked_mae

def train_bi_gru(
    X_sequence_train, X_static_train, target_mask_train, y_train, X_sequence_val, X_static_val, target_mask_val, y_val,
    **kwargs
):
    """
    Build, train, and save a Bi-GRU model for price imputation.

    Data Inputs:

    - Sequence Inputs:
        - X_sequence_train (np.ndarray): Sequence features (price series + mask), shape (n_samples, seq_len, num_features).
        - X_sequence_val (np.ndarray): Validation sequence features.
    - Static Inputs:
        - X_static_train (np.ndarray): Static features (store dummies), shape (n_samples, num_static_features).
        - X_static_val (np.ndarray): Validation static features.
    - Targets:
        - y_train (np.ndarray): Ground truth price series (scaled), shape (n_samples, seq_len).
        - y_val (np.ndarray): Validation ground truth.

    Hyperparameters (kwargs):
    - imputation_model_params (dict): Dictionary of hyperparameters passed from config.
        - seq_units (int): Number of BiGRU units.
        - dense_units (int): Number of dense layer units after concatenation.
        - dropout_rate (float): Dropout rate.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size.
        - callbacks: Optional list of Keras callbacks (if None, defaults with EarlyStopping & ReduceLROnPlateau)

    Returns:
    - model (tf.keras.Model): Trained Bi-GRU model.
    - history: Keras training history (loss, val_loss, etc.)
    """
    # Ensure sequence tensors are 3D and static tensors are 2D
    if X_sequence_train.ndim != 3:
        raise ValueError("X_sequence_train must be 3D: (n, seq_len, seq_features).")
    if X_sequence_val.ndim != 3:
        raise ValueError("X_sequence_val must be 3D: (n, seq_len, seq_features)")
    if X_static_train.ndim != 2:
            raise ValueError("X_static_train must be 2D: (n, n_static).")

    # ----------------------------------------------------
    # Extract hyperparameters with defaults
    # ----------------------------------------------------
    seq_units = kwargs.get("seq_units", 128)  # per-direction units for each SimpleGRU
    dense_units = kwargs.get("dense_units", 128)  # dense units used in static pathway and decoder
    dropout_rate = kwargs.get("dropout_rate", 0.2)
    recurrent_dropout = kwargs.get("recurrent_dropout", 0.1)  # regularize GRU recurrent connections
    epochs = kwargs.get("epochs", 30)
    batch_size = kwargs.get("batch_size", 64)
    learning_rate = kwargs.get("learning_rate", 1e-3)
    callbacks = kwargs.get("callbacks", None)

    # ----------------------------------------------------
    # Extract input shapes
    # ----------------------------------------------------
    # Define sequence input (time series + mask)
    # Calculate the number of time steps per series (e.g. ~731 days).
    seq_len = X_sequence_train.shape[1]  # number of time steps (T)
    # Calculate the number of channels per timestep (we used price and obs_mask, so usually 2).
    seq_features = X_sequence_train.shape[2]  # features per timestep (e.g. 2)

    # Define static input (store dummies)
    static_features = X_static_train.shape[1]  # static features dimension (D)

    # ----------------------------------------------------
    # Sequence input (Bi-LSTM for temporal features)
    # ----------------------------------------------------
    # Sequence input (price_with_gaps, obs_mask)
    sequence_input = Input(shape=(seq_len, seq_features), name="sequence_input")
    # Bi-LSTM encodes temporal context both forwards and backwards. return_sequences=True because we need predictions at every time step.
    # First BiGRU block (returns full sequence)
    x = Bidirectional(GRU(seq_units, return_sequences=True, activation="tanh", recurrent_dropout=recurrent_dropout), name="bidir_gru_1")(sequence_input)
    x = Dropout(dropout_rate, name="dropout_gru_1")(x)  # Regularize the sequence encoding

    # Second BiGRU block (returns full sequence)
    x2 = Bidirectional(GRU(seq_units, return_sequences=True, activation="tanh", recurrent_dropout=recurrent_dropout), name="bidir_gru_2")(x)
    x2 = Dropout(dropout_rate, name="dropout_gru_2")(x2)

    # ----------------------------------------------------
    # Static input (dense layers for store metadata)
    # ----------------------------------------------------
    static_input = Input(shape=(static_features,), name="static_input")

    s = Dense(dense_units, activation="relu")(static_input)
    s = Dropout(dropout_rate)(s)

    # Expand static features across timesteps, so we can concat them to GRU outputs
    # RepeatVector creates a (batch, seq_len, dense_units) tensor by copying the static vector T times
    # This allows concatenation with the GRU outputs (batch, T, hidden_dim) to create (batch, T, hidden_dim + D)
    # at each time step the model gets the same store metadata, so it can condition its predictions on store-level features
    s_expanded = tf.keras.layers.RepeatVector(seq_len)(s)

    # ----------------------------------------------------
    # Combine Sequence and Static Features
    # ----------------------------------------------------
    combined = Concatenate(axis=-1)([x2, s_expanded])  # Shape: (n, T, seq_units*2 + dense_units)
    # Projector to mix features before decoder
    combined = Dense(dense_units, activation="relu", name="combined_projector")(combined)
    combined = Dropout(dropout_rate, name="combined_dropout")(combined)

    # ----------------------------------------------------
    # Decoder: Fully connected layers to predict prices
    # ----------------------------------------------------
    # Decoder dense layers → output imputed prices
    out = Dense(dense_units, activation="relu")(combined)
    out = Dropout(dropout_rate)(out)
    # Output layer: predict price at each timestep
    output = Dense(1, activation="linear", name="price_output")(out)

    # ----------------------------------------------------
    # Define and compile the model
    # ----------------------------------------------------
    # Define model
    model = Model(inputs=[sequence_input, static_input], outputs=output)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_mse,
        metrics=[masked_mae]
    )

    # -----------------------
    # Prepare masked targets for training
    # -----------------------
    # Combine true values with their target masks
    y_train_masked = np.concatenate([y_train, target_mask_train], axis=-1)  # Shape: (n_samples, seq_len, 2)
    y_val_masked = np.concatenate([y_val, target_mask_val], axis=-1)  # Shape: (n_samples, seq_len, 2)

    # -----------------------
    # Callbacks
    # -----------------------
    if callbacks is None:
        callbacks = [
            # Stop early when val loss hasn't improved for some epochs (prevents overfitting)
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            # Reduce learning-rate when val loss plateaus (helps escape shallow minima)
            ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, verbose=1)
        ]

    # ----------------------------------------------------
    # Train the model
    # ----------------------------------------------------
    history = model.fit(
        [X_sequence_train, X_static_train], y_train_masked,
        validation_data=([X_sequence_val, X_static_val], y_val_masked),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history