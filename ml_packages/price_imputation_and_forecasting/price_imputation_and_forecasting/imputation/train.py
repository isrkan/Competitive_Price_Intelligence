import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Bidirectional, SimpleRNN, Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from .model_utils import masked_mse, masked_mae

def bi_rnn(
    X_sequence_train, X_static_train, target_mask_train, y_train, X_sequence_val, X_static_val, target_mask_val, y_val,
    seq_units=64, dense_units=64, dropout_rate=0.3, epochs=5, batch_size=32, callbacks=None
):
    """
    Build, train, and save a BiRNN (Bi-LSTM) model for price imputation.

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

    Hyperparameters:
    - seq_units (int): Number of BiRNN units.
    - dense_units (int): Number of dense layer units after concatenation.
    - dropout_rate (float): Dropout rate.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size.
    - callbacks: Optional list of Keras callbacks (if None, defaults with EarlyStopping & ReduceLROnPlateau)

    Returns:
    - model (tf.keras.Model): Trained model.
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
    # Extract Input Shapes
    # ----------------------------------------------------
    # Define sequence input (time series + mask)
    # Calculate the number of time steps per series (e.g. ~731 days).
    seq_len = X_sequence_train.shape[1]  # number of time steps (T)
    # Calculate the number of channels per timestep (we used price and obs_mask, so usually 2).
    seq_features = X_sequence_train.shape[2]  # features per timestep (e.g. 2)

    # Define static input (store dummies)
    static_features = X_static_train.shape[1]  # static features dimension (D)

    # ----------------------------------------------------
    # Sequence Input (Bi-LSTM for temporal features)
    # ----------------------------------------------------
    # Sequence input (price_with_gaps, obs_mask)
    sequence_input = Input(shape=(seq_len, seq_features), name="sequence_input")
    # Bi-LSTM encodes temporal context both forwards and backwards. return_sequences=True because we need predictions at every time step.
    x = Bidirectional(SimpleRNN(seq_units, return_sequences=True))(sequence_input)
    x = Dropout(dropout_rate)(x)  # Regularize the sequence encoding

    # ----------------------------------------------------
    # Static Input (dense layers for store metadata)
    # ----------------------------------------------------
    static_input = Input(shape=(static_features,), name="static_input")

    s = Dense(dense_units, activation="relu")(static_input)
    s = Dropout(dropout_rate)(s)

    # Expand static features across timesteps, so we can concat them to RNN outputs
    # RepeatVector creates a (batch, seq_len, dense_units) tensor by copying the static vector T times
    # This allows concatenation with the RNN outputs (batch, T, hidden_dim) to create (batch, T, hidden_dim + D)
    # at each time step the model gets the same store metadata, so it can condition its predictions on store-level features
    s_expanded = tf.keras.layers.RepeatVector(seq_len)(s)

    # ----------------------------------------------------
    # Combine Sequence and Static Features
    # ----------------------------------------------------
    combined = Concatenate(axis=-1)([x, s_expanded])  # Shape: (n, T, seq_units*2 + dense_units)

    # ----------------------------------------------------
    # Decoder: Fully Connected Layers to predict prices
    # ----------------------------------------------------
    # Decoder dense layers → output imputed prices
    out = Dense(dense_units, activation="relu")(combined)
    out = Dropout(dropout_rate)(out)
    # Output layer: predict price at each timestep
    output = Dense(1, activation="linear", name="price_output")(out)

    # ----------------------------------------------------
    # Define and Compile the Model
    # ----------------------------------------------------
    # Define model
    model = Model(inputs=[sequence_input, static_input], outputs=output)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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
    # Callbacks (early stop, LR scheduler, checkpoint)
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
        [X_sequence_train, X_static_train], y_train_masked,
        validation_data=([X_sequence_val, X_static_val], y_val_masked),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history