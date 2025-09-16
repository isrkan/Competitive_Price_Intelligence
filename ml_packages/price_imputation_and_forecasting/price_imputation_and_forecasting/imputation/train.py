from .bi_rnn import train_bi_rnn
from .bi_lstm import train_bi_lstm
from .bi_gru import train_bi_gru


def train_imputation_model(
    model_name, imputation_model_params,
    X_sequence_train, X_static_train, target_mask_train, y_train,
    X_sequence_val, X_static_val, target_mask_val, y_val,
):
    """
    Dispatcher function to train a price imputation model based on the specified model_name.

    Parameters:
    - model_name (str): Name of the model to train (currently only 'bi_rnn', 'bi_lstm', 'bi_gru' supported).
    - imputation_model_params (dict): Dictionary of model hyperparameters.
    - X_sequence_train (np.ndarray): Training sequence input (n, seq_len, seq_features).
    - X_static_train (np.ndarray): Training static input (n, static_features).
    - target_mask_train (np.ndarray): Mask for training targets (n, seq_len, 1).
    - y_train (np.ndarray): Training target values (n, seq_len, 1).
    - X_sequence_val (np.ndarray): Validation sequence input.
    - X_static_val (np.ndarray): Validation static input.
    - target_mask_val (np.ndarray): Mask for validation targets.
    - y_val (np.ndarray): Validation target values.

    Returns:
    - model (tf.keras.Model): Trained Keras model.
    - history (keras.callbacks.History): Training history object.
    """
    try:
        # Validate model_name
        valid_models = {"bi_rnn", "bi_lstm", "bi_gru"}
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if model_name not in valid_models:
            raise ValueError(f"Invalid model_name '{model_name}'. Choose from {valid_models}.")


        # Dispatch to the correct model
        if model_name == "bi_rnn":
            model, history = train_bi_rnn(
                X_sequence_train=X_sequence_train,
                X_static_train=X_static_train,
                target_mask_train=target_mask_train,
                y_train=y_train,
                X_sequence_val=X_sequence_val,
                X_static_val=X_static_val,
                target_mask_val=target_mask_val,
                y_val=y_val,
                **imputation_model_params  # <- inject hyperparameters from config
            )
        elif model_name == "bi_lstm":
            model, history = train_bi_lstm(
                X_sequence_train=X_sequence_train,
                X_static_train=X_static_train,
                target_mask_train=target_mask_train,
                y_train=y_train,
                X_sequence_val=X_sequence_val,
                X_static_val=X_static_val,
                target_mask_val=target_mask_val,
                y_val=y_val,
                **imputation_model_params
            )
        elif model_name == "bi_gru":
            model, history = train_bi_gru(
                X_sequence_train=X_sequence_train,
                X_static_train=X_static_train,
                target_mask_train=target_mask_train,
                y_train=y_train,
                X_sequence_val=X_sequence_val,
                X_static_val=X_static_val,
                target_mask_val=target_mask_val,
                y_val=y_val,
                **imputation_model_params
            )
        else:
            raise NotImplementedError(f"Model '{model_name}' is not implemented yet.")

        return model, history

    except Exception as e:
        print(f"Error in train_price_imputation_model: {e}")
        raise