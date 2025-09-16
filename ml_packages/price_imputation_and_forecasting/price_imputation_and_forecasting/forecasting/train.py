from .gru import train_gru
from .lstm import train_lstm
from .rnn import train_rnn

def train_forecasting_model(
    model_name, forecasting_model_params,
    X_train, y_train,
    X_val, y_val,
):
    """
    Dispatcher function to train a price forecasting model based on the specified model_name.

    Parameters:
    - model_name (str): Name of the model to train (currently only 'gru', 'lstm', 'rnn' supported).
    - forecasting_model_params (dict): Dictionary of model hyperparameters.
    - X_train (np.ndarray): Training input windows, shape (n_samples, lookback).
    - y_train (np.ndarray): Training targets, shape (n_samples, horizon).
    - X_val (np.ndarray): Validation input windows.
    - y_val (np.ndarray): Validation targets.

    Returns:
    - model (tf.keras.Model): Trained Keras model.
    - history (keras.callbacks.History): Training history.
    """
    try:
        # Validate model_name
        valid_models = {"gru", "lstm", "rnn"}
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if model_name not in valid_models:
            raise ValueError(f"Invalid model_name '{model_name}'. Choose from {valid_models}.")

        # Dispatch to the correct model
        if model_name == "gru":
            from .gru import train_gru
            model, history = train_gru(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **forecasting_model_params  # <- inject hyperparameters from config
            )
        elif model_name == "lstm":
            from .lstm import train_lstm
            model, history = train_lstm(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **forecasting_model_params
            )
        elif model_name == "rnn":
            from .rnn import train_rnn
            model, history = train_rnn(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **forecasting_model_params  # <- inject hyperparameters from config
            )
        else:
            raise NotImplementedError(f"Model '{model_name}' is not implemented yet.")

        return model, history

    except Exception as e:
        print(f"Error in train_price_forecasting_model: {e}")
        raise