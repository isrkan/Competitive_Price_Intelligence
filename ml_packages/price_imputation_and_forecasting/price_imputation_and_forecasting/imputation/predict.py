import tensorflow as tf
import numpy as np

def run_model_prediction(model, X_sequence, X_static):
    """
    Run prediction using a trained model and prepared inputs.

    Parameters:
    - model (tf.keras.Model): Trained Keras model.
    - X_sequence (np.ndarray): Sequence input, shape (n_samples, seq_len, num_features).
    - X_static (np.ndarray): Static input, shape (n_samples, num_static_features).

    Returns:
    - y_pred (np.ndarray): Model predictions, shape (n_samples, seq_len, 1).
    """
    try:
        # Run the forward pass of the trained model
        y_pred = model.predict([X_sequence, X_static], verbose=0)
        return y_pred
    except Exception as e:
        raise RuntimeError(f"[run_model_prediction] Error during prediction: {e}")


def replace_missing_with_predictions(df_price_input, y_pred, fill_nan_value):
    """
    Replace values in df_scaled_price_input that equal `fill_nan_value` with model predictions.

    Parameters:
    - df_price_input (np.ndarray): Price input matrix, shape (n_samples, seq_len).
    - y_pred (np.ndarray): Model predictions, shape (n_samples, seq_len, 1).
    - fill_nan_value (float): Placeholder value used for missing entries.

    Returns:
    - imputed_data (np.ndarray): Data with missing values replaced by predictions, same shape as df_price_input.
    """
    try:
        # Ensure predictions are squeezed to match (n_samples, seq_len)
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)

        # Copy input data to avoid modifying original in-place
        imputed_data = df_price_input.copy()

        # Find indices where data equals the fill_nan_value
        missing_indices = np.where(df_price_input == fill_nan_value)

        # Replace with corresponding predictions
        imputed_data[missing_indices] = y_pred[missing_indices]

        return imputed_data
    except Exception as e:
        raise RuntimeError(f"Failed to replace missing values with predictions in. Details: {e}")