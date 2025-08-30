import numpy as np
import tensorflow as tf
from ..forecasting import forecasting_model_utils

def evaluate_forecasting_model(model, X_test, y_test):
    """
    Evaluate a trained forecasting model on test data using standard regression metrics.

    Metrics:
    - MSE: Mean Squared Error
    - MAE: Mean Absolute Error
    - sMAPE: Symmetric Mean Absolute Percentage Error
    - R^2: Coefficient of Determination

    Parameters:
    - model (tf.keras.Model): trained forecasting model
    - X_test (np.ndarray): test input windows, shape (n_samples, lookback, num_features)
    - y_test (np.ndarray): true future values, shape (n_samples, horizon, num_features=1)

    Returns:
    - results (dict): {"mse": float, "mae": float, "smape": float, "r2": float}
    """
    try:
        # Run model prediction
        y_pred = model.predict(X_test, verbose=0)

        # Ensure shapes are compatible: squeeze last axis if needed
        if y_test.ndim == 3 and y_test.shape[-1] == 1:
            y_test = np.squeeze(y_test, axis=-1)  # (n_samples, horizon)
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)  # (n_samples, horizon)

        # Convert to tensors for metric computations
        y_true_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)
        y_pred_tf = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Compute metrics
        mse_value = tf.reduce_mean(tf.square(y_true_tf - y_pred_tf)).numpy()
        mae_value = tf.reduce_mean(tf.abs(y_true_tf - y_pred_tf)).numpy()
        smape_value = forecasting_model_utils.smape(y_true_tf, y_pred_tf).numpy()
        r2_value = forecasting_model_utils.r2_score(y_true_tf, y_pred_tf).numpy()

        results = {
            "mse": mse_value,
            "mae": mae_value,
            "smape": smape_value,
            "r2": r2_value
        }

        return results
    except Exception as e:
        raise RuntimeError(f"[evaluate_forecasting_model] Error during evaluation: {e}")