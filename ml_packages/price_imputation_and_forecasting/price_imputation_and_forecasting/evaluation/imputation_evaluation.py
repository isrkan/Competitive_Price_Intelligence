import numpy as np
import tensorflow as tf
from ..imputation import model_utils

def evaluate_model(model, X_sequence, X_static, y_true, target_mask):
    """
    Evaluate a trained model using masked MSE and masked MAE.

    Parameters:
    - model (tf.keras.Model): trained model
    - X_sequence (np.ndarray): sequence input, shape (n_samples, seq_len, num_features)
    - X_static (np.ndarray): static input, shape (n_samples, num_static_features)
    - y_true (np.ndarray): ground truth prices, shape (n_samples, seq_len)
    - target_mask (np.ndarray): mask indicating valid targets, shape (n_samples, seq_len)

    Returns:
    - results (dict): {"masked_mse": float, "masked_mae": float}
    """
    try:
        # Run model prediction
        y_pred = model.predict([X_sequence, X_static], verbose=0)

        # Ensure shapes align: expand last axis for true values + mask
        if y_true.ndim == 2:
            y_true = np.expand_dims(y_true, axis=-1)  # (n_samples, seq_len, 1)
        if target_mask.ndim == 2:
            target_mask = np.expand_dims(target_mask, axis=-1)  # (n_samples, seq_len, 1)

        # Prepare y_true_with_mask for evaluation
        y_true_with_mask = np.concatenate([y_true, target_mask], axis=-1)  # (n_samples, seq_len, 2)

        # Compute masked losses
        mse_value = model_utils.masked_mse(y_true_with_mask, y_pred).numpy()
        mae_value = model_utils.masked_mae(y_true_with_mask, y_pred).numpy()

        results = {"masked_mse": mse_value, "masked_mae": mae_value}

        return results
    except Exception as e:
        raise RuntimeError(f"[evaluate_model] Error during evaluation: {e}")