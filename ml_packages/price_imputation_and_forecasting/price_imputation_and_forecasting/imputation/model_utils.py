import tensorflow as tf
import json
import os


def masked_mse(y_true_with_mask, y_pred):
    """
    Compute masked Mean Squared Error (MSE).

    Parameters:
    y_true_with_mask: [..., 2] last-dim contains [true_value, mask]
      - true_value shape: (..., 1)
      - mask shape: (..., 1)  (1=compute loss here, 0=ignore)
    y_pred: predicted values, shape (..., 1)

    Returns: scalar MSE computed only where mask == 1
    """
    # extract true value and mask from the provided 'y_true_with_mask'
    y_true = y_true_with_mask[..., 0:1]  # shape (..., 1)
    mask = y_true_with_mask[..., 1:2]  # shape (..., 1)

    # squared error at every timestep / sample and zero out errors where mask == 0 (i.e., where we have no supervision)
    squared_error = tf.square(y_true - y_pred) * mask

    # sum errors over batch/time and normalize by number of active mask positions
    # add small eps to avoid division by zero if mask has no 1s in a batch
    return tf.reduce_sum(squared_error) / (tf.reduce_sum(mask) + 1e-6)


def masked_mae(y_true_with_mask, y_pred):
    """
    Compute masked Mean Absolute Error (MAE).

    Parameters:
    y_true_with_mask: [..., 2] last-dim contains [true_value, mask]
      - true_value shape: (..., 1)
      - mask shape: (..., 1)  (1=compute loss here, 0=ignore)
    y_pred: predicted values, shape (..., 1)

    Returns: scalar MAE computed only where mask == 1
    """
    # extract true value and mask from the provided 'y_true_with_mask'
    y_true = y_true_with_mask[..., 0:1]  # shape (..., 1)
    mask = y_true_with_mask[..., 1:2]  # shape (..., 1)

    # absolute error at every timestep / sample and zero out errors where mask == 0 (i.e., where we have no supervision)
    abs_error = tf.abs(y_true - y_pred) * mask

    # sum errors over batch/time and normalize by number of active mask positions
    # add small eps to avoid division by zero if mask has no 1s in a batch
    return tf.reduce_sum(abs_error) / (tf.reduce_sum(mask) + 1e-6)


def save_model(model, history, save_path):
    """
    Save a trained Keras model and its training history.

    Parameters:
    - model (tf.keras.Model): Trained model instance.
    - history (keras.callbacks.History): Training history (loss, metrics).
    - save_path (str): Directory path where model and history will be saved.
    """
    # Validate model type
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Expected 'model' to be a tf.keras.Model instance.")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    try:
        # Save model as .h5 file
        model_path = os.path.join(save_path, "model.h5")
        model.save(model_path)

        # Save training history as JSON
        history_path = os.path.join(save_path, "history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)
    except Exception as e:
        raise RuntimeError(f"Error saving model or history: {e}")


def load_model(save_path):
    """
    Load a trained Keras model.

    Parameters:
    - save_path (str): Directory path where model and history were saved.

    Returns:
    - model (tf.keras.Model): Loaded Keras model instance.
    """
    # Ensure provided path exists
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Provided save path does not exist: {save_path}")

    try:
        # Load model from .h5 file
        model_path = os.path.join(save_path, "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)

        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model or history: {e}")