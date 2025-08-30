import tensorflow as tf
import json
import os


def smape(y_true, y_pred):
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    Parameters:
    y_true: tensor of true values, shape (...,)
    y_pred: tensor of predicted values, shape (...,)

    Returns:
    scalar sMAPE value in percentage terms (range: [0, 200])
    """
    # denominator is the sum of magnitudes of true and predicted values
    # add small epsilon to avoid division by zero when both y_true and y_pred are 0
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) + tf.keras.backend.epsilon()

    # percentage error at each sample
    smape_per_sample = 200.0 * tf.abs(y_true - y_pred) / denominator

    # mean across all samples
    return tf.reduce_mean(smape_per_sample)

def r2_score(y_true, y_pred):
    """
    Compute coefficient of determination (R^2 score).

    Parameters:
    y_true: tensor of true values, shape (...,)
    y_pred: tensor of predicted values, shape (...,)

    Returns:
    scalar R^2 value:
        - 1.0 = perfect predictions
        - 0.0 = model no better than mean baseline
        - negative = model worse than predicting mean
    """
    # residual sum of squares
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))

    # total sum of squares relative to mean of y_true
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

    # compute R^2, with epsilon in denominator to avoid division by zero
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


def save_forecasting_model(model, history, save_path):
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