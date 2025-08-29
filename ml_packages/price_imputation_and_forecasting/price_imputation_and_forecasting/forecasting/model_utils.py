import tensorflow as tf

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
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