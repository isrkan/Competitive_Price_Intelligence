import tensorflow as tf


def masked_mse(y_true_with_mask, y_pred):
    """
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