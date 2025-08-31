import numpy as np
import tensorflow as tf

def run_forecasting_prediction(model, X_sequence):
    """
    Run prediction using a trained forecasting model and prepared inputs.

    Parameters:
    - model (tf.keras.Model): Trained Keras forecasting model.
    - X_sequence (np.ndarray): Sequence input for forecasting, shape (n_samples, lookback, num_features).

    Returns:
    - y_pred (np.ndarray): Model predictions, shape (n_samples, horizon, 1).
    """
    try:
        # Create model prediction
        y_pred = model.predict(X_sequence, verbose=0)
        return y_pred
    except Exception as e:
        raise RuntimeError(f"[run_forecasting_prediction] Error during prediction: {e}")


def append_forecast_to_imputed_data(df_imputed_data, y_forecast_pred):
    """
    Append forecasted values to the imputed price data along the time axis.

    Parameters:
    - df_imputed_data (np.ndarray): Imputed price data, shape (n_samples, seq_len).
    - y_forecast_pred (np.ndarray): Forecast predictions, shape (n_samples, horizon).

    Returns:
    - combined_data (np.ndarray): Concatenated array of imputed + forecasted values,
      shape (n_samples, seq_len + horizon).
    """
    try:
        # Ensure forecast predictions are 2D (n_samples, horizon)
        if y_forecast_pred.ndim == 3 and y_forecast_pred.shape[-1] == 1:
            y_forecast_pred = np.squeeze(y_forecast_pred, axis=-1)

        if df_imputed_data.shape[0] != y_forecast_pred.shape[0]:
            raise ValueError(
                f"Mismatch in number of samples: "
                f"imputed_data has {df_imputed_data.shape[0]}, "
                f"forecast_pred has {y_forecast_pred.shape[0]}."
            )

        # Concatenate along the time axis
        df_imputed_data_and_forecast = np.concatenate([df_imputed_data, y_forecast_pred], axis=1)

        return df_imputed_data_and_forecast
    except Exception as e:
        raise RuntimeError(f"Failed to append forecast to imputed data. Details: {e}")