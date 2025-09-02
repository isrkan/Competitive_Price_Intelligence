import unittest
from unittest.mock import MagicMock
import numpy as np
from price_imputation_and_forecasting.forecasting import predict as predict_mod


class TestForecastingPredictModule(unittest.TestCase):
    """
    Unit tests for forecasting.predict module.

    Functions tested:
      - run_forecasting_prediction
      - append_forecast_to_imputed_data

    Tests include:
      - successful prediction call and argument checking
      - prediction exceptions wrapped into RuntimeError
      - correct handling of 3D (n, horizon, 1) and 2D (n, horizon) forecasts
      - non-mutating behavior (input arrays are not changed)
      - shape mismatch and bad-rank cases producing RuntimeError (wrapped)
      - zero-horizon forecasting (empty forecast) handled correctly
    """

    def setUp(self):
        # Example imputed data for 2 series, 5 timesteps
        self.df_imputed = np.array([
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [2.0, 2.1, 2.2, 2.3, 2.4]
        ], dtype=float)  # shape (2, 5)

        # Forecast predictions as 3D Keras-style output (n_samples, horizon, 1)
        self.y_forecast_3d = np.array([
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]]
        ], dtype=float)  # shape (2, 3, 1)

        # Forecast predictions as 2D already-squeezed (n_samples, horizon)
        self.y_forecast_2d = np.squeeze(self.y_forecast_3d, axis=-1)  # shape (2, 3)

        # Forecast predictions with mismatched sample count (bad)
        self.y_forecast_bad_rows = np.array([
            [[9.0], [9.1], [9.2]],
        ], dtype=float)  # shape (1,3,1)

        # Forecast predictions with last-dimension != 1 (bad rank for concatenation)
        self.y_forecast_bad_rank = np.ones((2, 3, 2), dtype=float)  # shape (2,3,2)

        # Forecast predictions with zero horizon (empty second axis)
        self.y_forecast_zero_horizon = np.empty((2, 0), dtype=float)  # shape (2,0)

        # Example X_sequence for run_forecasting_prediction
        # small shape (2, lookback, features)
        self.X_sequence = np.array([
            [[0.1], [0.2], [0.3]],
            [[1.1], [1.2], [1.3]]
        ], dtype=float)

    # ---------------------------
    # run_forecasting_prediction
    # ---------------------------
    def test_run_forecasting_prediction_calls_model_predict_and_returns_value(self):
        """
        Verify run_forecasting_prediction calls model.predict(X_sequence, verbose=0), and returns exactly the array returned by the model.
        """
        model = MagicMock()
        expected = np.array([[[0.5], [0.6]], [[0.7], [0.8]]])
        # model.predict should accept X_sequence and verbose kwarg
        model.predict = MagicMock(return_value=expected)

        result = predict_mod.run_forecasting_prediction(model, self.X_sequence)

        # Ensure predict called once with the X_sequence and verbose=0
        model.predict.assert_called_once()
        called_args, called_kwargs = model.predict.call_args
        # first positional arg equals X_sequence
        np.testing.assert_array_equal(called_args[0], self.X_sequence)
        # verbose keyword must be present and 0
        self.assertIn("verbose", called_kwargs)
        self.assertEqual(called_kwargs["verbose"], 0)

        # result should be exactly model's output
        np.testing.assert_array_equal(result, expected)

    def test_run_forecasting_prediction_wraps_exception(self):
        """
        If underlying model.predict throws, run_forecasting_prediction should raise RuntimeError.
        """
        model = MagicMock()
        model.predict = MagicMock(side_effect=Exception("predict failed"))

        with self.assertRaises(RuntimeError) as cm:
            predict_mod.run_forecasting_prediction(model, self.X_sequence)

        self.assertIn("[run_forecasting_prediction] Error during prediction", str(cm.exception))

    # ---------------------------
    # append_forecast_to_imputed_data
    # ---------------------------
    def test_append_forecast_to_imputed_data_handles_3d_input(self):
        """
        When forecast is 3D with last-dim 1, function should squeeze and concatenate along time axis.
        """
        imputed_copy = self.df_imputed.copy()
        result = predict_mod.append_forecast_to_imputed_data(imputed_copy, self.y_forecast_3d)

        # Expect shape (2, 5+3)
        self.assertEqual(result.shape, (2, 8))
        # Check appended values are equal to squeezed forecast
        np.testing.assert_array_equal(result[:, 5:], self.y_forecast_2d)
        # Original input not mutated
        np.testing.assert_array_equal(imputed_copy, self.df_imputed)

    def test_append_forecast_to_imputed_data_handles_2d_input(self):
        """
        When forecast is already 2D (n, horizon), function should concatenate directly.
        """
        imputed_copy = self.df_imputed.copy()
        result = predict_mod.append_forecast_to_imputed_data(imputed_copy, self.y_forecast_2d)

        self.assertEqual(result.shape, (2, 8))
        np.testing.assert_array_equal(result[:, 5:], self.y_forecast_2d)
        np.testing.assert_array_equal(imputed_copy, self.df_imputed)  # ensure not mutated

    def test_append_forecast_to_imputed_data_zero_horizon_returns_original_copy(self):
        """
        If horizon is zero (forecast empty), concatenation should return a copy equal to the input.
        """
        imputed_copy = self.df_imputed.copy()
        result = predict_mod.append_forecast_to_imputed_data(imputed_copy, self.y_forecast_zero_horizon)

        # shape should be unchanged
        np.testing.assert_array_equal(result, self.df_imputed)
        # But result should be a new array (not the same object)
        self.assertIsNot(result, self.df_imputed)

    def test_append_forecast_to_imputed_data_mismatched_rows_raises_runtimeerror(self):
        """
        If number of series differ between imputed data and forecasts, function should raise RuntimeError (wrapped).
        """
        with self.assertRaises(RuntimeError) as cm:
            predict_mod.append_forecast_to_imputed_data(self.df_imputed, self.y_forecast_bad_rows)

        msg = str(cm.exception)
        # Ensure wrapper message is present and contains mismatch info (original ValueError details included)
        self.assertIn("Failed to append forecast to imputed data", msg)
        self.assertTrue(("Mismatch in number of samples" in msg) or ("imputed_data has" in msg))

    def test_append_forecast_to_imputed_data_bad_rank_raises_runtimeerror(self):
        """
        If forecast array has an unexpected trailing dimension (e.g. last-dim != 1), concatenation will fail and the function should raise RuntimeError.
        """
        with self.assertRaises(RuntimeError) as cm:
            predict_mod.append_forecast_to_imputed_data(self.df_imputed, self.y_forecast_bad_rank)

        self.assertIn("Failed to append forecast to imputed data", str(cm.exception))

    def test_append_forecast_to_imputed_data_preserves_dtype_and_values(self):
        """
        Confirm numeric type preserved and a few sample values retained after concatenation.
        """
        result = predict_mod.append_forecast_to_imputed_data(self.df_imputed, self.y_forecast_2d)
        # dtype should still be float (as inputs were float)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        # Check first row original values preserved in result start
        np.testing.assert_array_almost_equal(result[0, :5], self.df_imputed[0, :5])
        # Check appended tail equals forecast
        np.testing.assert_array_equal(result[0, 5:], self.y_forecast_2d[0])

    def test_append_forecast_to_imputed_data_nan_values_handled(self):
        """
        If forecasts contain NaN values, they should be appended as-is (no special handling).
        """
        y_with_nan = self.y_forecast_2d.copy()
        y_with_nan[1, 1] = np.nan
        result = predict_mod.append_forecast_to_imputed_data(self.df_imputed, y_with_nan)
        # NaN should be present in the concatenated result at expected location
        self.assertTrue(np.isnan(result[1, 6]))  # second series, second forecast column


if __name__ == "__main__":
    unittest.main()