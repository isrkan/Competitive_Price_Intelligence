import unittest
from unittest.mock import MagicMock
import numpy as np
import tensorflow as tf
from price_imputation_and_forecasting.evaluation import forecasting_evaluation as feval
from price_imputation_and_forecasting.forecasting import forecasting_model_utils as fmu
import numbers


class TestEvaluateForecastingModel(unittest.TestCase):
    """
    Extensive unit tests for evaluate_forecasting_model function.

    Coverage:
      - Correct numeric computation of MSE, MAE, sMAPE, R^2.
      - Handles 2D and 3D input shapes for y_test and y_pred.
      - Handles single-sample, multi-step forecasts.
      - Ensures proper squeezing of last axis when num_features=1.
      - Exception handling for model.predict or metrics failures.
      - Works with small arrays for deterministic tests.
    """

    def setUp(self):
        # Simple example with 2 samples, 3-step forecast, single feature
        self.X_test = np.array([
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ], dtype=float)  # shape (2,3,1)

        self.y_test = np.array([
            [[10.0], [20.0], [30.0]],
            [[40.0], [50.0], [60.0]]
        ], dtype=float)  # shape (2,3,1)

        # Mock predictions, same shape
        self.y_pred = np.array([
            [[11.0], [19.0], [31.0]],
            [[39.0], [51.0], [59.0]]
        ], dtype=float)

        # Mock model
        self.model = MagicMock()
        self.model.predict = MagicMock(return_value=self.y_pred)

    # ---------------------------
    # Basic functionality
    # ---------------------------
    def test_basic_evaluation_returns_all_metrics(self):
        """
        Standard test: ensure all expected keys exist and numeric.
        """
        results = feval.evaluate_forecasting_model(self.model, self.X_test, self.y_test)

        self.model.predict.assert_called_once_with(self.X_test, verbose=0)

        expected_keys = {"mse", "mae", "smape", "r2"}
        self.assertEqual(set(results.keys()), expected_keys)
        for val in results.values():
            self.assertIsInstance(val, numbers.Real)

    def test_evaluation_with_2d_y_test_and_pred(self):
        """
        Should work if y_test and y_pred are already 2D (squeezed).
        """
        y_test_2d = np.squeeze(self.y_test, axis=-1)
        y_pred_2d = np.squeeze(self.y_pred, axis=-1)
        self.model.predict = MagicMock(return_value=y_pred_2d)

        results = feval.evaluate_forecasting_model(self.model, self.X_test, y_test_2d)
        self.assertTrue(all(isinstance(val, numbers.Real) for val in results.values()))

    # ---------------------------
    # Numeric correctness
    # ---------------------------
    def test_mse_mae_manual_check(self):
        """
        Verify that mse and mae values match manual computation for small arrays.
        """
        results = feval.evaluate_forecasting_model(self.model, self.X_test, self.y_test)

        # Manual computations
        y_true_flat = self.y_test.squeeze()
        y_pred_flat = self.y_pred.squeeze()

        mse_manual = np.mean((y_true_flat - y_pred_flat) ** 2)
        mae_manual = np.mean(np.abs(y_true_flat - y_pred_flat))

        self.assertAlmostEqual(results["mse"], mse_manual, places=5)
        self.assertAlmostEqual(results["mae"], mae_manual, places=5)

    def test_smape_r2_manual_check(self):
        """
        Verify that sMAPE and R^2 values are correctly computed.
        """
        results = feval.evaluate_forecasting_model(self.model, self.X_test, self.y_test)

        y_true_tf = tf.constant(self.y_test.squeeze(), dtype=tf.float32)
        y_pred_tf = tf.constant(self.y_pred.squeeze(), dtype=tf.float32)

        smape_manual = fmu.smape(y_true_tf, y_pred_tf).numpy()
        r2_manual = fmu.r2_score(y_true_tf, y_pred_tf).numpy()

        self.assertAlmostEqual(results["smape"], smape_manual, places=5)
        self.assertAlmostEqual(results["r2"], r2_manual, places=5)

    # ---------------------------
    # Edge cases
    # ---------------------------
    def test_single_sample_forecast(self):
        """
        Ensure function works for single sample, multi-step forecast.
        """
        X_single = self.X_test[:1]
        y_single = self.y_test[:1]
        y_pred_single = self.y_pred[:1]
        self.model.predict = MagicMock(return_value=y_pred_single)

        results = feval.evaluate_forecasting_model(self.model, X_single, y_single)
        self.assertTrue(all(isinstance(val, numbers.Real) for val in results.values()))

    def test_large_values_forecast(self):
        """
        Ensure large numeric values do not cause overflow.
        """
        y_test_large = self.y_test * 1e6
        y_pred_large = self.y_pred * 1e6
        self.model.predict = MagicMock(return_value=y_pred_large)

        results = feval.evaluate_forecasting_model(self.model, self.X_test, y_test_large)
        self.assertTrue(all(isinstance(val, numbers.Real) for val in results.values()))
        self.assertGreater(results["mse"], 0)
        self.assertGreater(results["mae"], 0)

    def test_zero_difference_forecast_perfect_prediction(self):
        """
        If predictions exactly match y_test, mse, mae = 0, sMAPE = 0, r2 = 1.
        """
        self.model.predict = MagicMock(return_value=self.y_test)
        results = feval.evaluate_forecasting_model(self.model, self.X_test, self.y_test)

        self.assertAlmostEqual(results["mse"], 0.0, places=7)
        self.assertAlmostEqual(results["mae"], 0.0, places=7)
        self.assertAlmostEqual(results["smape"], 0.0, places=7)
        self.assertAlmostEqual(results["r2"], 1.0, places=7)

    # ---------------------------
    # Exception handling
    # ---------------------------
    def test_model_predict_raises_runtimeerror(self):
        """
        If model.predict fails, it should be wrapped in RuntimeError.
        """
        bad_model = MagicMock()
        bad_model.predict = MagicMock(side_effect=Exception("predict fail"))

        with self.assertRaises(RuntimeError) as cm:
            feval.evaluate_forecasting_model(bad_model, self.X_test, self.y_test)
        self.assertIn("[evaluate_forecasting_model] Error during evaluation", str(cm.exception))
        self.assertIn("predict fail", str(cm.exception))

    def test_metric_computation_raises_runtimeerror(self):
        """
        If smape or r2_score fail, should wrap as RuntimeError.
        """
        original_smape = fmu.smape
        fmu.smape = MagicMock(side_effect=Exception("smape fail"))
        try:
            with self.assertRaises(RuntimeError) as cm:
                feval.evaluate_forecasting_model(self.model, self.X_test, self.y_test)
            self.assertIn("smape fail", str(cm.exception))
            self.assertIn("[evaluate_forecasting_model] Error during evaluation", str(cm.exception))
        finally:
            fmu.smape = original_smape


if __name__ == "__main__":
    unittest.main()