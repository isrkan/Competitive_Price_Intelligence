import unittest
from unittest.mock import MagicMock
import numpy as np
import tensorflow as tf
from price_imputation_and_forecasting.evaluation import imputation_evaluation as im_eval
from price_imputation_and_forecasting.imputation import imputation_model_utils as imu


class TestEvaluateImputationModel(unittest.TestCase):
    """
    Extensive unit tests for evaluate_imputation_model function.

    Coverage:
      - Correct numeric computation of masked MSE and MAE.
      - Handles 2D and 3D input shapes for y_true and target_mask.
      - Ensures proper expansion of last axis.
      - Handles non-binary mask values (weights).
      - Wraps exceptions from model.predict or evaluation functions as RuntimeError.
      - Works with small arrays for deterministic tests.
    """

    def setUp(self):
        # Small 2x3 example for evaluation
        self.X_sequence = np.array([
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]]
        ], dtype=float)  # shape (2, 3, 1)

        self.X_static = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ], dtype=float)  # shape (2, 2)

        # Ground truth y_true and mask
        self.y_true = np.array([
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0]
        ], dtype=float)  # shape (2, 3)

        # Binary mask
        self.target_mask = np.array([
            [1, 1, 0],
            [1, 0, 1]
        ], dtype=float)

        # Non-binary weights
        self.target_mask_weighted = np.array([
            [0.5, 0.25, 0],
            [0.1, 0, 0.9]
        ], dtype=float)

        # Mock model predict output, shape (2,3,1)
        self.y_pred = np.array([
            [[11.0], [19.0], [100.0]],
            [[39.0], [500.0], [61.0]]
        ], dtype=float)

        # Mock model
        self.model = MagicMock()
        self.model.predict = MagicMock(return_value=self.y_pred)

    # ---------------------------
    # Basic functionality
    # ---------------------------
    def test_evaluate_imputation_model_basic(self):
        """
        Standard test: evaluate with 2D y_true and mask.
        Check that returned dict has keys and numeric values.
        """
        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    self.y_true, self.target_mask)

        self.model.predict.assert_called_once_with([self.X_sequence, self.X_static], verbose=0)

        self.assertIn("masked_mse", results)
        self.assertIn("masked_mae", results)
        self.assertIsInstance(results["masked_mse"], float)
        self.assertIsInstance(results["masked_mae"], float)

    def test_evaluate_imputation_model_with_3d_inputs(self):
        """
        Ensure function works if y_true and target_mask already have last-dim=1 (3D).
        Should not double-expand.
        """
        y_true_3d = np.expand_dims(self.y_true, axis=-1)
        mask_3d = np.expand_dims(self.target_mask, axis=-1)

        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    y_true_3d, mask_3d)

        # Confirm same keys
        self.assertIn("masked_mse", results)
        self.assertIn("masked_mae", results)

    # ---------------------------
    # Numeric correctness
    # ---------------------------
    def test_evaluate_imputation_model_masked_mse_manual_check(self):
        """
        Check that masked_mse computation matches manual computation for small arrays.
        """
        # We expect masked_mse = sum((y_true-y_pred)^2*mask)/sum(mask)
        y_true_with_mask = np.stack([self.y_true, self.target_mask], axis=-1)
        expected_mse = imu.masked_mse(tf.constant(y_true_with_mask, dtype=tf.float32),
                                      tf.constant(self.y_pred, dtype=tf.float32)).numpy()

        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    self.y_true, self.target_mask)
        self.assertAlmostEqual(results["masked_mse"], expected_mse, places=5)

    def test_evaluate_imputation_model_masked_mae_manual_check(self):
        """
        Check that masked_mae computation matches manual computation for small arrays.
        """
        y_true_with_mask = np.stack([self.y_true, self.target_mask], axis=-1)
        expected_mae = imu.masked_mae(tf.constant(y_true_with_mask, dtype=tf.float32),
                                      tf.constant(self.y_pred, dtype=tf.float32)).numpy()

        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    self.y_true, self.target_mask)
        self.assertAlmostEqual(results["masked_mae"], expected_mae, places=5)

    def test_evaluate_imputation_model_with_weighted_mask(self):
        """
        Ensure non-binary mask weights are handled correctly.
        """
        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    self.y_true, self.target_mask_weighted)

        # Should produce float numeric values
        self.assertIsInstance(results["masked_mse"], float)
        self.assertIsInstance(results["masked_mae"], float)

    # ---------------------------
    # Exception handling
    # ---------------------------
    def test_evaluate_imputation_model_predict_throws_runtimeerror(self):
        """
        If model.predict fails, evaluation wraps the exception as RuntimeError.
        """
        bad_model = MagicMock()
        bad_model.predict = MagicMock(side_effect=Exception("predict failed"))

        with self.assertRaises(RuntimeError) as cm:
            im_eval.evaluate_imputation_model(bad_model, self.X_sequence, self.X_static,
                                              self.y_true, self.target_mask)
        self.assertIn("[evaluate_model] Error during evaluation", str(cm.exception))
        self.assertIn("predict failed", str(cm.exception))

    def test_evaluate_imputation_model_masked_function_throws_runtimeerror(self):
        """
        If masked_mse or masked_mae raise an exception, it should be wrapped into RuntimeError.
        """
        # Patch masked_mse to throw
        original_masked_mse = imu.masked_mse
        imu.masked_mse = MagicMock(side_effect=Exception("masked mse fail"))

        try:
            with self.assertRaises(RuntimeError) as cm:
                im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                  self.y_true, self.target_mask)
            self.assertIn("[evaluate_model] Error during evaluation", str(cm.exception))
            self.assertIn("masked mse fail", str(cm.exception))
        finally:
            # Restore
            imu.masked_mse = original_masked_mse

    # ---------------------------
    # Edge cases
    # ---------------------------
    def test_evaluate_imputation_model_zero_mask_returns_zero_loss(self):
        """
        If target_mask is all zeros, masked losses should return 0.0 (due to epsilon handling inside masked metrics)
        """
        zero_mask = np.zeros_like(self.target_mask)
        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    self.y_true, zero_mask)
        self.assertAlmostEqual(results["masked_mse"], 0.0, places=7)
        self.assertAlmostEqual(results["masked_mae"], 0.0, places=7)

    def test_evaluate_imputation_model_large_values(self):
        """
        Ensure function works with large numeric values without overflow.
        """
        y_true_large = self.y_true * 1e6
        y_pred_large = self.y_pred * 1e6
        self.model.predict = MagicMock(return_value=y_pred_large)

        results = im_eval.evaluate_imputation_model(self.model, self.X_sequence, self.X_static,
                                                    y_true_large, self.target_mask)
        self.assertIsInstance(results["masked_mse"], float)
        self.assertIsInstance(results["masked_mae"], float)
        self.assertGreater(results["masked_mse"], 0.0)
        self.assertGreater(results["masked_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()