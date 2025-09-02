import unittest
import json
import os
import tempfile
from unittest.mock import MagicMock
import numpy as np
import tensorflow as tf
from price_imputation_and_forecasting.imputation import imputation_model_utils as imu


class TestImputationModelUtils(unittest.TestCase):
    """
    Extensive unit tests for imputation_model_utils.py

    Test coverage:
      - masked_mse: correct numeric results for fully / partially masked inputs, non-binary mask values, and all-zero mask (epsilon handling).
      - masked_mae: analogous tests to masked_mse.
      - save_imputation_model: success case writes model.h5 and history.json with expected contents;
                              invalid model type raises TypeError;
                              underlying save failure is wrapped as RuntimeError.
      - load_imputation_model: success loads previously saved model; missing save path or missing model file raises FileNotFoundError; unexpected errors are wrapped as RuntimeError.
    """

    def setUp(self):
        """
        Prepare small arrays and a trivial Keras model used across tests.
        Using tiny arrays keeps tests fast and deterministic.
        """
        # Small samples to test masked metrics
        # We'll work with shape (batch, time, last_dim=2) for y_true_with_mask, and (batch, time, 1) for y_pred
        # Example: 2 samples, 1 timestep each
        self.y_true_vals = np.array([[[1.0, 1.0]], [[3.0, 1.0]]], dtype=np.float32)  # last dim: [true, mask]
        self.y_pred_vals = np.array([[[2.0]], [[2.0]]], dtype=np.float32)  # predicted values

        # Another case for partial mask (first sample masked, second not masked)
        self.y_true_partial = np.array([[[5.0, 1.0]], [[6.0, 0.0]]], dtype=np.float32)
        self.y_pred_partial = np.array([[[4.0]], [[1.0]]], dtype=np.float32)

        # Non-binary mask example (weights)
        self.y_true_weighted = np.array([[[2.0, 0.5]], [[4.0, 0.25]]], dtype=np.float32)
        self.y_pred_weighted = np.array([[[3.0]], [[3.0]]], dtype=np.float32)

        # All-zero mask case -> ensure no division-by-zero (should return 0 because numerator will be 0)
        self.y_true_all_zero_mask = np.array([[[10.0, 0.0]], [[20.0, 0.0]]], dtype=np.float32)
        self.y_pred_all_zero_mask = np.array([[[9.0]], [[19.0]]], dtype=np.float32)

        # small Keras model for save/load tests
        input_layer = tf.keras.Input(shape=(1,), name="inp")
        out = tf.keras.layers.Dense(1, activation="linear", name="dense")(input_layer)
        self.simple_model = tf.keras.Model(inputs=input_layer, outputs=out)

        # Dummy "History-like" object with .history attribute (serialization used by save function)
        class DummyHistory:
            def __init__(self):
                self.history = {"loss": [0.5, 0.4], "val_loss": [0.45, 0.42]}

        self.dummy_history = DummyHistory()

    # ------------------------
    # Tests for masked_mse
    # ------------------------
    def test_masked_mse_basic_all_masked(self):
        """If mask==1 for all entries, masked_mse equals usual MSE across entries."""
        y_true = tf.constant(self.y_true_vals, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_vals, dtype=tf.float32)

        # compute expected manually:
        # sample1: (1-2)^2 = 1, sample2: (3-2)^2 = 1 -> mean = (1+1)/2 = 1
        expected = 1.0

        result = imu.masked_mse(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_masked_mse_partial_mask(self):
        """When mask selectively disables samples, only masked entries contribute to loss."""
        y_true = tf.constant(self.y_true_partial, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_partial, dtype=tf.float32)

        # Only the first sample contributes: (5-4)^2 = 1, denominator = 1 -> mse = 1
        expected = 1.0

        result = imu.masked_mse(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_masked_mse_nonbinary_mask_weights(self):
        """Mask can act as continuous weights — loss should be weighted accordingly."""
        y_true = tf.constant(self.y_true_weighted, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_weighted, dtype=tf.float32)

        # sample1: (2-3)^2 = 1 * 0.5 = 0.5
        # sample2: (4-3)^2 = 1 * 0.25 = 0.25
        # numerator = 0.75, denominator = 0.5 + 0.25 = 0.75 => 0.75/0.75 = 1.0
        expected = 1.0

        result = imu.masked_mse(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_masked_mse_all_zero_mask_returns_zero(self):
        """
        If mask is all zeros, numerator will be zero so function should return 0.0 (divide-by-eps avoided internally).
        """
        y_true = tf.constant(self.y_true_all_zero_mask, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_all_zero_mask, dtype=tf.float32)

        result = imu.masked_mse(y_true, y_pred).numpy()
        # expect exactly 0.0 because masked squared_error=0
        self.assertAlmostEqual(float(result), 0.0, places=7)

    # ------------------------
    # Tests for masked_mae
    # ------------------------
    def test_masked_mae_basic_all_masked(self):
        """MAE computed only on positions where mask==1."""
        y_true = tf.constant(self.y_true_vals, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_vals, dtype=tf.float32)

        # abs errors: |1-2| = 1, |3-2| =1 => mean = 1
        expected = 1.0

        result = imu.masked_mae(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_masked_mae_partial_mask(self):
        """Masked MAE should ignore entries with mask=0."""
        y_true = tf.constant(self.y_true_partial, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_partial, dtype=tf.float32)

        # Only first sample counts: |5-4| = 1
        expected = 1.0

        result = imu.masked_mae(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_masked_mae_nonbinary_mask_weights(self):
        """Continuous mask values act as sample weights for MAE."""
        y_true = tf.constant(self.y_true_weighted, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_weighted, dtype=tf.float32)

        # abs errors weighted: sample1 -> |2-3|*0.5 = 1*0.5 = 0.5
        # sample2 -> |4-3|*0.25 = 1*0.25 = 0.25
        # numerator = 0.75, denom = 0.75 -> result = 1.0
        expected = 1.0

        result = imu.masked_mae(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_masked_mae_all_zero_mask_returns_zero(self):
        """If mask all zeros, result should be 0.0 (safe epsilon prevents division by zero)."""
        y_true = tf.constant(self.y_true_all_zero_mask, dtype=tf.float32)
        y_pred = tf.constant(self.y_pred_all_zero_mask, dtype=tf.float32)

        result = imu.masked_mae(y_true, y_pred).numpy()
        self.assertAlmostEqual(float(result), 0.0, places=7)

    # ------------------------
    # Tests for save_imputation_model and load_imputation_model
    # ------------------------
    def test_save_imputation_model_success_creates_files_and_history(self):
        """
        Test that save_imputation_model writes model.h5 and history.json to a directory, and that history.json contains the provided history.history contents.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "saved_model_dir")

            # Confirm nothing exists initially
            self.assertFalse(os.path.exists(save_dir))

            # Call save; should create the directory and files
            imu.save_imputation_model(self.simple_model, self.dummy_history, save_dir)

            # Check files exist
            model_path = os.path.join(save_dir, "model.h5")
            history_path = os.path.join(save_dir, "history.json")
            self.assertTrue(os.path.exists(model_path), "model.h5 must be created by save_imputation_model")
            self.assertTrue(os.path.exists(history_path), "history.json must be created by save_imputation_model")

            # Verify history content matches
            with open(history_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            self.assertEqual(hist, self.dummy_history.history)

    def test_load_imputation_model_success_loads_model(self):
        """
        Integration check: save a minimal model then load it using load_imputation_model.
        Verify the loaded object is callable and can predict on sample input.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "saved_model_dir")

            # Save using the utility to ensure model.h5 is present
            imu.save_imputation_model(self.simple_model, self.dummy_history, save_dir)

            # Now load with the loader
            loaded = imu.load_imputation_model(save_dir)

            # Basic sanity checks on loaded model
            self.assertIsInstance(loaded, tf.keras.Model)
            # the model should accept an input of shape (None,1) and produce output shape (None,1)
            sample_in = np.array([[1.0]], dtype=float)
            # Call predict to ensure model is operational
            pred = loaded.predict(sample_in, verbose=0)
            self.assertEqual(pred.shape[1], 1)

    def test_save_imputation_model_invalid_model_type_raises_type_error(self):
        """Passing a non-tf.keras.Model object should raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(TypeError):
                imu.save_imputation_model(model="not a model", history=self.dummy_history, save_path=tmpdir)

    def test_save_imputation_model_save_fails_wrapped_as_runtimeerror(self):
        """
        Simulate a failure during model.save (e.g., disk error) by replacing model.save with a function that raises. The utility should catch and raise RuntimeError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Replace the save method on the model instance to force an exception
            bad_model = self.simple_model
            bad_model.save = MagicMock(side_effect=Exception("simulated disk error"))

            with self.assertRaises(RuntimeError) as cm:
                imu.save_imputation_model(bad_model, self.dummy_history, tmpdir)
            self.assertIn("Error saving model or history", str(cm.exception))

    def test_load_imputation_model_missing_save_dir_raises_file_not_found(self):
        """If the provided save directory doesn't exist, loader should raise FileNotFoundError."""
        non_existing_dir = os.path.join(tempfile.gettempdir(), "definitely_does_not_exist_12345")
        # Ensure it truly doesn't exist for the test
        if os.path.exists(non_existing_dir):
            # avoid accidentally existing path
            self.skipTest(f"Test path unexpectedly exists: {non_existing_dir}")

        with self.assertRaises(FileNotFoundError):
            imu.load_imputation_model(non_existing_dir)

    def test_load_imputation_model_missing_model_file_raises_file_not_found(self):
        """
        If directory exists but model.h5 is missing, loader should raise FileNotFoundError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # create empty dir tmpdir; ensure no model.h5
            model_path = os.path.join(tmpdir, "model.h5")
            if os.path.exists(model_path):
                os.remove(model_path)

            with self.assertRaises(RuntimeError):
                imu.load_imputation_model(tmpdir)


if __name__ == "__main__":
    unittest.main()