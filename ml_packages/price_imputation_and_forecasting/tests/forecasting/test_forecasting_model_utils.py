import unittest
import json
import os
import tempfile
from unittest.mock import MagicMock
import numpy as np
import tensorflow as tf
from price_imputation_and_forecasting.forecasting import forecasting_model_utils as fmu


class TestForecastingModelUtils(unittest.TestCase):
    """
    Extensive unit tests for forecasting_model_utils.py

    Coverage:
      - smape: correctness on perfect predictions, known numeric examples, zero-edge cases, vector mean.
      - r2_score: correctness for perfect fit, mean baseline (R^2 == 0), and simple negative/worse-than-mean examples.
      - save_forecasting_model: success writes model.h5 and history.json with expected contents;
                                invalid model type raises TypeError;
                                underlying model.save failure raises RuntimeError.
      - load_forecasting_model: success loads previously saved model and it can predict;
                                missing save directory raises FileNotFoundError;
                                present directory but missing model file raises RuntimeError (as implementation wraps load errors).
    """

    def setUp(self):
        """
        Create small tensors and a trivial Keras model used across tests.
        Using tiny arrays keeps tests fast and deterministic.
        """
        # --- Data for smape tests ---
        # Scalars / small arrays are clearer to compute expected results by hand.
        self.y_true_scalar = tf.constant([100.0], dtype=tf.float32)
        self.y_pred_scalar = tf.constant([110.0], dtype=tf.float32)

        # Perfect prediction case
        self.y_true_perfect = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.y_pred_perfect = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

        # Zero-case (both zeros)
        self.y_true_zeros = tf.constant([0.0, 0.0], dtype=tf.float32)
        self.y_pred_zeros = tf.constant([0.0, 0.0], dtype=tf.float32)

        # Mixed vector for mean smape testing
        self.y_true_vec = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        self.y_pred_vec = tf.constant([0.0, 2.0, 1.0], dtype=tf.float32)

        # --- Data for R^2 tests ---
        # Perfect fit -> R^2 = 1
        self.y_true_r2 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.y_pred_r2_perfect = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        # Baseline mean prediction -> R^2 around 0 (if predictions equal mean)
        self.y_pred_r2_mean = tf.constant([2.0, 2.0, 2.0], dtype=tf.float32)
        # Worse than mean example
        self.y_pred_r2_bad = tf.constant([10.0, -5.0, 0.0], dtype=tf.float32)

        # --- Minimal Keras model and dummy history for save/load tests ---
        # Build a tiny model to test saving & loading integration
        inp = tf.keras.Input(shape=(1,), name="inp")
        out = tf.keras.layers.Dense(1, activation="linear", name="dense")(inp)
        self.simple_model = tf.keras.Model(inputs=inp, outputs=out)

        # Dummy History-like object
        class DummyHistory:
            def __init__(self):
                self.history = {"loss": [0.5, 0.4], "val_loss": [0.45, 0.42]}

        self.dummy_history = DummyHistory()

    # ------------------------
    # Tests for SMAPE
    # ------------------------
    def test_smape_perfect_prediction_is_zero(self):
        """When predictions equal ground truth, SMAPE should be exactly 0."""
        res = fmu.smape(self.y_true_perfect, self.y_pred_perfect).numpy()
        self.assertAlmostEqual(float(res), 0.0, places=7)

    def test_smape_known_scalar_example(self):
        """
        Numerical check with a single scalar:
        SMAPE = 200 * |y - y_hat| / (|y| + |y_hat|)
        For y=100, y_hat=110 -> 200 * 10 / 210 = 2000/210 ≈ 9.5238095238
        """
        res = fmu.smape(self.y_true_scalar, self.y_pred_scalar).numpy()
        expected = 200.0 * abs(100.0 - 110.0) / (abs(100.0) + abs(110.0))
        self.assertAlmostEqual(float(res), expected, places=6)

    def test_smape_zero_zero_case(self):
        """
        When both true and predicted values are zero, numerator=0 and denominator=eps, so result should be exactly 0 (no contribution).
        """
        res = fmu.smape(self.y_true_zeros, self.y_pred_zeros).numpy()
        self.assertAlmostEqual(float(res), 0.0, places=7)

    def test_smape_vector_mean(self):
        """
        Compute SMAPE on a small vector and compare to hand-calculated mean of sample-wise SMAPEs.
        y_true = [0,1,2], y_pred = [0,2,1]
        sample SMAPEs:
         - sample0: y=0,yhat=0 -> 0
         - sample1: y=1,yhat=2 -> 200 * 1 / (1+2) = 200/3 ≈ 66.6666667
         - sample2: y=2,yhat=1 -> 200 * 1 / (2+1) = 200/3 ≈ 66.6666667
        mean = (0 + 66.6666667 + 66.6666667) / 3 ≈ 44.4444444
        """
        res = fmu.smape(self.y_true_vec, self.y_pred_vec).numpy()
        expected_s1 = 200.0 * 1.0 / (1.0 + 2.0)
        expected = (0.0 + expected_s1 + expected_s1) / 3.0
        self.assertAlmostEqual(float(res), expected, places=5)

    # ------------------------
    # Tests for R^2 score
    # ------------------------
    def test_r2_perfect_prediction_is_one(self):
        """Perfect predictions should yield R^2 == 1.0."""
        res = fmu.r2_score(self.y_true_r2, self.y_pred_r2_perfect).numpy()
        self.assertAlmostEqual(float(res), 1.0, places=7)

    def test_r2_mean_prediction_is_zero(self):
        """
        If predictions equal the mean of y_true, R^2 should be about 0. Example:
        y_true = [1,2,3], mean=2 => predictions [2,2,2] should give R^2 == 0
        """
        res = fmu.r2_score(self.y_true_r2, self.y_pred_r2_mean).numpy()
        self.assertAlmostEqual(float(res), 0.0, places=6)

    def test_r2_worse_than_mean_returns_negative(self):
        """
        If predictions are worse than predicting the mean, R^2 < 0.
        We assert that it is strictly less than zero for the constructed bad predictions.
        """
        res = fmu.r2_score(self.y_true_r2, self.y_pred_r2_bad).numpy()
        self.assertLess(float(res), 0.0)

    # ------------------------
    # Tests for save_forecasting_model and load_forecasting_model
    # ------------------------
    def test_save_forecasting_model_success_creates_files_and_history(self):
        """
        Test that save_forecasting_model writes model.h5 and history.json, and the history.json content matches the provided history.history dictionary.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "saved_model_dir")

            # Directory does not exist yet
            self.assertFalse(os.path.exists(save_dir))

            # Call save utility
            fmu.save_forecasting_model(self.simple_model, self.dummy_history, save_dir)

            model_path = os.path.join(save_dir, "model.h5")
            history_path = os.path.join(save_dir, "history.json")

            # Both files should exist after saving
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(history_path))

            # history.json should contain the same dictionary
            with open(history_path, "r", encoding="utf-8") as f:
                loaded_hist = json.load(f)
            self.assertEqual(loaded_hist, self.dummy_history.history)

    def test_load_forecasting_model_success_loads_model(self):
        """
        Integration smoke test: save a trivial model then load it via load_forecasting_model, then invoke predict to verify the loaded object behaves like a Keras model.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "saved_model_dir")

            # Save model with provided util
            fmu.save_forecasting_model(self.simple_model, self.dummy_history, save_dir)

            # Load model back
            loaded = fmu.load_forecasting_model(save_dir)

            # Basic sanity checks
            self.assertIsInstance(loaded, tf.keras.Model)

            # Should accept input shape (None, 1)
            sample = np.array([[1.23]], dtype=float)
            pred = loaded.predict(sample, verbose=0)
            # Output shape should be (N, 1)
            self.assertEqual(pred.shape[1], 1)

    def test_save_forecasting_model_invalid_model_type_raises(self):
        """Passing a non-tf.keras.Model should raise TypeError immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(TypeError):
                fmu.save_forecasting_model("not a model", self.dummy_history, tmpdir)

    def test_save_forecasting_model_save_failure_wrapped_as_runtimeerror(self):
        """
        Simulate a disk/save failure by patching model.save to raise an exception.
        The save utility should wrap this and raise RuntimeError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Replace the save method on the model instance to force an exception
            bad_model = self.simple_model
            bad_model.save = MagicMock(side_effect=Exception("simulated disk error"))

            with self.assertRaises(RuntimeError) as cm:
                fmu.save_forecasting_model(bad_model, self.dummy_history, tmpdir)
            self.assertIn("Error saving model or history", str(cm.exception))

    def test_load_forecasting_model_missing_save_dir_raises_file_not_found(self):
        """
        If the provided save directory does not exist, the loader should raise FileNotFoundError.
        """
        non_existing_dir = os.path.join(tempfile.gettempdir(), "definitely_does_not_exist_98765")
        if os.path.exists(non_existing_dir):
            # Avoid false-positive if path unexpectedly exists in the environment
            self.skipTest(f"Test path unexpectedly exists: {non_existing_dir}")

        with self.assertRaises(FileNotFoundError):
            fmu.load_forecasting_model(non_existing_dir)

    def test_load_forecasting_model_missing_model_file_raises_runtime_error(self):
        """
        If the directory exists but model.h5 is missing, implementation wraps the internal error and raises RuntimeError. We assert that behavior.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directory exists but model.h5 not present
            # Confirm model.h5 absent
            model_path = os.path.join(tmpdir, "model.h5")
            if os.path.exists(model_path):
                os.remove(model_path)

            # Based on implementation, missing model file will be raised inside try and caught/re-wrapped -> RuntimeError
            with self.assertRaises(RuntimeError):
                fmu.load_forecasting_model(tmpdir)


if __name__ == "__main__":
    unittest.main()