import unittest
from unittest.mock import MagicMock
import numpy as np
from price_imputation_and_forecasting.imputation import predict as predict_mod


class TestPredictModule(unittest.TestCase):
    """
    Unit tests for imputation.predict module.

    Covered functions:
      - run_model_prediction: successful call, correct arguments passed to model.predict, and wrapping of exceptions into RuntimeError.
      - replace_missing_with_predictions: correct replacement for 3D and 2D predictions, no-op when no missing values, does not mutate input array,
                                          and raises RuntimeError (wrapped) when shapes mismatch or unexpected errors occur.
    """

    def setUp(self):
        # Sequence / static inputs used for run_model_prediction tests
        # Use small arrays to keep tests fast and deterministic
        self.X_sequence = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]
        ])  # shape (2, 3, 2)

        self.X_static = np.array([
            [0, 1, 0],
            [1, 0, 1]
        ])  # shape (2, 3)

        # Data for replace_missing_with_predictions tests
        # df_price_input uses 'fill_nan_value' placeholder 0.0 for missing entries
        self.df_price_input = np.array([
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 4.0]
        ], dtype=float)  # shape (2, 3)

        # Predictions shaped (n_samples, seq_len, 1) (common Keras output)
        self.y_pred_3d = np.array([
            [[9.0], [8.0], [7.0]],
            [[6.0], [5.0], [4.0]]
        ], dtype=float)

        # Predictions already squeezed to 2D
        self.y_pred_2d = np.squeeze(self.y_pred_3d, axis=-1)

    # ---------------------------
    # Tests for run_model_prediction
    # ---------------------------
    def test_run_model_prediction_calls_predict_and_returns_array(self):
        """Model.predict should be called with [X_sequence, X_static] and returned array returned as-is."""
        model = MagicMock()
        expected_out = np.array([[[0.1], [0.2], [0.3]], [[1.0], [1.1], [1.2]]])
        model.predict = MagicMock(return_value=expected_out)

        # Call the wrapper
        result = predict_mod.run_model_prediction(model, self.X_sequence, self.X_static)

        # Model.predict called once with correct arguments
        model.predict.assert_called_once()
        called_args, called_kwargs = model.predict.call_args
        # First positional arg should be the list [X_sequence, X_static]
        self.assertEqual(len(called_args), 1)
        self.assertIsInstance(called_args[0], list)
        # Confirm the exact arrays were passed
        np.testing.assert_array_equal(called_args[0][0], self.X_sequence)
        np.testing.assert_array_equal(called_args[0][1], self.X_static)
        # Ensure verbose kwarg passed as expected
        self.assertIn("verbose", called_kwargs)
        self.assertEqual(called_kwargs["verbose"], 0)

        # Result is the same array returned by model.predict
        np.testing.assert_array_equal(result, expected_out)

    def test_run_model_prediction_wraps_predict_errors(self):
        """If model.predict raises, run_model_prediction should raise RuntimeError with informative message."""
        model = MagicMock()
        model.predict = MagicMock(side_effect=Exception("model failure"))

        with self.assertRaises(RuntimeError) as cm:
            predict_mod.run_model_prediction(model, self.X_sequence, self.X_static)

        self.assertIn("[run_model_prediction] Error during prediction", str(cm.exception))

    # ---------------------------
    # Tests for replace_missing_with_predictions
    # ---------------------------
    def test_replace_missing_with_predictions_3d_predictions(self):
        """
        When predictions are shape (n, seq_len, 1), they should be squeezed and used to replace elements equal to fill_nan_value in the input array.
        """
        df_in = self.df_price_input.copy()
        fill_nan_value = 0.0

        result = predict_mod.replace_missing_with_predictions(df_in, self.y_pred_3d, fill_nan_value)

        # Original not mutated
        np.testing.assert_array_equal(df_in, self.df_price_input)

        # Expected: replace zeros with predictions (squeezed)
        expected = self.df_price_input.copy()
        expected[0, 0] = self.y_pred_3d[0, 0, 0]  # 9.0
        expected[0, 2] = self.y_pred_3d[0, 2, 0]  # 7.0
        expected[1, 1] = self.y_pred_3d[1, 1, 0]  # 5.0

        np.testing.assert_array_equal(result, expected)

    def test_replace_missing_with_predictions_2d_predictions(self):
        """
        When predictions are already 2D (n, seq_len), function should use them directly.
        """
        df_in = self.df_price_input.copy()
        fill_nan_value = 0.0

        result = predict_mod.replace_missing_with_predictions(df_in, self.y_pred_2d, fill_nan_value)

        expected = self.df_price_input.copy()
        expected[0, 0] = self.y_pred_2d[0, 0]
        expected[0, 2] = self.y_pred_2d[0, 2]
        expected[1, 1] = self.y_pred_2d[1, 1]

        np.testing.assert_array_equal(result, expected)

    def test_replace_missing_with_predictions_mismatched_shapes_wrapped_as_runtimeerror(self):
        """
        If shapes mismatch between df_price_input and y_pred so that indexing fails, the function should catch and re-raise as RuntimeError.
        Example mismatch: predictions have different number of samples than the input.
        """
        df_in = self.df_price_input.copy()
        # Create a wrongly-shaped y_pred (1 sample instead of 2)
        bad_pred = np.array([[[1.0], [2.0], [3.0]]], dtype=float)  # shape (1,3,1)

        with self.assertRaises(RuntimeError) as cm:
            predict_mod.replace_missing_with_predictions(df_in, bad_pred, fill_nan_value=0.0)

        self.assertIn("Failed to replace missing values with predictions", str(cm.exception))

    def test_replace_missing_with_predictions_unexpected_pred_rank_raises(self):
        """
        If y_pred has rank>3 or last-dim != 1 (and assignment fails), function should raise RuntimeError.
        """
        df_in = self.df_price_input.copy()
        # Predictions with last-dim != 1 (e.g., shape (2,3,2)) will cause assignment mismatch
        bad_pred = np.ones((2, 3, 2), dtype=float)

        with self.assertRaises(RuntimeError) as cm:
            predict_mod.replace_missing_with_predictions(df_in, bad_pred, fill_nan_value=0.0)

        self.assertIn("Failed to replace missing values with predictions", str(cm.exception))

    def test_replace_missing_with_predictions_non_array_pred_raises_runtimeerror(self):
        """
        If y_pred is not an array-like with .ndim attribute in the expected way, function should raise RuntimeError. (Covers unexpected types.)
        """
        df_in = self.df_price_input.copy()
        non_array_pred = "not an array"

        with self.assertRaises(RuntimeError):
            predict_mod.replace_missing_with_predictions(df_in, non_array_pred, fill_nan_value=0.0)


if __name__ == "__main__":
    unittest.main()