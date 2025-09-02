import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from price_imputation_and_forecasting.imputation import train


class TestTrainImputationModelDispatcher(unittest.TestCase):
    """
    Tests for train.train_imputation_model dispatcher.

    The dispatcher validates `model_name` and then forwards the inputs + hyperparameters to the model-specific trainer (currently train_bi_rnn). We do NOT unit-test the
    actual DL architecture (bi_rnn.py) here — that is intentionally out-of-scope and should be covered by integration tests.

    Tests include:
    - Validation of `model_name` type and allowed values.
    - Correct forwarding of arguments (X_*, y_*, masks, and imputation_model_params).
    - Propagation of exceptions from the called trainer.
    - Behavior when imputation_model_params is malformed (None or conflicting keys).
    """

    def setUp(self):
        """
        Create small dummy arrays used for calls.
        Arrays are intentionally tiny because the dispatcher does not execute model code;
        we patch the trainer in tests that need to check forwarding.
        """
        # shapes: (n_samples, seq_len, seq_features) and (n_samples, static_features)
        self.X_sequence_train = np.zeros((2, 5, 2), dtype=float)
        self.X_static_train = np.zeros((2, 3), dtype=float)
        self.target_mask_train = np.zeros((2, 5, 1), dtype=float)
        self.y_train = np.zeros((2, 5, 1), dtype=float)

        self.X_sequence_val = np.zeros((1, 5, 2), dtype=float)
        self.X_static_val = np.zeros((1, 3), dtype=float)
        self.target_mask_val = np.zeros((1, 5, 1), dtype=float)
        self.y_val = np.zeros((1, 5, 1), dtype=float)

        # a small valid hyperparameter dict
        self.hp = {"seq_units": 16, "epochs": 1, "batch_size": 2}

    def test_model_name_type_validation_raises_type_error(self):
        """If model_name is not a string, a TypeError should be raised (early validation)."""
        with self.assertRaises(TypeError) as cm:
            train.train_imputation_model(
                model_name=123,  # invalid type
                imputation_model_params=self.hp,
                X_sequence_train=self.X_sequence_train,
                X_static_train=self.X_static_train,
                target_mask_train=self.target_mask_train,
                y_train=self.y_train,
                X_sequence_val=self.X_sequence_val,
                X_static_val=self.X_static_val,
                target_mask_val=self.target_mask_val,
                y_val=self.y_val,
            )
        self.assertIn("model_name must be a string", str(cm.exception))

    def test_invalid_model_name_value_raises_value_error(self):
        """A string model_name not in the supported set should raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            train.train_imputation_model(
                model_name="unknown_model",
                imputation_model_params=self.hp,
                X_sequence_train=self.X_sequence_train,
                X_static_train=self.X_static_train,
                target_mask_train=self.target_mask_train,
                y_train=self.y_train,
                X_sequence_val=self.X_sequence_val,
                X_static_val=self.X_static_val,
                target_mask_val=self.target_mask_val,
                y_val=self.y_val,
            )
        self.assertIn("Invalid model_name", str(cm.exception))

    @patch("price_imputation_and_forecasting.imputation.train.train_bi_rnn")
    def test_dispatch_calls_train_bi_rnn_with_forwarded_params(self, mock_train_bi_rnn):
        """
        Ensure that when model_name == 'bi_rnn', the dispatcher calls train_bi_rnn exactly once and forwards all the provided arrays and hyperparameters as keyword arguments.
        """
        # prepare mock return (model, history)
        mock_model = MagicMock(name="mock_model")
        mock_history = MagicMock(name="mock_history")
        mock_train_bi_rnn.return_value = (mock_model, mock_history)

        # call dispatcher
        result_model, result_history = train.train_imputation_model(
            model_name="bi_rnn",
            imputation_model_params=self.hp,
            X_sequence_train=self.X_sequence_train,
            X_static_train=self.X_static_train,
            target_mask_train=self.target_mask_train,
            y_train=self.y_train,
            X_sequence_val=self.X_sequence_val,
            X_static_val=self.X_static_val,
            target_mask_val=self.target_mask_val,
            y_val=self.y_val,
        )

        # assert the trainer was called exactly once
        mock_train_bi_rnn.assert_called_once()

        # check that kwargs forwarded to train_bi_rnn contain our arrays AND hyperparams
        called_kwargs = mock_train_bi_rnn.call_args.kwargs

        # arrays must be forwarded as keyword args
        self.assertIs(called_kwargs["X_sequence_train"], self.X_sequence_train)
        self.assertIs(called_kwargs["X_static_train"], self.X_static_train)
        self.assertIs(called_kwargs["target_mask_train"], self.target_mask_train)
        self.assertIs(called_kwargs["y_train"], self.y_train)
        self.assertIs(called_kwargs["X_sequence_val"], self.X_sequence_val)
        self.assertIs(called_kwargs["X_static_val"], self.X_static_val)
        self.assertIs(called_kwargs["target_mask_val"], self.target_mask_val)
        self.assertIs(called_kwargs["y_val"], self.y_val)

        # hyperparameters from imputation_model_params should appear in the kwargs
        for k, v in self.hp.items():
            self.assertIn(k, called_kwargs)
            self.assertEqual(called_kwargs[k], v)

        # the dispatcher should return exactly what the trainer returned
        self.assertIs(result_model, mock_model)
        self.assertIs(result_history, mock_history)

    @patch("price_imputation_and_forecasting.imputation.train.train_bi_rnn")
    def test_trainer_side_effect_propagates(self, mock_train_bi_rnn):
        """
        If train_bi_rnn raises an exception, train_imputation_model should not swallow it, but should re-raise it (after printing inside the function).
        """
        mock_train_bi_rnn.side_effect = RuntimeError("simulated training failure")
        with self.assertRaises(RuntimeError) as cm:
            train.train_imputation_model(
                model_name="bi_rnn",
                imputation_model_params=self.hp,
                X_sequence_train=self.X_sequence_train,
                X_static_train=self.X_static_train,
                target_mask_train=self.target_mask_train,
                y_train=self.y_train,
                X_sequence_val=self.X_sequence_val,
                X_static_val=self.X_static_val,
                target_mask_val=self.target_mask_val,
                y_val=self.y_val,
            )
        self.assertIn("simulated training failure", str(cm.exception))

    def test_imputation_model_params_none_raises_type_error(self):
        """
        Passing imputation_model_params=None leads to a Python TypeError because **None is invalid.
        The dispatcher currently expects a mapping; test that the error surfaces.
        """
        with self.assertRaises(TypeError):
            train.train_imputation_model(
                model_name="bi_rnn",
                imputation_model_params=None,  # invalid (must be dict)
                X_sequence_train=self.X_sequence_train,
                X_static_train=self.X_static_train,
                target_mask_train=self.target_mask_train,
                y_train=self.y_train,
                X_sequence_val=self.X_sequence_val,
                X_static_val=self.X_static_val,
                target_mask_val=self.target_mask_val,
                y_val=self.y_val,
            )

    def test_imputation_model_params_conflicting_keys_raise_type_error(self):
        """
        If imputation_model_params includes keys that conflict with explicit keyword args (e.g. 'X_sequence_train'), Python will raise TypeError: multiple values for arg.
        We verify that behavior surfaces from the dispatcher.
        """
        bad_hp = {"X_sequence_train": "malicious_override"}  # conflict key
        with self.assertRaises(TypeError):
            train.train_imputation_model(
                model_name="bi_rnn",
                imputation_model_params=bad_hp,
                X_sequence_train=self.X_sequence_train,
                X_static_train=self.X_static_train,
                target_mask_train=self.target_mask_train,
                y_train=self.y_train,
                X_sequence_val=self.X_sequence_val,
                X_static_val=self.X_static_val,
                target_mask_val=self.target_mask_val,
                y_val=self.y_val,
            )

    def test_model_name_case_sensitive(self):
        """Dispatcher expects exact 'bi_rnn' string; different casing should raise ValueError."""
        with self.assertRaises(ValueError):
            train.train_imputation_model(
                model_name="BI_RNN",  # wrong casing
                imputation_model_params=self.hp,
                X_sequence_train=self.X_sequence_train,
                X_static_train=self.X_static_train,
                target_mask_train=self.target_mask_train,
                y_train=self.y_train,
                X_sequence_val=self.X_sequence_val,
                X_static_val=self.X_static_val,
                target_mask_val=self.target_mask_val,
                y_val=self.y_val,
            )


if __name__ == "__main__":
    unittest.main()