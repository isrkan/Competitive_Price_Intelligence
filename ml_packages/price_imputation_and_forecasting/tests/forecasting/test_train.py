import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from price_imputation_and_forecasting.forecasting import train as train_module


class TestTrainForecastingModelDispatcher(unittest.TestCase):
    """
    Unit tests for forecasting.train.train_forecasting_model dispatcher.

    These tests focus on the dispatcher's validation and wiring only.
    The actual training function (train_gru) is patched so that tests are fast, deterministic, and do not require TensorFlow or GPU resources.

    The key behaviors tested:
      - model_name type and allowed-value validation
      - correct forwarding of array inputs and hyperparameters to train_gru
      - propagation of exceptions from train_gru
      - behavior with forecasting_model_params=None
      - conflicting keys in forecasting_model_params producing TypeError
      - case-sensitivity of model_name
    """

    def setUp(self):
        """
        Prepare small dummy arrays used in multiple tests.
        Using tiny arrays keeps tests fast and avoids heavy numeric operations.
        """
        # X_train, y_train shapes expected: (n_samples, lookback) and (n_samples, horizon)
        self.X_train = np.zeros((3, 10), dtype=float)
        self.y_train = np.zeros((3, 4), dtype=float)

        self.X_val = np.zeros((1, 10), dtype=float)
        self.y_val = np.zeros((1, 4), dtype=float)

        # Typical hyperparameters dictionary a user would pass from config
        self.hp = {
            "gru_units": 32,
            "dense_units": 16,
            "dropout_rate": 0.2,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3
        }

    def test_model_name_type_validation_raises_type_error(self):
        """If model_name is not a string, a TypeError should be raised."""
        with self.assertRaises(TypeError) as cm:
            train_module.train_forecasting_model(
                model_name=123,  # invalid type
                forecasting_model_params=self.hp,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
            )
        self.assertIn("model_name must be a string", str(cm.exception))

    def test_invalid_model_name_value_raises_value_error(self):
        """If model_name is an unsupported string, raise ValueError."""
        with self.assertRaises(ValueError) as cm:
            train_module.train_forecasting_model(
                model_name="unknown_forecaster",
                forecasting_model_params=self.hp,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
            )
        self.assertIn("Invalid model_name", str(cm.exception))

    @patch("price_imputation_and_forecasting.forecasting.gru.train_gru")
    def test_dispatch_calls_train_gru_with_forwarded_params(self, mock_train_gru):
        """
        When model_name == 'gru', the dispatcher must call train_gru exactly once and
        forward all arrays + hyperparameters as keyword args.
        """
        # Prepare a fake return (model, history)
        dummy_model = MagicMock(name="dummy_model")
        dummy_history = MagicMock(name="dummy_history")
        mock_train_gru.return_value = (dummy_model, dummy_history)

        # Call the dispatcher
        model_out, history_out = train_module.train_forecasting_model(
            model_name="gru",
            forecasting_model_params=self.hp,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
        )

        # Ensure train_gru was invoked once
        mock_train_gru.assert_called_once()

        # Inspect the kwargs forwarded to train_gru
        called_kwargs = mock_train_gru.call_args.kwargs

        # Confirm that arrays are forwarded unchanged (identity check)
        self.assertIs(called_kwargs["X_train"], self.X_train)
        self.assertIs(called_kwargs["y_train"], self.y_train)
        self.assertIs(called_kwargs["X_val"], self.X_val)
        self.assertIs(called_kwargs["y_val"], self.y_val)

        # Confirm hyperparameters from forecasting_model_params got injected
        for k, v in self.hp.items():
            self.assertIn(k, called_kwargs)
            self.assertEqual(called_kwargs[k], v)

        # Check dispatcher returned exactly what the trainer returned
        self.assertIs(model_out, dummy_model)
        self.assertIs(history_out, dummy_history)

    @patch("price_imputation_and_forecasting.forecasting.gru.train_gru")
    def test_trainer_side_effect_propagates(self, mock_train_gru):
        """
        If train_gru raises an exception, the dispatcher should not swallow it;
        it should propagate (i.e., the caller sees the exception).
        """
        mock_train_gru.side_effect = RuntimeError("simulated training error")

        with self.assertRaises(RuntimeError) as cm:
            train_module.train_forecasting_model(
                model_name="gru",
                forecasting_model_params=self.hp,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
            )
        self.assertIn("simulated training error", str(cm.exception))

    def test_forecasting_model_params_none_raises_type_error(self):
        """
        Passing forecasting_model_params=None will result in a TypeError because
        the dispatcher uses **forecasting_model_params internally. Ensure this surfaces.
        """
        with self.assertRaises(TypeError):
            train_module.train_forecasting_model(
                model_name="gru",
                forecasting_model_params=None,  # invalid usage
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
            )

    def test_forecasting_model_params_conflicting_keys_raise_type_error(self):
        """
        If forecasting_model_params includes keys that conflict with explicit keyword args (e.g., 'X_train'), Python will raise TypeError for multiple values for that parameter.
        Ensure the dispatcher does not mask that error.
        """
        bad_hp = {"X_train": "malicious_override"}
        with self.assertRaises(TypeError):
            train_module.train_forecasting_model(
                model_name="gru",
                forecasting_model_params=bad_hp,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
            )

    def test_model_name_case_sensitive(self):
        """Model name is expected to match exactly; wrong casing should raise ValueError."""
        with self.assertRaises(ValueError):
            train_module.train_forecasting_model(
                model_name="GRU",  # wrong casing
                forecasting_model_params=self.hp,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
            )


if __name__ == "__main__":
    unittest.main()