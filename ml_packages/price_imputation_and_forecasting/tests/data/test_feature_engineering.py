import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from price_imputation_and_forecasting.data.feature_engineering import encode_store_data, scale_price_inputs, inverse_scale_price_inputs_with_expm1, round_prices_to_nearest_10_cents


class TestFeatureEngineering(unittest.TestCase):
    """
    Extensive unit tests for feature_engineering.py

    These tests exercise:
    - encode_store_data: correct one-hot encoding, index setting, missing required column handling, and runtime error wrapping when internal pandas.get_dummies fails.
    - scale_price_inputs: correct log1p scaling, input validation (types & shapes), and error wrapping.
    - inverse_scale_price_inputs_with_expm1: correct inverse transform and error wrapping.
    - round_prices_to_nearest_10_cents: rounding semantics, zero-handling (set to 0.1), dtype & shape preservation and error wrapping.
    """

    def setUp(self):
        """Prepare common fixtures used across tests."""
        # ---- store data fixture for encode_store_data tests ----
        # 3 rows, simple categorical values that will generate (k-1) dummies because drop_first=True
        self.store_df = pd.DataFrame(
            {
                "StoreID": [11, 22, 33],
                "ChainID": ["C1", "C2", "C1"],
                "DistrictName": ["D1", "D1", "D2"],
                "StoreType": ["S1", "S2", "S1"],
                "LocationType": ["L1", "L2", "L1"],
                # other columns should be ignored by encode_store_data (it subsets to required columns)
                "ExtraCol": ["x", "y", "z"],
            }
        )

        # ---- numeric fixtures for scaling and rounding tests ----
        # shape (2, 3) example array
        self.raw = np.array([[0.0, 1.0, 9.0], [2.0, 3.0, 4.0]], dtype=float)

    # =========================
    # Tests for encode_store_data
    # =========================
    def test_encode_store_data_happy_path(self):
        """
        Happy path: One-hot encoding with drop_first=True should produce (k-1) dummy columns for each categorical column,
        set index to StoreID, and preserve numeric types (non-bool).
        """
        encoded = encode_store_data(self.store_df.copy())

        # Expected dummy columns (because each column has 2 unique values -> 1 dummy each)
        expected_dummy_cols = {
            "ChainID_C2",         # ChainID had C1 (dropped) and C2 (kept)
            "DistrictName_D2",    # DistrictName had D1 (dropped) and D2 (kept)
            "StoreType_S2",       # StoreType had S1 (dropped) and S2 (kept)
            "LocationType_L2",    # LocationType had L1 (dropped) and L2 (kept)
        }

        self.assertTrue(set(expected_dummy_cols).issubset(set(encoded.columns)))
        # Index must be StoreID and length must match original
        self.assertEqual(encoded.index.name, "StoreID")
        self.assertEqual(len(encoded), len(self.store_df))

        # Value checks for a specific row
        # StoreID 11 had ChainID C1 → ChainID_C2 == 0
        self.assertEqual(int(encoded.loc[11, "ChainID_C2"]), 0)
        # StoreID 22 had ChainID C2 → ChainID_C2 == 1
        self.assertEqual(int(encoded.loc[22, "ChainID_C2"]), 1)

        # All encoded columns should be numeric (not boolean)
        for col in expected_dummy_cols:
            self.assertTrue(np.issubdtype(encoded[col].dtype, np.integer) or np.issubdtype(encoded[col].dtype, np.floating))

    def test_encode_store_data_missing_required_columns_raises(self):
        """If one of the required columns is missing, a ValueError should be raised listing the missing columns."""
        bad_df = self.store_df.drop(columns=["StoreType"])
        with self.assertRaises(ValueError) as ctx:
            encode_store_data(bad_df)
        self.assertIn("Missing required columns", str(ctx.exception))
        self.assertIn("StoreType", str(ctx.exception))

    def test_encode_store_data_runtime_error_wrapped(self):
        """
        Simulate an internal failure in pandas.get_dummies (e.g., unexpected error).
        encode_store_data should catch and re-raise as RuntimeError with context.
        """
        # Patch pandas.get_dummies used inside our module to raise an Exception
        with patch("price_imputation_and_forecasting.data.feature_engineering.pd.get_dummies") as mock_get_dummies:
            mock_get_dummies.side_effect = Exception("simulated failure in get_dummies")
            with self.assertRaises(RuntimeError) as ctx:
                encode_store_data(self.store_df)
            self.assertIn("Error encoding store data", str(ctx.exception))
            self.assertIn("simulated failure in get_dummies", str(ctx.exception))

    # =========================
    # Tests for scale_price_inputs
    # =========================
    def test_scale_price_inputs_happy_path(self):
        """log1p transform should match numpy's np.log1p result and shape should be preserved."""
        a = self.raw.copy()
        b = self.raw.copy()
        s_a, s_b = scale_price_inputs(a, b)
        np.testing.assert_allclose(s_a, np.log1p(self.raw))
        np.testing.assert_allclose(s_b, np.log1p(self.raw))
        self.assertEqual(s_a.shape, self.raw.shape)
        self.assertEqual(s_b.shape, self.raw.shape)

    def test_scale_price_inputs_type_validation(self):
        """If inputs are not NumPy arrays, TypeError is raised."""
        with self.assertRaises(TypeError):
            scale_price_inputs(self.raw.tolist(), self.raw)

    def test_scale_price_inputs_shape_mismatch_raises(self):
        """If shapes mismatch, ValueError should be raised."""
        arr1 = np.zeros((2, 3))
        arr2 = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            scale_price_inputs(arr1, arr2)

    def test_scale_price_inputs_internal_error_wrapped(self):
        """
        Patch np.log1p inside the module to throw an Exception and assert we wrap it as RuntimeError.
        """
        with patch("price_imputation_and_forecasting.data.feature_engineering.np.log1p") as mock_log1p:
            mock_log1p.side_effect = Exception("simulated log1p failure")
            with self.assertRaises(RuntimeError) as ctx:
                scale_price_inputs(self.raw, self.raw)
            self.assertIn("Failed during log1p scaling", str(ctx.exception))
            self.assertIn("simulated log1p failure", str(ctx.exception))

    # =========================================
    # Tests for inverse_scale_price_inputs_with_expm1
    # =========================================
    def test_inverse_scale_price_inputs_round_trip(self):
        """Round-trip: original -> log1p -> expm1 should recover original values (within float tolerance)."""
        original = np.array([[0.0, 1.0, 9.0], [2.0, 3.0, 4.0]], dtype=float)
        scaled = np.log1p(original)
        recovered = inverse_scale_price_inputs_with_expm1(scaled)
        np.testing.assert_allclose(recovered, original, rtol=1e-7, atol=1e-8)

    def test_inverse_scale_type_validation(self):
        """Non-numpy input should raise TypeError."""
        with self.assertRaises(TypeError):
            inverse_scale_price_inputs_with_expm1([1, 2, 3])

    def test_inverse_scale_runtime_error_wrapped(self):
        """Simulate np.expm1 raising to ensure RuntimeError is raised with context."""
        with patch("price_imputation_and_forecasting.data.feature_engineering.np.expm1") as mock_expm1:
            mock_expm1.side_effect = Exception("simulated expm1 error")
            scaled = np.log1p(self.raw)
            with self.assertRaises(RuntimeError) as ctx:
                inverse_scale_price_inputs_with_expm1(scaled)
            self.assertIn("Failed during inverse expm1 scaling", str(ctx.exception))
            self.assertIn("simulated expm1 error", str(ctx.exception))

    # =========================================
    # Tests for round_prices_to_nearest_10_cents
    # =========================================
    def test_round_prices_to_nearest_10_cents_happy_path(self):
        """Values are rounded to nearest 0.1 and zeros are replaced with 0.1."""
        arr = np.array([9.91, 4.46, 0.0, 0.06, 0.04], dtype=float)
        rounded = round_prices_to_nearest_10_cents(arr.copy())
        # Expected:
        # 9.91 -> 9.9
        # 4.46 -> 4.5
        # 0.0  -> 0.1 (special rule)
        # 0.06 -> 0.1
        # 0.04 -> 0.0 -> then replaced to 0.1
        expected = np.array([9.9, 4.5, 0.1, 0.1, 0.1], dtype=float)
        np.testing.assert_allclose(rounded, expected)
        # original shape preserved
        self.assertEqual(rounded.shape, arr.shape)
        # dtype is float
        self.assertTrue(np.issubdtype(rounded.dtype, np.floating))

    def test_round_prices_multi_dimensional_preserved(self):
        """Ensure rounding preserves shape for 2D arrays and applies elementwise."""
        arr2d = np.array([[9.91, 0.0], [4.46, 0.06]], dtype=float)
        rounded2d = round_prices_to_nearest_10_cents(arr2d.copy())
        expected2d = np.array([[9.9, 0.1], [4.5, 0.1]], dtype=float)
        np.testing.assert_allclose(rounded2d, expected2d)
        self.assertEqual(rounded2d.shape, arr2d.shape)

    def test_round_prices_type_validation(self):
        """Non-numpy input should raise TypeError."""
        with self.assertRaises(TypeError):
            round_prices_to_nearest_10_cents([9.91, 4.46, 0.0])

    def test_round_prices_runtime_error_wrapped(self):
        """Simulate np.round raising to ensure RuntimeError is raised with context."""
        with patch("price_imputation_and_forecasting.data.feature_engineering.np.round") as mock_round:
            mock_round.side_effect = Exception("simulated round error")
            with self.assertRaises(RuntimeError) as ctx:
                round_prices_to_nearest_10_cents(np.array([1.0, 2.0]))
            self.assertIn("Failed during rounding prices", str(ctx.exception))
            self.assertIn("simulated round error", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()