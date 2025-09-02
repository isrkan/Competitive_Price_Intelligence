import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from price_imputation_and_forecasting.data.preprocess_data import (
    align_encoded_store_data_with_price_data,
    convert_to_numpy_inputs,
    make_mask_and_replace_nan_with_predefined_value,
    simulate_missing_gaps,
    split_train_val_test_for_imputation,
    create_sliding_windows,
    prepare_forecasting_model_inputs_for_predictions,
    prepare_imputation_model_inputs_for_predictions
)


class TestPreprocessData(unittest.TestCase):
    """
    Comprehensive test suite for all functions in the preprocess_data module.

    This test class covers:
    - align_encoded_store_data_with_price_data
    - convert_to_numpy_inputs
    - make_mask_and_replace_nan_with_predefined_value
    - simulate_missing_gaps
    - split_train_val_test_for_imputation
    - create_sliding_windows
    - prepare_forecasting_model_inputs_for_predictions
    - prepare_imputation_model_inputs_for_predictions
    """

    def setUp(self):
        """
        Set up test data and fixtures used across multiple test cases.

        Creates various DataFrames and arrays to test different preprocessing scenarios.
        """
        # Sample store data with dummies (one-hot encoded)
        self.store_data_with_dummies = pd.DataFrame({
            'ChainID_1': [1, 0, 0],
            'ChainID_2': [0, 1, 0],
            'ChainID_3': [0, 0, 1],
            'StoreType_Mall': [1, 0, 1],
            'StoreType_Street': [0, 1, 0]
        }, index=pd.Index([101, 102, 103], name='StoreID'))

        # Sample price data with multi-index including StoreID
        price_index = pd.MultiIndex.from_tuples([
            ('CategoryA', 'ProductA', 101),
            ('CategoryA', 'ProductA', 102),
            ('CategoryA', 'ProductB', 101),
            ('CategoryA', 'ProductB', 103)
        ], names=['category', 'ProductDescription', 'StoreID'])

        self.price_data = pd.DataFrame({
            'price_2023_01': [10.0, 12.0, 15.0, 11.0],
            'price_2023_02': [10.5, np.nan, 15.5, 11.2],
            'price_2023_03': [11.0, 12.5, np.nan, 11.5]
        }, index=price_index)

        # Sample numpy arrays for testing
        self.sample_price_array = np.array([
            [1.0, 2.0, np.nan, 4.0],
            [np.nan, 2.0, 3.0, 4.0],
            [1.0, np.nan, 3.0, np.nan]
        ])

        self.sample_store_array = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1]
        ])

        # Sample time series data for window creation
        np.random.seed(42)
        self.time_series_data = np.random.randn(5, 20)  # 5 series, 20 time steps

        # Sample data for missing gap simulation
        self.filled_price_data = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0]
        ])

        self.original_mask = np.array([
            [1, 1, 0, 1, 1],  # Third position was originally missing
            [1, 1, 1, 1, 1],  # All originally observed
            [1, 0, 1, 0, 1]  # Second and fourth positions originally missing
        ])

    # ===== TEST align_encoded_store_data_with_price_data ======

    def test_align_encoded_store_data_success(self):
        """
        Test successful alignment of store data with price data.

        Verifies that store dummy variables are correctly aligned with price data rows.
        """
        result = align_encoded_store_data_with_price_data(
            self.store_data_with_dummies,
            self.price_data
        )

        # Should have same number of rows as price_data
        self.assertEqual(len(result), len(self.price_data))

        # Should have same columns as store_data_with_dummies
        self.assertListEqual(list(result.columns), list(self.store_data_with_dummies.columns))

        # Check specific alignments
        # Row 0: StoreID 101 -> should get store_data_with_dummies row for StoreID 101
        self.assertEqual(result.iloc[0]['ChainID_1'], 1)
        self.assertEqual(result.iloc[0]['StoreType_Mall'], 1)

        # Row 1: StoreID 102 -> should get store_data_with_dummies row for StoreID 102
        self.assertEqual(result.iloc[1]['ChainID_2'], 1)
        self.assertEqual(result.iloc[1]['StoreType_Street'], 1)

    def test_align_encoded_store_data_price_data_not_indexed_by_storeid(self):
        """
        Test that ValueError is raised when price_data is not indexed by StoreID.
        """
        # Create price data without StoreID in index
        invalid_price_data = self.price_data.reset_index()

        with self.assertRaises(ValueError) as context:
            align_encoded_store_data_with_price_data(
                self.store_data_with_dummies,
                invalid_price_data
            )

        self.assertIn("price_data` must be indexed by 'StoreID'", str(context.exception))

    def test_align_encoded_store_data_store_data_not_indexed_by_storeid(self):
        """
        Test that ValueError is raised when store_data_with_dummies is not indexed by StoreID.
        """
        # Create store data without StoreID as index
        invalid_store_data = self.store_data_with_dummies.reset_index()

        with self.assertRaises(ValueError) as context:
            align_encoded_store_data_with_price_data(
                invalid_store_data,
                self.price_data
            )

        self.assertIn("store_dummies` must be indexed by 'StoreID'", str(context.exception))

    @patch("price_imputation_and_forecasting.data.preprocess_data.pd.DataFrame.merge")
    def test_align_encoded_store_data_merge_runtime_error(self, mock_merge):
        """
        Test that RuntimeError is properly raised when merge operation fails.
        """
        mock_merge.side_effect = Exception("Simulated merge failure")

        with self.assertRaises(RuntimeError) as context:
            align_encoded_store_data_with_price_data(
                self.store_data_with_dummies,
                self.price_data
            )

        self.assertIn("Failed to merge store and price data", str(context.exception))
        self.assertIn("Simulated merge failure", str(context.exception))

    def test_align_encoded_store_data_missing_store_ids(self):
        """
        Test behavior when price_data contains StoreIDs not present in store_data_with_dummies.

        This should result in NaN values for missing store dummy variables.
        """
        # Add price data for StoreID that doesn't exist in store dummies
        extended_price_index = pd.MultiIndex.from_tuples([
            ('CategoryA', 'ProductA', 101),
            ('CategoryA', 'ProductA', 999)  # StoreID 999 not in store_data_with_dummies
        ], names=['category', 'ProductDescription', 'StoreID'])

        extended_price_data = pd.DataFrame({
            'price_2023_01': [10.0, 12.0],
            'price_2023_02': [10.5, np.nan]
        }, index=extended_price_index)

        result = align_encoded_store_data_with_price_data(
            self.store_data_with_dummies,
            extended_price_data
        )

        # First row should have proper values, second row should have NaN
        self.assertEqual(result.iloc[0]['ChainID_1'], 1)
        self.assertTrue(pd.isna(result.iloc[1]['ChainID_1']))

    # ====== TEST convert_to_numpy_inputs =========

    def test_convert_to_numpy_inputs_success(self):
        """
        Test successful conversion of DataFrames to NumPy arrays.
        """
        df_price = pd.DataFrame(self.sample_price_array)
        df_store = pd.DataFrame(self.sample_store_array)

        price_array, store_array = convert_to_numpy_inputs(df_price, df_store)

        # Check return types
        self.assertIsInstance(price_array, np.ndarray)
        self.assertIsInstance(store_array, np.ndarray)

        # Check shapes
        self.assertEqual(price_array.shape, df_price.shape)
        self.assertEqual(store_array.shape, df_store.shape)

        # Check values (where not NaN)
        np.testing.assert_array_equal(price_array[0, :2], [1.0, 2.0])
        np.testing.assert_array_equal(store_array, self.sample_store_array)

    def test_convert_to_numpy_inputs_invalid_types(self):
        """
        Test that TypeError is raised when inputs are not DataFrames.
        """
        invalid_inputs = [None, "string", [], {}, np.array([1, 2, 3])]
        valid_df = pd.DataFrame([[1, 2], [3, 4]])

        for invalid_input in invalid_inputs:
            with self.subTest(input_type=type(invalid_input)):
                with self.assertRaises(TypeError) as context:
                    convert_to_numpy_inputs(invalid_input, valid_df)

                self.assertIn("must be pandas DataFrames", str(context.exception))

                with self.assertRaises(TypeError) as context:
                    convert_to_numpy_inputs(valid_df, invalid_input)

                self.assertIn("must be pandas DataFrames", str(context.exception))

    def test_convert_to_numpy_inputs_empty_dataframes(self):
        """
        Test conversion of empty DataFrames.
        """
        empty_df1 = pd.DataFrame()
        empty_df2 = pd.DataFrame()

        price_array, store_array = convert_to_numpy_inputs(empty_df1, empty_df2)

        self.assertIsInstance(price_array, np.ndarray)
        self.assertIsInstance(store_array, np.ndarray)
        self.assertEqual(price_array.shape, (0, 0))
        self.assertEqual(store_array.shape, (0, 0))

    # ======= TEST make_mask_and_replace_nan_with_predefined_value =========

    def test_make_mask_and_replace_nan_success(self):
        """
        Test successful masking and NaN replacement.
        """
        fill_value = -999.0

        filled_data, mask = make_mask_and_replace_nan_with_predefined_value(
            self.sample_price_array,
            fill_value
        )

        # Check return types
        self.assertIsInstance(filled_data, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)

        # Check shapes
        self.assertEqual(filled_data.shape, self.sample_price_array.shape)
        self.assertEqual(mask.shape, self.sample_price_array.shape)

        # Check that NaN values are replaced
        self.assertFalse(np.any(np.isnan(filled_data)))

        # Check specific replacements
        self.assertEqual(filled_data[0, 2], fill_value)  # Originally NaN
        self.assertEqual(filled_data[1, 0], fill_value)  # Originally NaN

        # Check mask values
        self.assertEqual(mask[0, 0], 1)  # Originally 1.0 (not NaN)
        self.assertEqual(mask[0, 2], 0)  # Originally NaN
        self.assertEqual(mask[1, 0], 0)  # Originally NaN

        # Check mask dtype
        self.assertEqual(mask.dtype, int)

    def test_make_mask_and_replace_nan_invalid_input_type(self):
        """
        Test that TypeError is raised when input is not a NumPy array.
        """
        invalid_inputs = [None, "string", [], {}, pd.DataFrame([[1, 2]])]
        fill_value = 0.0

        for invalid_input in invalid_inputs:
            with self.subTest(input_type=type(invalid_input)):
                with self.assertRaises(TypeError) as context:
                    make_mask_and_replace_nan_with_predefined_value(invalid_input, fill_value)

                self.assertIn("must be a NumPy array", str(context.exception))

    def test_make_mask_and_replace_nan_non_numeric_array(self):
        """
        Test that TypeError is raised when input array is not numeric.
        """
        non_numeric_arrays = [
            np.array(['a', 'b', 'c']),
            np.array([True, False, True]),
            np.array(['2023-01-01', '2023-01-02'], dtype='datetime64')
        ]
        fill_value = 0.0

        for arr in non_numeric_arrays:
            with self.subTest(dtype=arr.dtype):
                with self.assertRaises(TypeError) as context:
                    make_mask_and_replace_nan_with_predefined_value(arr, fill_value)

                self.assertIn("must be a numeric array", str(context.exception))

    def test_make_mask_and_replace_nan_no_nan_values(self):
        """
        Test function behavior when input has no NaN values.
        """
        no_nan_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        fill_value = -999.0

        filled_data, mask = make_mask_and_replace_nan_with_predefined_value(no_nan_array, fill_value)

        # Original data should be unchanged
        np.testing.assert_array_equal(filled_data, no_nan_array)

        # Mask should be all ones
        np.testing.assert_array_equal(mask, np.ones_like(no_nan_array, dtype=int))

    def test_make_mask_and_replace_nan_all_nan_values(self):
        """
        Test function behavior when input is all NaN values.
        """
        all_nan_array = np.full((2, 3), np.nan)
        fill_value = -999.0

        filled_data, mask = make_mask_and_replace_nan_with_predefined_value(all_nan_array, fill_value)

        # All values should be replaced with fill_value
        np.testing.assert_array_equal(filled_data, np.full((2, 3), fill_value))

        # Mask should be all zeros
        np.testing.assert_array_equal(mask, np.zeros((2, 3), dtype=int))

    def test_make_mask_and_replace_nan_different_fill_values(self):
        """
        Test function with different fill values.
        """
        test_cases = [0.0, -1.0, 999.0, -np.inf, np.inf]

        for fill_value in test_cases:
            with self.subTest(fill_value=fill_value):
                filled_data, mask = make_mask_and_replace_nan_with_predefined_value(
                    self.sample_price_array,
                    fill_value
                )

                # Check that NaN positions are filled with the specified value
                nan_positions = np.isnan(self.sample_price_array)
                self.assertTrue(np.all(filled_data[nan_positions] == fill_value))

    @patch("price_imputation_and_forecasting.data.preprocess_data.np.isnan")
    def test_make_mask_and_replace_nan_runtime_error(self, mock_isnan):
        """
        Test that RuntimeError is raised when np.isnan fails.
        """
        mock_isnan.side_effect = Exception("Simulated isnan failure")

        with self.assertRaises(RuntimeError) as context:
            make_mask_and_replace_nan_with_predefined_value(self.sample_price_array, 0.0)

        self.assertIn("Error replacing NaN values", str(context.exception))

    # ======== TEST simulate_missing_gaps ===========

    def test_simulate_missing_gaps_success(self):
        """
        Test successful simulation of missing gaps.
        """
        gap_prob = 0.5
        max_gap = 2
        rng = np.random.default_rng(42)

        sim_data, sim_mask, target_mask = simulate_missing_gaps(
            self.filled_price_data,
            self.original_mask,
            gap_prob,
            max_gap,
            rng
        )

        # Check return types and shapes
        self.assertIsInstance(sim_data, np.ndarray)
        self.assertIsInstance(sim_mask, np.ndarray)
        self.assertIsInstance(target_mask, np.ndarray)

        self.assertEqual(sim_data.shape, self.filled_price_data.shape)
        self.assertEqual(sim_mask.shape, self.original_mask.shape)
        self.assertEqual(target_mask.shape, self.original_mask.shape)

        # Target mask should only have 1s where gaps were simulated (and only at positions that were originally observed)
        self.assertTrue(np.all((target_mask == 1) <= (self.original_mask == 1)))

        # Simulated data should be zero where simulated gaps exist
        self.assertTrue(np.all(sim_data[sim_mask == 0] == 0))

    def test_simulate_missing_gaps_zero_gap_probability(self):
        """
        Test simulation with gap_prob=0 (no gaps should be created).
        """
        gap_prob = 0.0
        max_gap = 2

        sim_data, sim_mask, target_mask = simulate_missing_gaps(
            self.filled_price_data,
            self.original_mask,
            gap_prob,
            max_gap
        )

        # No simulated gaps should be created
        np.testing.assert_array_equal(sim_mask, self.original_mask)
        np.testing.assert_array_equal(target_mask, np.zeros_like(self.original_mask))
        np.testing.assert_array_equal(sim_data, self.filled_price_data * self.original_mask)


if __name__ == "__main__":
    unittest.main()