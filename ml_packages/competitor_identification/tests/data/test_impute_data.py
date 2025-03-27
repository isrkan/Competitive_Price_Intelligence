import unittest
import pandas as pd
import numpy as np
from competitor_identification.data.impute_data import impute_missing_data

class TestImputeData(unittest.TestCase):
    """
    Unit tests for the impute_missing_data function.
    """

    def setUp(self):
        """
        Set up any mock data or initial conditions required for the tests.
        """
        # Sample data for testing
        self.valid_data = pd.DataFrame({
            'Period1': [1.0, 2.0, 3.0],
            'Period2': [2.0, np.nan, 4.0],
            'Period3': [3.0, 4.0, np.nan]
        }, index=['Product1', 'Product2', 'Product3'])

        self.empty_data = pd.DataFrame()

        self.non_numeric_data = pd.DataFrame({
            'Period1': [1.0, 2.0, 'X'],
            'Period2': [2.0, 'Y', 4.0],
            'Period3': [3.0, 4.0, 5.0]
        }, index=['Product1', 'Product2', 'Product3'])

        self.single_row_data = pd.DataFrame({
            'Period1': [1.0],
            'Period2': [np.nan],
            'Period3': [3.0]
        }, index=['Product1'])

        self.all_missing_data = pd.DataFrame({
            'Period1': [np.nan, np.nan, np.nan],
            'Period2': [np.nan, np.nan, np.nan],
            'Period3': [np.nan, np.nan, np.nan]
        }, index=['Product1', 'Product2', 'Product3'])

        self.no_missing_data = pd.DataFrame({
            'Period1': [1.0, 2.0, 3.0],
            'Period2': [4.0, 5.0, 6.0],
            'Period3': [7.0, 8.0, 9.0]
        }, index=['Product1', 'Product2', 'Product3'])

    def test_impute_data_valid(self):
        """
        Test imputation on a valid DataFrame with missing values.
        """
        imputed_df = impute_missing_data(self.valid_data)

        # Ensure that the imputed DataFrame has the same shape as the original
        self.assertEqual(imputed_df.shape, self.valid_data.shape)

        # Check that the NaNs were filled
        self.assertFalse(imputed_df.isna().any().any())

        # Verify the imputation process (we won't know exact values, but they should be filled)
        # Since the linear interpolation will take the average between Period1 and Period3
        self.assertEqual(imputed_df.loc['Product2', 'Period2'], 3.0)  # Expected value after interpolation for Product2, Period2

    def test_impute_data_empty(self):
        """
        Test that an error is raised when an empty DataFrame is passed.
        """
        with self.assertRaises(ValueError):
            impute_missing_data(self.empty_data)

    def test_impute_data_non_numeric(self):
        """
        Test that an error is raised when a DataFrame contains non-numeric data.
        """
        with self.assertRaises(ValueError):
            impute_missing_data(self.non_numeric_data)

    def test_impute_data_single_row(self):
        """
        Test imputation on a DataFrame with a single row and missing values.
        """
        imputed_df = impute_missing_data(self.single_row_data)

        # Ensure the imputed DataFrame is the same shape
        self.assertEqual(imputed_df.shape, self.single_row_data.shape)

        # Check that NaNs are filled (as we have only one row, interpolation and forward/backward fill will not change)
        self.assertFalse(imputed_df.isna().any().any())

        # Since the interpolation and imputation will not change a single-row DataFrame
        self.assertEqual(imputed_df.iloc[0]['Period2'], 2.0)

    def test_impute_data_all_missing(self):
        """
        Test that the function correctly handles a DataFrame with all missing values.
        """
        imputed_df = impute_missing_data(self.all_missing_data)

        # The imputation should be done, but since all values are missing, we expect the result to be NaN
        self.assertTrue(imputed_df.isna().all().all())

    def test_impute_data_no_missing(self):
        """
        Test that a DataFrame with no missing values returns unchanged.
        """
        imputed_df = impute_missing_data(self.no_missing_data)

        # The DataFrame should remain unchanged
        pd.testing.assert_frame_equal(imputed_df, self.no_missing_data)

    def test_impute_data_forward_fill(self):
        """
        Test that LOCB (Last Observation Carried Forward) works as expected.
        """
        df_with_nans = pd.DataFrame({
            'Period1': [1.0, np.nan, 3.0],
            'Period2': [4.0, np.nan, 6.0],
            'Period3': [7.0, np.nan, 9.0]
        }, index=['Product1', 'Product2', 'Product3'])

        imputed_df = impute_missing_data(df_with_nans)

        # After forward fill, Product1's values should match Product2's values
        self.assertEqual(imputed_df.loc['Product1', 'Period1'], 1.0)
        self.assertEqual(imputed_df.loc['Product1', 'Period2'], 4.0)
        self.assertEqual(imputed_df.loc['Product1', 'Period3'], 7.0)

if __name__ == "__main__":
    unittest.main()