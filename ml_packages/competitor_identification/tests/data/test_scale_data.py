import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from competitor_identification.data.scale_data import scale_data


class TestScaleData(unittest.TestCase):
    """
    Unit tests for the scale_data function.
    """

    def setUp(self):
        """
        Set up any mock data or initial conditions required for the tests.
        """
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'Period1': [1.0, 2.0, 3.0],
            'Period2': [2.0, 3.0, 4.0],
            'Period3': [3.0, 4.0, 5.0]
        }, index=['Product1', 'Product2', 'Product3'])

        self.empty_data = pd.DataFrame()

        self.non_numeric_data = pd.DataFrame({
            'Period1': [1.0, 2.0, 'X'],
            'Period2': [2.0, 'Y', 4.0],
            'Period3': [3.0, 4.0, 5.0]
        }, index=['Product1', 'Product2', 'Product3'])

        self.single_row_data = pd.DataFrame({
            'Period1': [1.0],
            'Period2': [2.0],
            'Period3': [3.0]
        }, index=['Product1'])

        self.same_value_data = pd.DataFrame({
            'Period1': [5.0, 5.0, 5.0],
            'Period2': [5.0, 5.0, 5.0],
            'Period3': [5.0, 5.0, 5.0]
        }, index=['Product1', 'Product2', 'Product3'])

    def test_scale_data_success(self):
        """
        Test the scaling of data with valid input.
        """
        price_movement_df, price_level_df = scale_data(self.sample_data)

        # Assert that the returned DataFrames are not empty
        self.assertFalse(price_movement_df.empty)
        self.assertFalse(price_level_df.empty)

        # Check that the shape of the scaled DataFrames matches the input
        self.assertEqual(price_movement_df.shape, self.sample_data.shape)
        self.assertEqual(price_level_df.shape, self.sample_data.shape)

    def test_scale_data_empty_dataframe(self):
        """
        Test that an empty DataFrame raises a ValueError.
        """
        with self.assertRaises(ValueError):
            scale_data(self.empty_data)

    def test_scale_data_non_numeric_data(self):
        """
        Test that non-numeric data raises a ValueError.
        """
        with self.assertRaises(ValueError):
            scale_data(self.non_numeric_data)

    def test_scale_data_type_error(self):
        """
        Test that input of non-DataFrame type raises a TypeError.
        """
        with self.assertRaises(TypeError):
            scale_data([1.0, 2.0, 3.0])  # List input instead of DataFrame

    def test_scale_data_invalid_scaling(self):
        """
        Test that scaling raises a ValueError when there is an issue during scaling.
        """
        with patch('sklearn.preprocessing.StandardScaler.fit_transform', side_effect=Exception("Scaling error")):
            with self.assertRaises(ValueError):
                scale_data(self.sample_data)

    def test_scale_data_check_index_and_columns(self):
        """
        Test that the scaling does not change the original DataFrame's index and column labels.
        """
        price_movement_df, price_level_df = scale_data(self.sample_data)

        # Check that index and columns are retained after scaling
        self.assertTrue(np.array_equal(price_movement_df.index, self.sample_data.index))
        self.assertTrue(np.array_equal(price_level_df.index, self.sample_data.index))
        self.assertTrue(np.array_equal(price_movement_df.columns, self.sample_data.columns))
        self.assertTrue(np.array_equal(price_level_df.columns, self.sample_data.columns))

    def test_scale_data_check_shape(self):
        """
        Test that scaling works correctly for DataFrame of any shape.
        """
        data = pd.DataFrame({
            'Period1': [1.0, 2.0],
            'Period2': [2.0, 3.0],
            'Period3': [3.0, 4.0]
        }, index=['Product1', 'Product2'])

        price_movement_df, price_level_df = scale_data(data)

        self.assertEqual(price_movement_df.shape, data.shape)
        self.assertEqual(price_level_df.shape, data.shape)


if __name__ == "__main__":
    unittest.main()