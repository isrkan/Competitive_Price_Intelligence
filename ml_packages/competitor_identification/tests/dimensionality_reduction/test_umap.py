import unittest
import pandas as pd
from competitor_identification.dimensionality_reduction.umap import perform_umap_on_price_data


class TestUMAPFunction(unittest.TestCase):

    def setUp(self):
        """Set up test data for UMAP."""
        # DataFrame containing price data (P_M_1 and P_L_1 are placeholders for price features)
        self.price_data = pd.DataFrame({
            'P_M_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'P_L_1': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        })

        # DataFrame containing product metadata
        self.product_metadata = pd.DataFrame({
            'ChainID': [1, 2, 1, 3, 4, 5],
            'ChainName': ['Chain A', 'Chain B', 'Chain A', 'Chain C', 'Chain D', 'Chain E'],
            'SubChainID': [10, 20, 10, 30, 40, 50],
            'SubChainName': ['Sub A1', 'Sub B1', 'Sub A2', 'Sub C1', 'Sub D1', 'Sub E1'],
            'StoreID': [100, 101, 102, 103, 104, 105],
            'StoreName': ['Store A', 'Store B', 'Store C', 'Store D', 'Store E', 'Store F'],
            'DistrictName': ['District 1', 'District 2', 'District 3', 'District 4', 'District 5', 'District 6'],
            'SubDistrictName': ['SubD 1', 'SubD 2', 'SubD 3', 'SubD 4', 'SubD 5', 'SubD 6'],
            'CityName': ['City 1', 'City 2', 'City 3', 'City 4', 'City 5', 'City 6']
        })

        # Parameters for UMAP
        self.method_params = {'n_components': 2, 'random_state': 42, 'n_neighbors': 2}

    def test_valid_umap_transformation(self):
        """Test that UMAP transformation works with valid data."""
        result = perform_umap_on_price_data(self.price_data, self.product_metadata, self.method_params)

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame, "The result should be a pandas DataFrame.")

        # Check if the number of columns in the result matches the UMAP components and metadata columns
        self.assertEqual(result.shape[1], 2 + 9, "The result should have 2 UMAP components and 9 metadata columns.")

        # Check if the UMAP transformation worked by verifying component columns
        self.assertIn(0, result.columns, "UMAP component 0 column should be present.")
        self.assertIn(1, result.columns, "UMAP component 1 column should be present.")

    def test_empty_price_data(self):
        """Test that the function raises an error when price data (df) is empty."""
        empty_price_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            perform_umap_on_price_data(empty_price_data, self.product_metadata, self.method_params)

    def test_empty_metadata(self):
        """Test that the function raises an error when product metadata is empty."""
        empty_metadata = pd.DataFrame()
        with self.assertRaises(ValueError):
            perform_umap_on_price_data(self.price_data, empty_metadata, self.method_params)

    def test_invalid_price_data_type(self):
        """Test that the function raises a TypeError when the price data is not a DataFrame."""
        with self.assertRaises(TypeError):
            perform_umap_on_price_data("invalid_data", self.product_metadata, self.method_params)

    def test_invalid_metadata_type(self):
        """Test that the function raises a TypeError when the metadata is not a DataFrame."""
        with self.assertRaises(TypeError):
            perform_umap_on_price_data(self.price_data, "invalid_metadata", self.method_params)

    def test_invalid_method_params_type(self):
        """Test that the function raises a TypeError when method_params is not a dictionary."""
        with self.assertRaises(TypeError):
            perform_umap_on_price_data(self.price_data, self.product_metadata, "invalid_params")

    def test_check_umap_components(self):
        """Test that UMAP components are actually being computed (non-empty values in UMAP columns)."""
        result = perform_umap_on_price_data(self.price_data, self.product_metadata, self.method_params)

        # Check that the UMAP component columns are not empty
        self.assertTrue(result[0].notnull().any(), "UMAP component 0 should have values.")
        self.assertTrue(result[1].notnull().any(), "UMAP component 1 should have values.")

    def test_check_column_order_after_merge(self):
        """Test that the correct order of columns is maintained after UMAP and merge."""
        result = perform_umap_on_price_data(self.price_data, self.product_metadata, self.method_params)

        # Expected order of columns: UMAP components first, followed by metadata columns
        expected_column_order = [0, 1, 'ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']

        self.assertEqual(list(result.columns), expected_column_order, "The column order in the result is incorrect.")

    def test_invalid_merge_columns(self):
        """Test that the function raises an error if merge columns are missing in the metadata DataFrame."""
        missing_columns_metadata = self.product_metadata.drop(columns=['StoreID'])
        with self.assertRaises(KeyError):
            perform_umap_on_price_data(self.price_data, missing_columns_metadata, self.method_params)

    def test_non_numeric_price_data(self):
        """Test that an error is raised when price data contains non-numeric values."""
        non_numeric_price_data = self.price_data.copy()
        non_numeric_price_data['P_M_1'] = ['a', 'b', 'c', 'd', 'e', 'f']  # Non-numeric values
        with self.assertRaises(ValueError):
            perform_umap_on_price_data(non_numeric_price_data, self.product_metadata, self.method_params)

    def test_duplicate_entries_in_metadata(self):
        """Test that UMAP can still work if metadata contains duplicate entries."""
        duplicated_metadata = pd.concat([self.product_metadata, self.product_metadata])
        result = perform_umap_on_price_data(self.price_data, duplicated_metadata.iloc[:3], self.method_params)

        # Should return a DataFrame with the correct number of rows
        self.assertEqual(result.shape[0], 6, "The result should have 6 rows.")

    def test_missing_values_in_price_data(self):
        """Test that UMAP raises an error when price data contains missing values."""
        missing_value_data = self.price_data.copy()
        missing_value_data.iloc[0, 0] = None  # Introduce a NaN value
        with self.assertRaises(ValueError):
            perform_umap_on_price_data(missing_value_data, self.product_metadata, self.method_params)

    def test_missing_values_in_metadata(self):
        """Test that UMAP can handle missing values in metadata (since they are not part of the transformation)."""
        missing_value_metadata = self.product_metadata.copy()
        missing_value_metadata.iloc[0, 1] = None  # Introduce a NaN in metadata
        result = perform_umap_on_price_data(self.price_data, missing_value_metadata, self.method_params)

        # Should return a DataFrame with the correct number of rows
        self.assertEqual(result.shape[0], 6, "The result should have 6 rows.")
        self.assertTrue(result.iloc[0]['ChainName'] is None, "Missing metadata should be preserved.")

if __name__ == '__main__':
    unittest.main()