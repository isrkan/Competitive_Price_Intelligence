import unittest
import pandas as pd
from competitor_identification.dimensionality_reduction.t_sne import perform_tsne_on_price_data


class TestTSNEFunction(unittest.TestCase):

    def setUp(self):
        """Set up test data for t-SNE."""
        # DataFrame containing price data (P_M_1 and P_L_1 are just placeholders for price features)
        self.price_data = pd.DataFrame({
            'P_M_1': [1.0, 2.0, 3.0],
            'P_L_1': [4.0, 5.0, 6.0]
        })

        # DataFrame containing product metadata
        self.product_metadata = pd.DataFrame({
            'ChainID': [1, 2, 1],
            'ChainName': ['Chain A', 'Chain B', 'Chain A'],
            'SubChainID': [10, 20, 10],
            'SubChainName': ['Sub A1', 'Sub B1', 'Sub A2'],
            'StoreID': [100, 101, 102],
            'StoreName': ['Store A', 'Store B', 'Store C'],
            'DistrictName': ['District 1', 'District 2', 'District 3'],
            'SubDistrictName': ['SubD 1', 'SubD 2', 'SubD 3'],
            'CityName': ['City 1', 'City 2', 'City 3']
        })

        # Parameters for t-SNE
        self.method_params = {'n_components': 2, 'random_state': 42, 'perplexity': 2}

    def test_valid_tsne_transformation(self):
        """Test that t-SNE transformation works with valid data."""
        result = perform_tsne_on_price_data(self.price_data, self.product_metadata, self.method_params)

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame, "The result should be a pandas DataFrame.")

        # Check if the number of columns in the result matches the t-SNE components and metadata columns
        self.assertEqual(result.shape[1], 2 + 9, "The result should have 2 t-SNE components and 9 metadata columns.")

        # Check if the t-SNE transformation worked by checking if the t-SNE component columns exist
        self.assertIn(0, result.columns, "t-SNE component 0 column should be present.")
        self.assertIn(1, result.columns, "t-SNE component 1 column should be present.")

    def test_empty_price_data(self):
        """Test that the function raises an error when price data (df) is empty."""
        empty_price_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            perform_tsne_on_price_data(empty_price_data, self.product_metadata, self.method_params)

    def test_empty_metadata(self):
        """Test that the function raises an error when product metadata is empty."""
        empty_metadata = pd.DataFrame()
        with self.assertRaises(ValueError):
            perform_tsne_on_price_data(self.price_data, empty_metadata, self.method_params)

    def test_invalid_price_data_type(self):
        """Test that the function raises a TypeError when the price data is not a DataFrame."""
        with self.assertRaises(TypeError):
            perform_tsne_on_price_data("invalid_data", self.product_metadata, self.method_params)

    def test_invalid_metadata_type(self):
        """Test that the function raises a TypeError when the metadata is not a DataFrame."""
        with self.assertRaises(TypeError):
            perform_tsne_on_price_data(self.price_data, "invalid_metadata", self.method_params)

    def test_invalid_method_params_type(self):
        """Test that the function raises a TypeError when method_params is not a dictionary."""
        with self.assertRaises(TypeError):
            perform_tsne_on_price_data(self.price_data, self.product_metadata, "invalid_params")

    def test_invalid_method_params_value(self):
        """Test that the function raises an error when t-SNE parameters are invalid (e.g., n_components greater than the number of features)."""
        invalid_params = {'n_components': 5}  # n_components > number of features (we have 2 features)
        with self.assertRaises(ValueError):
            perform_tsne_on_price_data(self.price_data, self.product_metadata, invalid_params)

    def test_check_tsne_components(self):
        """Test that t-SNE components are actually being computed (non-empty values in t-SNE columns)."""
        result = perform_tsne_on_price_data(self.price_data, self.product_metadata, self.method_params)

        # Check that the t-SNE component columns are not empty
        self.assertTrue(result[0].notnull().any(), "t-SNE component 0 should have values.")
        self.assertTrue(result[1].notnull().any(), "t-SNE component 1 should have values.")

    def test_check_column_order_after_merge(self):
        """Test that the correct order of columns is maintained after t-SNE and merge."""
        result = perform_tsne_on_price_data(self.price_data, self.product_metadata, self.method_params)

        # Expected order of columns: t-SNE components first, followed by metadata columns
        expected_column_order = [0, 1, 'ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']

        self.assertEqual(list(result.columns), expected_column_order, "The column order in the result is incorrect.")

    def test_method_params_as_dict(self):
        """Test that method_params should be a dictionary."""
        invalid_params = {'n_components': 'two'}  # Invalid n_components type (should be integer)
        with self.assertRaises(ValueError):
            perform_tsne_on_price_data(self.price_data, self.product_metadata, invalid_params)

    def test_invalid_merge_columns(self):
        """Test that the function raises an error if merge columns are missing in the metadata DataFrame."""
        missing_columns_metadata = self.product_metadata.drop(columns=['StoreID'])
        with self.assertRaises(KeyError):
            perform_tsne_on_price_data(self.price_data, missing_columns_metadata, self.method_params)

if __name__ == '__main__':
    unittest.main()