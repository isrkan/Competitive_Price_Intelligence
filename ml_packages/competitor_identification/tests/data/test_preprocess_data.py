import unittest
from unittest.mock import MagicMock
import pandas as pd
from competitor_identification.data.preprocess_data import get_store_and_chain_ids, filter_nans, filter_by_specific_product_and_add_store_details, filter_by_geographic_region


class TestPreprocessData(unittest.TestCase):
    """
    Test case class for testing the functionality of the preprocess_data functions.
    """

    def setUp(self):
        """
        Set up necessary mock data used across multiple test cases.
        """
        # Mock store data
        self.store_data = pd.DataFrame({
            'StoreID': [1001, 1002, 1003],
            'StoreName': ['StoreA', 'StoreB', 'StoreC'],
            'SubChainName': ['SubChainA', 'SubChainA', 'SubChainB'],
            'CityName': ['CityA', 'CityB', 'CityC'],
            'SubDistrictName': ['SubDistrictA', 'SubDistrictB', 'SubDistrictC'],
            'DistrictName': ['DistrictA', 'DistrictB', 'DistrictC']
        })

        # Mock category data (price data)
        self.category_data = pd.DataFrame({
            'ChainID': [1, 1, 2],
            'ChainName': ['ChainA', 'ChainA', 'ChainB'],
            'SubChainID': [101, 102, 201],
            'SubChainName': ['SubChainA', 'SubChainA', 'SubChainB'],
            'category': ['CategoryA', 'CategoryA', 'CategoryB'],
            'ProductDescription': ['ProductA', 'ProductB', 'ProductC'],
            'StoreID': [1001, 1002, 1003],
            'price': [10.0, 15.0, 20.0],
            'CityName': ['CityA', 'CityB', 'CityC'],
            'SubDistrictName': ['SubDistrictA', 'SubDistrictB', 'SubDistrictC'],
            'DistrictName': ['DistrictA', 'DistrictB', 'DistrictC']
        })

    def test_get_store_and_chain_ids_success(self):
        """
        Test the successful execution of get_store_and_chain_ids when valid data is provided.
        """
        # Call the function
        result = get_store_and_chain_ids(self.store_data, 'SubChainA', 'StoreA')

        # Assert that the result contains the correct store data
        self.assertEqual(result.shape[0], 1)  # Should return 1 row
        self.assertEqual(result['StoreName'].iloc[0], 'StoreA')
        self.assertEqual(result['SubChainName'].iloc[0], 'SubChainA')

    def test_get_store_and_chain_ids_no_match(self):
        """
        Test that get_store_and_chain_ids returns an empty DataFrame when no match is found.
        """
        result = get_store_and_chain_ids(self.store_data, 'SubChainB', 'StoreD')
        self.assertEqual(result.shape[0], 0)  # No rows should be returned

    def test_get_store_and_chain_ids_type_error(self):
        """
        Test that get_store_and_chain_ids raises a TypeError when invalid data types are passed.
        """
        with self.assertRaises(TypeError):
            get_store_and_chain_ids(None, 'SubChainA', 'StoreA')  # Invalid Store_data type

        with self.assertRaises(TypeError):
            get_store_and_chain_ids(self.store_data, 123, 'StoreA')  # Invalid SubChainName type

        with self.assertRaises(TypeError):
            get_store_and_chain_ids(self.store_data, 'SubChainA', 123)  # Invalid StoreName type

    def test_filter_nans_success(self):
        """
        Test the successful execution of filter_nans function when valid data is provided.
        """
        # Test with a threshold of 0.5 (50% NaN tolerance)
        self.category_data.iloc[1, 3] = None  # Introduce a NaN value
        result = filter_nans(self.category_data, 0.5)

        # Assert that the row with the NaN value was dropped
        self.assertEqual(result.shape[0], 3)  # All 3 rows should remain

    def test_filter_nans_no_filtering(self):
        """
        Test the filter_nans function when there are no NaNs in the data.
        """
        result = filter_nans(self.category_data, 0.5)

        # Assert that no rows are removed
        self.assertEqual(result.shape[0], 3)  # Should return 3 rows


    def test_filter_nans_invalid_data_type(self):
        """
        Test that filter_nans raises a TypeError when invalid data types are passed.
        """
        with self.assertRaises(TypeError):
            filter_nans(None, 0.5)  # Invalid category_df type

    def test_filter_nans_invalid_threshold(self):
        """
        Test that filter_nans raises a ValueError when an invalid threshold is passed.
        """
        with self.assertRaises(ValueError):
            filter_nans(self.category_data, -0.1)  # Invalid threshold

        with self.assertRaises(ValueError):
            filter_nans(self.category_data, 1.5)  # Invalid threshold

    def test_filter_by_specific_product_and_add_store_details_success(self):
        """
        Test the successful execution of filter_by_specific_product_and_add_store_details.
        """
        result = filter_by_specific_product_and_add_store_details(self.category_data, 'ProductA', self.store_data)

        # Assert that the product data is correctly filtered and merged with store data
        self.assertEqual(result.shape[0], 1)  # Should return 1 row for ProductA
        self.assertEqual(result['ProductDescription'].iloc[0], 'ProductA')
        self.assertEqual(result['StoreName'].iloc[0], 'StoreA')

    def test_filter_by_specific_product_and_add_store_details_no_match(self):
        """
        Test that filter_by_specific_product_and_add_store_details raises an error when no product matches.
        """
        with self.assertRaises(ValueError):
            filter_by_specific_product_and_add_store_details(self.category_data, 'NonExistentProduct', self.store_data)

    def test_filter_by_specific_product_and_add_store_details_empty_dataframe(self):
        """
        Test that filter_by_specific_product_and_add_store_details raises an error when the input DataFrame is empty.
        """
        empty_df = pd.DataFrame(columns=self.category_data.columns)
        with self.assertRaises(ValueError):
            filter_by_specific_product_and_add_store_details(empty_df, 'ProductA', self.store_data)

    def test_filter_by_specific_product_and_add_store_details_type_error(self):
        """
        Test that filter_by_specific_product_and_add_store_details raises a TypeError for invalid input types.
        """
        with self.assertRaises(TypeError):
            filter_by_specific_product_and_add_store_details(None, 'ProductA', self.store_data)

        with self.assertRaises(TypeError):
            filter_by_specific_product_and_add_store_details(self.category_data, 123, self.store_data)

        with self.assertRaises(TypeError):
            filter_by_specific_product_and_add_store_details(self.category_data, 'ProductA', None)

    def test_filter_by_specific_product_and_add_store_details_invalid_store_data(self):
        """
        Test filter_by_specific_product_and_add_store_details with invalid store data (no 'StoreID' column).
        """
        invalid_store_data = pd.DataFrame({
            'StoreName': ['StoreA', 'StoreB'],
            'SubChainName': ['SubChainA', 'SubChainB']
        })

        with self.assertRaises(KeyError):
            filter_by_specific_product_and_add_store_details(self.category_data, 'ProductA', invalid_store_data)


if __name__ == "__main__":
    unittest.main()