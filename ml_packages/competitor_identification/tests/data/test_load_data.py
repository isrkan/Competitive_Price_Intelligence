import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from competitor_identification.data.load_data import load_store_data, load_price_data


class TestLoadData(unittest.TestCase):
    """
    Test case class for testing the functionality of `load_store_data` and `load_price_data` functions.
    """

    def setUp(self):
        """
        Set up any necessary mock values that are used across multiple test cases.
        This method runs before each individual test case.
        'Mock' refers to simulated objects or functions that replace real ones during testing.
        These mock objects simulate the behavior of real objects in a controlled way.
        """
        # Paths to files that will be mocked in the tests
        self.store_file_path = "mock_store.csv"
        self.subchain_file_path = "mock_subchain.xlsx"
        self.price_data_dir = "mock_dir"
        self.category = 'CategoryA'

        # Mocked data for testing: Mocked data is predefined data used to simulate the real data
        self.mock_store_data = pd.DataFrame({
            'ChainID': [1],
            'ChainName': ['ChainA'],
            'SubChainID': [101],
            'SubChainName': ['SubChainA'],
            'StoreID': [1001],
            'StoreName': ['StoreA'],
            'DistrictName': ['DistrictA'],
            'SubDistrictName': ['SubDistrictA'],
            'CityName': ['CityA']
        })

        self.mock_subchain_data = pd.DataFrame({
            'SubChainID': [101],
            'SubChainName': ['SubChainA'],
            'EnglishName': ['SubChainA_English']
        })

        self.mock_price_data = pd.DataFrame({
            'category': ['CategoryA'],
            'ProductDescription': ['ProductA'],
            'StoreID': [1001],
            'price': [10.0]
        })

    def patch_helpers(self, mock_exists, mock_read_csv, mock_read_excel, mock_read_parquet):
        """
        Helper function to patch the necessary methods used in the tests.
        This prevents repetition of the patching logic across test cases.

        'Patching' refers to replacing a function or method with a mock version for the scope of the test.
        This is useful when we don't want to use the actual implementation, but instead want to simulate behavior.

        Parameters:
        - mock_exists: Mock for os.path.exists to simulate file existence.
        - mock_read_csv: Mock for pd.read_csv to simulate reading CSV files.
        - mock_read_excel: Mock for pd.read_excel to simulate reading Excel files.
        - mock_read_parquet: Mock for pd.read_parquet to simulate reading Parquet files.
        """
        # Define the file existence checks for the mock
        # The `mock_exists` is patched to simulate the behavior of `os.path.exists`
        mock_exists.side_effect = lambda path: path in [self.store_file_path, self.subchain_file_path]
        # Define the mock data for the CSV and Excel files
        # `mock_read_csv` and `mock_read_excel` are patched to simulate the reading of CSV and Excel files
        mock_read_csv.return_value = self.mock_store_data
        mock_read_excel.return_value = self.mock_subchain_data
        # Mock the Parquet reading function
        # `mock_read_parquet` simulates loading a Parquet file
        mock_read_parquet.return_value = self.mock_price_data

    @patch("competitor_identification.data.load_data.pd.read_csv")
    @patch("competitor_identification.data.load_data.pd.read_excel")
    @patch("competitor_identification.data.load_data.os.path.exists")
    def test_load_store_data_success(self, mock_exists, mock_read_excel, mock_read_csv):
        """
        Test the successful execution of the `load_store_data` function when the necessary files exist and are correctly formatted.

        In this test, we are 'mocking' the behavior of the file system (os.path.exists) and the reading functions (read_csv, read_excel) using mocked data (the predefined pandas DataFrames).
        We simulate that the files exist and contain the expected content without actually needing the files to exist.
        """
        # Apply the helper function to set up all necessary patches
        self.patch_helpers(mock_exists, mock_read_csv, mock_read_excel, MagicMock())
        # Call the function to test
        result = load_store_data(self.store_file_path, self.subchain_file_path)

        # Assert that the merged SubChainName column is correctly populated
        self.assertEqual(result['SubChainName'][0], 'SubChainA_English')
        # Ensure the mock functions were called with the correct arguments
        mock_read_csv.assert_called_once_with(self.store_file_path)
        mock_read_excel.assert_called_once_with(self.subchain_file_path)

    @patch("competitor_identification.data.load_data.pd.read_csv")
    @patch("competitor_identification.data.load_data.pd.read_excel")
    @patch("competitor_identification.data.load_data.os.path.exists")
    def test_load_store_data_missing_file(self, mock_exists, mock_read_excel, mock_read_csv):
        """
        Test that `load_store_data` raises a FileNotFoundError when one of the required files is missing.

        This test uses 'mocking' to simulate the scenario where one of the files is missing.
        We mock `os.path.exists` to return False for one of the paths, simulating a missing file.
        """
        # Simulate that only the store file exists
        mock_exists.side_effect = lambda path: path == self.store_file_path

        # Call the function and check that the correct exception is raised
        with self.assertRaises(FileNotFoundError):
            load_store_data(self.store_file_path, self.subchain_file_path)

    @patch("competitor_identification.data.load_data.pd.read_csv")
    @patch("competitor_identification.data.load_data.pd.read_excel")
    @patch("competitor_identification.data.load_data.os.path.exists")
    def test_load_store_data_missing_columns(self, mock_exists, mock_read_excel, mock_read_csv):
        """
        Test that `load_store_data` raises a ValueError when required columns are missing from the store data.
        """
        # Simulate missing 'SubChainName' column in the store data
        mock_exists.side_effect = lambda path: path in [self.store_file_path, self.subchain_file_path]
        # Mock data with missing required columns in the store data (e.g., missing 'SubChainName')
        store_data_mock = pd.DataFrame({
            'ChainID': [1],
            'ChainName': ['ChainA'],
            'SubChainID': [101],
            'StoreID': [1001],
            'StoreName': ['StoreA'],
            'DistrictName': ['DistrictA'],
            'SubDistrictName': ['SubDistrictA'],
            'CityName': ['CityA']
        })
        mock_read_csv.return_value = store_data_mock
        mock_read_excel.return_value = self.mock_subchain_data

        # Call the function and check that the correct exception is raised
        with self.assertRaises(ValueError):
            load_store_data(self.store_file_path, self.subchain_file_path)

    @patch("competitor_identification.data.load_data.pd.read_parquet")
    @patch("competitor_identification.data.load_data.os.path.exists")
    def test_load_price_data_missing_file(self, mock_exists, mock_read_parquet):
        """
        Test that `load_price_data` raises a FileNotFoundError when the required Parquet file is missing.
        """
        # Simulate missing file scenario
        mock_exists.side_effect = lambda path: False

        # Call the function and check that the correct exception is raised
        with self.assertRaises(FileNotFoundError):
            load_price_data(self.category, self.price_data_dir)

    @patch("competitor_identification.data.load_data.pd.read_parquet")
    @patch("competitor_identification.data.load_data.os.path.exists")
    def test_load_price_data_invalid_parquet(self, mock_exists, mock_read_parquet):
        """
        Test that `load_price_data` raises a ValueError when there is an issue reading the Parquet file (e.g., invalid format).
        """
        # Simulate valid file path but invalid Parquet file format
        mock_exists.side_effect = lambda path: path == os.path.join(self.price_data_dir, f"{self.category}.parquet")
        mock_read_parquet.side_effect = ValueError("Invalid Parquet file")

        # Call the function and check that the correct exception is raised
        with self.assertRaises(ValueError):
            load_price_data(self.category, self.price_data_dir)

    @patch("competitor_identification.data.load_data.pd.read_parquet")
    @patch("competitor_identification.data.load_data.os.path.exists")
    def test_load_price_data_success(self, mock_exists, mock_read_parquet):
        """
        Test the successful execution of the `load_price_data` function when the Parquet file exists and is correctly formatted.
        """
        # Simulate valid file existence and mock the Parquet data
        mock_exists.side_effect = lambda path: path == os.path.join(self.price_data_dir, f"{self.category}.parquet")
        mock_read_parquet.return_value = self.mock_price_data

        # Call the function to test
        result = load_price_data(self.category, self.price_data_dir)

        # Assert that the returned dataframe has the correct index set
        self.assertEqual(result.index.names, ['category', 'ProductDescription', 'StoreID'])
        # Ensure that the correct file path was used for the Parquet read
        mock_read_parquet.assert_called_once_with(os.path.join(self.price_data_dir, f"{self.category}.parquet"))


if __name__ == "__main__":
    unittest.main()