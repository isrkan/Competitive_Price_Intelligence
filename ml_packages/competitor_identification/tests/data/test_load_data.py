import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from competitor_identification.data import load_data

class TestLoadData(unittest.TestCase):

    # Test for loading store data
    @patch('competitor_identification.data.load_data.pd.read_csv')
    @patch('competitor_identification.data.load_data.pd.read_excel')
    @patch('competitor_identification.data.load_data.os.path.exists')
    def test_load_store_data_success(self, mock_exists, mock_read_excel, mock_read_csv):
        # Mock file existence
        mock_exists.side_effect = lambda path: True

        # Mock CSV and Excel loading
        mock_read_csv.return_value = pd.DataFrame({
            'ChainID': [1],
            'ChainName': ['Chain A'],
            'SubChainID': [101],
            'SubChainName': ['SubChain X'],
            'StoreID': [1001],
            'StoreName': ['Store A'],
            'DistrictName': ['District 1'],
            'SubDistrictName': ['SubDistrict 1'],
            'CityName': ['City A']
        })
        mock_read_excel.return_value = pd.DataFrame({
            'SubChainID': [101],
            'SubChainName': ['SubChain X'],
            'EnglishName': ['SubChain X English']
        })

        # Call the function
        result = load_data.load_store_data('store_data.csv', 'subchain_data.xlsx')

        # Assertions
        self.assertEqual(result.shape[0], 1)  # One row should be loaded
        self.assertTrue('EnglishName' not in result.columns)  # 'EnglishName' should be dropped
        self.assertEqual(result['SubChainName'][0], 'SubChain X English')  # Name should be replaced


if __name__ == '__main__':
    unittest.main()