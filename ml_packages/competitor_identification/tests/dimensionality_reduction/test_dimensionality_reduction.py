import unittest
import pandas as pd
import numpy as np
from competitor_identification.dimensionality_reduction.dimensionality_reduction import perform_dimensionality_reduction_on_price_movement_and_level


class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        # Create mock data for price movement and price level
        self.price_movement_df = pd.DataFrame(
            np.random.rand(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        self.price_level_df = pd.DataFrame(
            np.random.rand(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        # Create mock product metadata
        self.product_df = pd.DataFrame({
            'ChainID': range(10),
            'ChainName': [f'Chain_{i}' for i in range(10)],
            'SubChainID': range(10),
            'SubChainName': [f'SubChain_{i}' for i in range(10)],
            'StoreID': range(10),
            'StoreName': [f'Store_{i}' for i in range(10)],
            'DistrictName': [f'District_{i}' for i in range(10)],
            'SubDistrictName': [f'SubDistrict_{i}' for i in range(10)],
            'CityName': [f'City_{i}' for i in range(10)]
        })

        # PCA parameters
        self.pca_params = {'n_components': 2}
        # t-SNE parameters
        self.tsne_params = {'n_components': 2, 'perplexity': 5, 'random_state': 42}
        # UMAP parameters
        self.umap_params = {'n_components': 2, 'n_neighbors': 3, 'min_dist': 0.1}

    def test_pca_reduction(self):
        combined, movement, level = perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, self.product_df, 'pca', self.pca_params)
        self.assertEqual(combined.shape[0], 10)
        self.assertEqual(movement.shape[0], 10)
        self.assertEqual(level.shape[0], 10)

    def test_tsne_reduction(self):
        combined, movement, level = perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, self.product_df, 't-sne', self.tsne_params)
        self.assertEqual(combined.shape[0], 10)
        self.assertEqual(movement.shape[0], 10)
        self.assertEqual(level.shape[0], 10)

    def test_umap_reduction(self):
        combined, movement, level = perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, self.product_df, 'umap', self.umap_params)
        self.assertEqual(combined.shape[0], 10)
        self.assertEqual(movement.shape[0], 10)
        self.assertEqual(level.shape[0], 10)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, self.product_df, 'invalid_method', {})

    def test_empty_dataframes(self):
        with self.assertRaises(ValueError):
            perform_dimensionality_reduction_on_price_movement_and_level(pd.DataFrame(), self.price_level_df, self.product_df, 'pca', self.pca_params)
        with self.assertRaises(ValueError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, pd.DataFrame(), self.product_df, 'pca', self.pca_params)
        with self.assertRaises(ValueError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, pd.DataFrame(), 'pca', self.pca_params)

    def test_invalid_input_types(self):
        with self.assertRaises(TypeError):
            perform_dimensionality_reduction_on_price_movement_and_level("not a dataframe", self.price_level_df, self.product_df, 'pca', self.pca_params)
        with self.assertRaises(TypeError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, "not a dataframe", self.product_df, 'pca', self.pca_params)
        with self.assertRaises(TypeError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, "not a dataframe", 'pca', self.pca_params)
        with self.assertRaises(TypeError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, self.product_df, 123, self.pca_params)
        with self.assertRaises(TypeError):
            perform_dimensionality_reduction_on_price_movement_and_level(self.price_movement_df, self.price_level_df, self.product_df, 'pca', "not a dict")

if __name__ == '__main__':
    unittest.main()