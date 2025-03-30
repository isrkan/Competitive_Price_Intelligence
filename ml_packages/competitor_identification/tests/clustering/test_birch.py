import unittest
import pandas as pd
import numpy as np
from competitor_identification.clustering.birch import birch_clustering


class TestBirchClustering(unittest.TestCase):

    def setUp(self):
        """
        Set up the data and parameters for testing the birch_clustering function.
        """
        # Create mock DataFrame with 10 samples and 5 features for dimensionality-reduced data
        self.dr_components_df = pd.DataFrame(
            np.random.rand(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        # Define method params for Birch clustering
        self.method_params = {
            'n_clusters': 3,  # Number of clusters
            'threshold': 0.5  # Distance threshold for merging clusters
        }

        # Define alternative method params for testing different configurations
        self.method_params_large_clusters = {
            'n_clusters': 2,
            'threshold': 1.0
        }

    def test_birch_clustering_success(self):
        """
        Test that Birch clustering runs successfully and updates the DataFrame with cluster labels.
        """
        # Run Birch clustering
        updated_df, cluster_labels = birch_clustering(self.dr_components_df, self.method_params)

        # Check that the DataFrame has the correct number of rows
        self.assertEqual(updated_df.shape[0], 10)

        # Ensure the DataFrame has the 'dr_cluster_labels' column with the correct number of rows
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df['dr_cluster_labels'].shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))

    def test_invalid_input_df(self):
        """
        Test that the function raises a TypeError if the input DataFrame is not of type pandas DataFrame.
        """
        with self.assertRaises(TypeError):
            birch_clustering("not a dataframe", self.method_params)

    def test_invalid_method_params(self):
        """
        Test that the function raises a TypeError if the method_params is not a dictionary.
        """
        with self.assertRaises(TypeError):
            birch_clustering(self.dr_components_df, "not a dictionary")

    def test_empty_dataframe(self):
        """
        Test that the function raises an error when an empty DataFrame is provided.
        """
        empty_df = pd.DataFrame()

        with self.assertRaises(ValueError):
            birch_clustering(empty_df, self.method_params)

    def test_birch_no_clusters(self):
        """
        Test Birch clustering with an invalid n_clusters value (e.g., 0 clusters).
        """
        invalid_params = {
            'n_clusters': 0,  # Invalid as clustering requires at least 1 cluster
            'threshold': 0.5
        }

        with self.assertRaises(ValueError):
            birch_clustering(self.dr_components_df, invalid_params)

    def test_birch_with_no_threshold(self):
        """
        Test Birch clustering when the threshold is not specified (ensure it defaults correctly).
        """
        # Provide Birch method params without threshold, this should be valid
        params_without_threshold = {
            'n_clusters': 3  # Only n_clusters is provided
        }

        updated_df, cluster_labels = birch_clustering(self.dr_components_df, params_without_threshold)

        # Ensure the DataFrame has the 'dr_cluster_labels' column
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df['dr_cluster_labels'].shape[0], 10)

    def test_birch_with_invalid_threshold(self):
        """
        Test that the function handles an invalid threshold parameter (e.g., negative value).
        """
        invalid_params = {
            'n_clusters': 3,
            'threshold': -0.5  # Invalid threshold value
        }

        with self.assertRaises(ValueError):
            birch_clustering(self.dr_components_df, invalid_params)

    def test_birch_with_non_numeric_data(self):
        """
        Test that the function raises an error if the DataFrame contains non-numeric data.
        """
        # Create a DataFrame with mixed numeric and string data
        non_numeric_df = self.dr_components_df.copy()
        non_numeric_df['feature_0'] = 'string_data'

        with self.assertRaises(ValueError):
            birch_clustering(non_numeric_df, self.method_params)

    def test_birch_clustering_edge_case_single_row(self):
        """
        Test Birch clustering with a single row of data (edge case).
        """
        single_row_df = pd.DataFrame(np.random.rand(1, 5), columns=[f'feature_{i}' for i in range(5)])

        updated_df, cluster_labels = birch_clustering(single_row_df, self.method_params)

        # Ensure the single row is assigned a valid cluster label
        self.assertEqual(updated_df.shape[0], 1)
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(len(np.unique(cluster_labels)), 1)  # Only one cluster should be assigned


if __name__ == '__main__':
    unittest.main()