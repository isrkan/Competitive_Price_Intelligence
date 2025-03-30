import unittest
import pandas as pd
import numpy as np
from competitor_identification.clustering.dbscan import find_optimal_eps, dbscan_clustering


class TestDBSCANClustering(unittest.TestCase):

    def setUp(self):
        """
        Set up the data and parameters for testing the dbscan_clustering function.
        """
        # Create mock DataFrame with 10 samples and 5 features for dimensionality-reduced data
        self.dr_components_df = pd.DataFrame(
            np.random.rand(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        # Define method params for DBSCAN clustering
        self.method_params = {
            'eps': 0.3,  # Epsilon parameter for DBSCAN
            'min_samples': 3  # Minimum number of samples per cluster
        }

        # Define alternative method params for testing different configurations
        self.method_params_no_eps = {
            'min_samples': 3  # Missing eps, should trigger the calculation of eps using elbow method
        }

        self.method_params_default = {
            'eps': 0.5,  # Default eps for DBSCAN
            'min_samples': 5  # Minimum samples for DBSCAN
        }

    def test_find_optimal_eps_success(self):
        """
        Test the successful execution of the find_optimal_eps function.
        """
        distances, knee_point = find_optimal_eps(self.dr_components_df, n_neighbors=3)

        # Ensure distances is a numpy array
        self.assertIsInstance(distances, np.ndarray)

        # Ensure knee_point is an integer (index of the knee point)
        self.assertIsInstance(knee_point, np.integer)

        # Ensure the knee point is within the valid range of distances
        self.assertGreater(knee_point, 0)
        self.assertLess(knee_point, len(distances))

    def test_dbscan_clustering_success(self):
        """
        Test DBSCAN clustering when eps is provided in method_params.
        """
        updated_df, cluster_labels = dbscan_clustering(self.dr_components_df, self.method_params)

        # Check if the DataFrame has the correct number of rows
        self.assertEqual(updated_df.shape[0], 10)

        # Ensure the DataFrame has the 'dr_cluster_labels' column
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df['dr_cluster_labels'].shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))

    def test_dbscan_clustering_without_eps(self):
        """
        Test DBSCAN clustering when eps is not provided, triggering automatic calculation.
        """
        updated_df, cluster_labels = dbscan_clustering(self.dr_components_df, self.method_params_no_eps)

        # Check if the DataFrame has the correct number of rows
        self.assertEqual(updated_df.shape[0], 10)

        # Ensure the DataFrame has the 'dr_cluster_labels' column
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df['dr_cluster_labels'].shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))

    def test_dbscan_invalid_method_params(self):
        """
        Test that the function raises a TypeError if the method_params is not a dictionary.
        """
        with self.assertRaises(TypeError):
            dbscan_clustering(self.dr_components_df, "not_a_dictionary")

    def test_dbscan_invalid_eps_value(self):
        """
        Test that the function raises an error if an invalid eps value is passed.
        """
        invalid_params = {
            'eps': -0.1,  # Invalid eps value (eps must be greater than 0)
            'min_samples': 3
        }

        with self.assertRaises(ValueError):
            dbscan_clustering(self.dr_components_df, invalid_params)

    def test_dbscan_empty_dataframe(self):
        """
        Test that the function raises an error when an empty DataFrame is provided.
        """
        empty_df = pd.DataFrame()

        with self.assertRaises(ValueError):
            dbscan_clustering(empty_df, self.method_params)

    def test_dbscan_no_clusters_found(self):
        """
        Test DBSCAN clustering when no clusters are found.
        """
        no_cluster_df = pd.DataFrame(np.random.rand(10, 5), columns=[f'feature_{i}' for i in range(5)])

        updated_df, cluster_labels = dbscan_clustering(no_cluster_df, self.method_params)

        # Ensure that the labels are -1 (indicating noise for DBSCAN)
        self.assertTrue(np.all(cluster_labels == -1))

    def test_dbscan_edge_case_single_row(self):
        """
        Test DBSCAN clustering with a single row of data (edge case).
        """
        single_row_df = pd.DataFrame(np.random.rand(1, 5), columns=[f'feature_{i}' for i in range(5)])

        updated_df, cluster_labels = dbscan_clustering(single_row_df, self.method_params)

        # Ensure the single row is assigned a valid cluster label
        self.assertEqual(updated_df.shape[0], 1)
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(len(np.unique(cluster_labels)), 1)  # Only one cluster should be assigned

    def test_dbscan_with_missing_eps_in_params(self):
        """
        Test DBSCAN clustering with missing 'eps' in params and check if it uses the elbow method to compute it.
        """
        updated_df, cluster_labels = dbscan_clustering(self.dr_components_df, self.method_params_no_eps)

        # Ensure that the function completed successfully and returned labels
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df.shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))


if __name__ == '__main__':
    unittest.main()