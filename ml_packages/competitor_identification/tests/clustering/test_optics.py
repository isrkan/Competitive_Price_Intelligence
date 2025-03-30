import unittest
import pandas as pd
import numpy as np
from competitor_identification.clustering.optics import optics_clustering


class TestOPTICSClustering(unittest.TestCase):

    def setUp(self):
        """
        Set up the data and parameters for testing the optics_clustering function.
        """
        # Create mock DataFrame with 10 samples and 5 features for dimensionality-reduced data
        self.dr_components_df = pd.DataFrame(
            np.random.rand(10, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )

        # Define method params for OPTICS clustering
        self.method_params = {
            'min_samples': 3,  # Minimum number of samples per cluster
            'xi': 0.05,  # The steepness parameter for OPTICS
            'min_cluster_size': 2  # Minimum size of a cluster
        }

        # Define alternative method params for testing different configurations
        self.method_params_no_min_samples = {
            'xi': 0.05,  # The steepness parameter for OPTICS
            'min_cluster_size': 2  # Minimum size of a cluster
        }

        self.method_params_default = {
            'min_samples': 5,  # Minimum samples for OPTICS
            'xi': 0.05,  # Default xi for OPTICS
            'min_cluster_size': 2  # Minimum cluster size for OPTICS
        }

    def test_optics_clustering_success(self):
        """
        Test OPTICS clustering with valid parameters.
        """
        updated_df, cluster_labels = optics_clustering(self.dr_components_df, self.method_params)

        # Check if the DataFrame has the correct number of rows
        self.assertEqual(updated_df.shape[0], 10)

        # Ensure the DataFrame has the 'dr_cluster_labels' column
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df['dr_cluster_labels'].shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))

    def test_optics_clustering_without_min_samples(self):
        """
        Test OPTICS clustering when 'min_samples' is missing in method_params.
        """
        updated_df, cluster_labels = optics_clustering(self.dr_components_df, self.method_params_no_min_samples)

        # Check if the DataFrame has the correct number of rows
        self.assertEqual(updated_df.shape[0], 10)

        # Ensure the DataFrame has the 'dr_cluster_labels' column
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df['dr_cluster_labels'].shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))

    def test_optics_invalid_input_df(self):
        """
        Test that the function raises a TypeError if the input DataFrame is not of type pandas DataFrame.
        """
        with self.assertRaises(TypeError):
            optics_clustering("not_a_dataframe", self.method_params)

    def test_optics_invalid_method_params(self):
        """
        Test that the function raises a TypeError if the method_params is not a dictionary.
        """
        with self.assertRaises(TypeError):
            optics_clustering(self.dr_components_df, "not_a_dictionary")

    def test_optics_invalid_param_value(self):
        """
        Test that the function raises an error if an invalid parameter value is passed.
        """
        invalid_params = {
            'min_samples': -1,  # Invalid 'min_samples' value (it should be positive)
            'xi': 0.05,  # The steepness parameter for OPTICS
            'min_cluster_size': 2  # Minimum size of a cluster
        }

        with self.assertRaises(ValueError):
            optics_clustering(self.dr_components_df, invalid_params)

    def test_optics_empty_dataframe(self):
        """
        Test that the function raises an error when an empty DataFrame is provided.
        """
        empty_df = pd.DataFrame()

        with self.assertRaises(ValueError):
            optics_clustering(empty_df, self.method_params)

    def test_optics_with_missing_min_samples_in_params(self):
        """
        Test OPTICS clustering with missing 'min_samples' in params and check if it works.
        """
        updated_df, cluster_labels = optics_clustering(self.dr_components_df, self.method_params_no_min_samples)

        # Ensure that the function completed successfully and returned labels
        self.assertIn('dr_cluster_labels', updated_df.columns)
        self.assertEqual(updated_df.shape[0], 10)

        # Check that the cluster labels are integers
        self.assertTrue(np.issubdtype(cluster_labels.dtype, np.integer))


if __name__ == '__main__':
    unittest.main()