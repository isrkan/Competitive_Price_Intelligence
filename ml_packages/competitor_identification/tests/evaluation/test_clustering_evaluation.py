import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from competitor_identification.evaluation.clustering_evaluation import evaluate_clustering


class TestEvaluateClustering(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for use in various tests.
        """
        # Creating a sample DataFrame for testing
        self.dr_components_df = pd.DataFrame({
            'P_M_1': np.random.rand(10),
            'P_M_2': np.random.rand(10),
            'P_L_1': np.random.rand(10),
            'P_L_2': np.random.rand(10)
        })

        # Performing KMeans clustering as an example of cluster_labels
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']])

    def test_valid_clustering(self):
        """
        Test the function with valid cluster labels and data.
        """
        silhouette, davies_bouldin = evaluate_clustering(self.cluster_labels, self.dr_components_df)

        # Check that the silhouette and Davies-Bouldin scores are numbers
        self.assertIsInstance(silhouette, float)
        self.assertIsInstance(davies_bouldin, float)

        # Silhouette score should be between -1 and 1
        self.assertGreaterEqual(silhouette, -1)
        self.assertLessEqual(silhouette, 1)

        # Davies-Bouldin score should be >= 0
        self.assertGreaterEqual(davies_bouldin, 0)

    def test_mismatched_labels_and_data(self):
        """
        Test the function with mismatched number of cluster labels and rows in the data.
        """
        # Create a mismatched cluster_labels list
        mismatched_labels = np.array([0, 1, 2])  # Only 3 labels, but 10 rows in the dataframe

        with self.assertRaises(ValueError) as context:
            evaluate_clustering(mismatched_labels, self.dr_components_df)

        # Check if ValueError was raised with appropriate message
        self.assertEqual(str(context.exception),
                         "The number of cluster labels must match the number of rows in dr_components_df.")

    def test_empty_dataframe(self):
        """
        Test the function with an empty dataframe.
        """
        empty_df = pd.DataFrame(columns=['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2'])
        empty_labels = np.array([])  # No labels since the dataframe is empty

        # Expecting silhouette and davies_bouldin to return NaN or handle empty case gracefully
        silhouette, davies_bouldin = evaluate_clustering(empty_labels, empty_df)

        self.assertTrue(np.isnan(silhouette))
        self.assertTrue(np.isnan(davies_bouldin))

    def test_single_cluster(self):
        """
        Test the function when all points belong to a single cluster.
        """
        # Create a cluster_labels array where all data points belong to the same cluster (label 0)
        single_cluster_labels = np.zeros(self.dr_components_df.shape[0])

        with self.assertRaises(ValueError) as context:
            evaluate_clustering(single_cluster_labels, self.dr_components_df)

        # Check that the exception is raised because only one cluster is present
        self.assertTrue('Valid values are 2 to n_samples - 1' in str(context.exception))

    def test_inconsistent_data_columns(self):
        """
        Test the function with inconsistent data columns (missing P_M_1, P_M_2, etc.).
        """
        # Creating a DataFrame with missing some required columns
        invalid_df = pd.DataFrame({
            'P_M_1': np.random.rand(10),
            'P_M_2': np.random.rand(10),
        })

        with self.assertRaises(KeyError) as context:
            evaluate_clustering(self.cluster_labels, invalid_df)

        # Check if KeyError is raised due to missing columns
        self.assertTrue('P_L_1' in str(context.exception))
        self.assertTrue('P_L_2' in str(context.exception))

    def test_invalid_labels_type(self):
        """
        Test the function with invalid type for cluster_labels (e.g., passing a string instead of a list/array).
        """
        invalid_labels = "invalid_labels"  # Passing a string instead of an array/list

        with self.assertRaises(ValueError) as context:
            evaluate_clustering(invalid_labels, self.dr_components_df)

        # Check if ValueError is raised with the correct message
        self.assertIn('The number of cluster labels must match the number of rows in dr_components_df', str(context.exception))

    def test_missing_column_in_dataframe(self):
        """
        Test the function when required columns are missing from the DataFrame.
        """
        # Create a DataFrame missing the 'P_M_1' column
        df_missing_column = self.dr_components_df.drop(columns=['P_M_1'])

        with self.assertRaises(KeyError) as context:
            evaluate_clustering(self.cluster_labels, df_missing_column)

        # Ensure that the error message mentions the missing column
        self.assertTrue('P_M_1' in str(context.exception))

    def test_correct_values_for_silhouette_and_davies_bouldin(self):
        """
        Test that the silhouette and Davies-Bouldin scores return reasonable values for a known dataset.
        """
        # Perform clustering with known method (KMeans)
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(self.dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']])

        # Using sklearn's built-in metrics to validate results
        expected_silhouette = silhouette_score(self.dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']],
                                               cluster_labels)
        expected_davies_bouldin = davies_bouldin_score(self.dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']],
                                                       cluster_labels)

        silhouette, davies_bouldin = evaluate_clustering(cluster_labels, self.dr_components_df)

        # Check if the results match the expected values
        self.assertAlmostEqual(silhouette, expected_silhouette, delta=0.1)
        self.assertAlmostEqual(davies_bouldin, expected_davies_bouldin, delta=0.1)


if __name__ == '__main__':
    unittest.main()