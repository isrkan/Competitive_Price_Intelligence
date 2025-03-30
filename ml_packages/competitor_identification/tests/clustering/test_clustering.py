import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from competitor_identification.clustering.clustering import perform_clustering, find_top_competitors


class TestClustering(unittest.TestCase):

    def setUp(self):
        """Set up test data for clustering functions."""
        self.dr_components_df = pd.DataFrame({
            'StoreID': [1, 2, 3, 4],
            'SubChainID': [101, 102, 103, 104],
            'ChainID': [201, 202, 203, 204],  # Added missing ChainID
            'ChainName': ["Chain A", "Chain B", "Chain C", "Chain D"],  # Added missing ChainName
            'SubChainName': ["Sub A", "Sub B", "Sub C", "Sub D"],  # Added missing SubChainName
            'StoreName': ["Store 1", "Store 2", "Store 3", "Store 4"],  # Added missing StoreName
            'P_M_1': [1.0, 2.0, 3.0, 4.0],
            'P_M_2': [1.1, 2.1, 3.1, 4.1],
            'P_L_1': [1.2, 2.2, 3.2, 4.2],
            'P_L_2': [1.3, 2.3, 3.3, 4.3]
        })

        self.chosen_store_info = pd.DataFrame({'StoreID': [1], 'SubChainID': [101]})
        self.same_chain_dr_components_df = pd.DataFrame({'StoreID': [5], 'SubChainID': [101]})
        self.method_params = {}

    @patch("competitor_identification.clustering.birch_clustering")
    def test_perform_clustering_birch(self, mock_birch):
        """Test perform_clustering using the Birch method."""
        mock_birch.return_value = (self.dr_components_df.assign(dr_cluster_labels=[0, 1, 0, 1]), np.array([0, 1, 0, 1]))

        clustered_dr_components_df, cluster_labels, store_df, competitors_df = perform_clustering(
            self.dr_components_df, self.chosen_store_info, self.same_chain_dr_components_df, "birch", self.method_params
        )

        self.assertIsInstance(clustered_dr_components_df, pd.DataFrame)
        self.assertEqual(len(clustered_dr_components_df), 5)  # Including same-chain store
        self.assertIsInstance(cluster_labels, np.ndarray)
        self.assertIsInstance(store_df, pd.DataFrame)
        self.assertIsInstance(competitors_df, pd.DataFrame)
        self.assertTrue("dr_cluster_labels" in clustered_dr_components_df.columns)

    @patch("competitor_identification.clustering.dbscan_clustering")
    def test_perform_clustering_dbscan(self, mock_dbscan):
        """Test perform_clustering using the DBSCAN method."""
        mock_dbscan.return_value = (
        self.dr_components_df.assign(dr_cluster_labels=[0, 0, 1, 1]), np.array([0, 0, 1, 1]))

        clustered_dr_components_df, cluster_labels, store_df, competitors_df = perform_clustering(
            self.dr_components_df, self.chosen_store_info, self.same_chain_dr_components_df, "dbscan",
            self.method_params
        )

        self.assertEqual(len(clustered_dr_components_df), 5)  # Including same-chain store
        self.assertEqual(cluster_labels[-1], max(cluster_labels))

    def test_perform_clustering_invalid_method(self):
        """Test perform_clustering with an invalid method."""
        with self.assertRaises(ValueError):
            perform_clustering(self.dr_components_df, self.chosen_store_info, self.same_chain_dr_components_df,
                               "invalid", self.method_params)

    def test_find_top_competitors(self):
        """Test find_top_competitors function."""
        clustered_store_df = self.dr_components_df.iloc[:1]  # Store 1
        clustered_competitors_df = self.dr_components_df.iloc[1:].copy()

        top_competitors = find_top_competitors(clustered_store_df, clustered_competitors_df, top_n=2)

        self.assertEqual(len(top_competitors), 2)
        self.assertTrue("StoreID" in top_competitors.columns)
        self.assertTrue(top_competitors.iloc[0]["StoreID"] in [2, 3, 4])

    def test_find_top_competitors_empty(self):
        """Test find_top_competitors when no competitors exist."""
        clustered_store_df = self.dr_components_df.iloc[:1]  # Store 1
        clustered_competitors_df = pd.DataFrame(columns=self.dr_components_df.columns)  # Empty competitors

        top_competitors = find_top_competitors(clustered_store_df, clustered_competitors_df, top_n=2)
        self.assertTrue(top_competitors.empty)

    def test_find_top_competitors_invalid_top_n(self):
        """Test find_top_competitors with an invalid top_n value."""
        clustered_store_df = self.dr_components_df.iloc[:1]
        clustered_competitors_df = self.dr_components_df.iloc[1:].copy()

        with self.assertRaises(ValueError):
            find_top_competitors(clustered_store_df, clustered_competitors_df, top_n=0)


if __name__ == "__main__":
    unittest.main()