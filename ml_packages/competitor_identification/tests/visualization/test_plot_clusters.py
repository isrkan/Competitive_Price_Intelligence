import unittest
import pandas as pd
import numpy as np
from plotly.graph_objects import Figure
import plotly.graph_objects as go
from competitor_identification.visualization.plot_clusters import plot_clusters


class TestPlotClusters(unittest.TestCase):

    def setUp(self):
        """Set up test data for clustering."""
        # DataFrame with the dimensionality reduction components and clustering labels
        self.dr_components_df = pd.DataFrame({
            'StoreID': [1, 2, 3, 4, 5],
            'P_M_1': [0.5, -0.2, 0.3, -0.5, 0.0],
            'P_L_1': [1.2, -0.7, 0.5, -1.0, 0.3],
            'ChainName': ['Chain A', 'Chain B', 'Chain A', 'Chain C', 'Chain B'],
            'SubChainName': ['Sub A1', 'Sub B1', 'Sub A1', 'Sub C1', 'Sub B2'],
            'StoreName': ['Store X', 'Store Y', 'Store Z', 'Store W', 'Store V'],
            'dr_cluster_labels': [0, 1, 0, 2, 1]
        })

        # Store's dimensionality reduction components
        self.store_dr_components_df = pd.DataFrame({
            'StoreID': [3],
            'P_M_1': [0.3],
            'P_L_1': [0.5],
            'ChainName': ['Chain A'],
            'SubChainName': ['Sub A1'],
            'StoreName': ['Store Z'],
            'dr_cluster_labels': [0]
        })

        # Cluster labels (matching the dr_cluster_labels in the components df)
        self.dr_cluster_labels = np.array([0, 1, 0, 2, 1])

        # Clustering method
        self.clustering_method = "KMeans"

    def test_output_type(self):
        """Test that the function returns a Plotly Figure."""
        fig = plot_clusters(self.dr_components_df, self.dr_cluster_labels, self.store_dr_components_df,
                            self.clustering_method)
        self.assertIsInstance(fig, Figure, "Function should return a Plotly Figure.")

    def test_invalid_dr_components_df(self):
        """Test that the function raises a TypeError when dr_components_df is not a DataFrame."""
        with self.assertRaises(TypeError):
            plot_clusters("not_a_dataframe", self.dr_cluster_labels, self.store_dr_components_df,
                          self.clustering_method)

    def test_invalid_store_dr_components_df(self):
        """Test that the function raises a TypeError when store_dr_components_df is not a DataFrame."""
        with self.assertRaises(TypeError):
            plot_clusters(self.dr_components_df, self.dr_cluster_labels, "not_a_dataframe", self.clustering_method)

    def test_invalid_dr_cluster_labels(self):
        """Test that the function raises a TypeError when dr_cluster_labels is not a numpy array."""
        with self.assertRaises(TypeError):
            plot_clusters(self.dr_components_df, "not_a_numpy_array", self.store_dr_components_df,
                          self.clustering_method)

    def test_plot_structure(self):
        """Test that the plot has the expected number of traces and correct layout title."""
        fig = plot_clusters(self.dr_components_df, self.dr_cluster_labels, self.store_dr_components_df,
                            self.clustering_method)

        # Check the number of traces in the figure (there should be 3 traces: clusters, same chain, and the chosen store's cluster)
        self.assertEqual(len(fig.data), 3, "The plot should have 3 traces.")

        # Check if the plot title matches the expected title with the clustering method
        expected_title = f'Store Clustering: Pricing Similarity using {self.clustering_method} Algorithm'
        self.assertEqual(fig.layout.title.text, expected_title, "The plot title should match the expected title.")

    def test_annotations_for_chosen_store(self):
        """Test that an annotation is added for the chosen store."""
        fig = plot_clusters(self.dr_components_df, self.dr_cluster_labels, self.store_dr_components_df,
                            self.clustering_method)

        # Check if the annotation for the chosen store exists
        annotations = fig.layout.annotations
        self.assertGreater(len(annotations), 0, "There should be an annotation for the chosen store.")

        # Check if the annotation has the correct text
        annotation_texts = [anno['text'] for anno in annotations]
        self.assertIn('The chosen store', annotation_texts, "The annotation should mention the chosen store.")

    def test_marker_for_chosen_store(self):
        """Test that the marker for the chosen store's cluster is bolder with a black border."""
        fig = plot_clusters(self.dr_components_df, self.dr_cluster_labels, self.store_dr_components_df,
                            self.clustering_method)

        # Check if any of the traces has the marker with a black border (for the chosen store's cluster)
        found_bold_marker = any(
            trace.marker.line.color == 'black' and trace.marker.line.width == 1.5
            for trace in fig.data if isinstance(trace, go.Scatter)
        )
        self.assertTrue(found_bold_marker,
                        "The chosen store's cluster should have a bolder marker with a black border.")

    def test_cluster_color_scale(self):
        """Test that the cluster markers have the correct color scale."""
        fig = plot_clusters(self.dr_components_df, self.dr_cluster_labels, self.store_dr_components_df,
                            self.clustering_method)

        # Check if the color scale for clusters is applied correctly
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.marker.colorscale == 'Rainbow':
                self.assertEqual(trace.marker.colorscale, 'Rainbow', "The colorscale should be Rainbow.")

if __name__ == '__main__':
    unittest.main()