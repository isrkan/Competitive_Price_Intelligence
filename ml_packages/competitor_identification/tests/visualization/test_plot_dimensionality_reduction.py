import unittest
import pandas as pd
import plotly.graph_objects as go
from competitor_identification.visualization.plot_dimensionality_reduction import plot_dimensionality_reduction


class TestPlotDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        self.combined_components_df = pd.DataFrame({
            'StoreID': [1, 2, 3, 4, 5],
            'PrincipalPriceMovement': [0.5, -0.2, 0.3, -0.5, 0.0],
            'PrincipalPriceLevel': [1.2, -0.7, 0.5, -1.0, 0.3],
            'ChainName': ['Chain A', 'Chain B', 'Chain A', 'Chain C', 'Chain B'],
            'SubChainName': ['Sub A1', 'Sub B1', 'Sub A1', 'Sub C1', 'Sub B2'],
            'StoreName': ['Store X', 'Store Y', 'Store Z', 'Store W', 'Store V']
        })

        self.chosen_store_info = pd.DataFrame({
            'StoreID': [3],
            'SubChainName': ['Sub A1']
        })

    def test_output_type(self):
        """Test that the function returns a Plotly Figure."""
        fig = plot_dimensionality_reduction(self.combined_components_df, self.chosen_store_info)
        self.assertIsInstance(fig, go.Figure, "Function should return a Plotly Figure.")

    def test_invalid_inputs(self):
        """Test that the function raises errors for invalid inputs."""
        with self.assertRaises(TypeError):
            plot_dimensionality_reduction("not a dataframe", self.chosen_store_info)

        with self.assertRaises(TypeError):
            plot_dimensionality_reduction(self.combined_components_df, "not a dataframe")

        with self.assertRaises(TypeError):
            plot_dimensionality_reduction(123, 456)

    def test_highlighted_store(self):
        """Test that the chosen store is correctly highlighted in the plot."""
        fig = plot_dimensionality_reduction(self.combined_components_df, self.chosen_store_info)
        found_highlight = any(
            trace.marker.line.color == 'black' for trace in fig.data if isinstance(trace, go.Scatter)
        )
        self.assertTrue(found_highlight, "The chosen store should be highlighted with a black border.")

    def test_legend_format(self):
        """Test that the chosen store's subchain is bold in the legend."""
        fig = plot_dimensionality_reduction(self.combined_components_df, self.chosen_store_info)
        legend_names = [trace.name for trace in fig.data if trace.name]
        expected_bold_name = f"<b>{self.chosen_store_info['SubChainName'].iloc[0]}</b>"
        self.assertIn(expected_bold_name, legend_names, "The chosen store's subchain should be bold in the legend.")

    def test_plot_structure(self):
        """Test that the plot has the expected number of traces and correct layout title."""
        fig = plot_dimensionality_reduction(self.combined_components_df, self.chosen_store_info)
        self.assertEqual(len(fig.data), len(self.combined_components_df['SubChainName'].unique()) + 1,
                         "There should be one trace per unique subchain plus one for highlighting the chosen store.")
        self.assertEqual(fig.layout.title.text,
                         'Market Structure: Pricing Similarity in Terms of Price Level and Price Movement',
                         "Plot title should match expected.")


if __name__ == '__main__':
    unittest.main()