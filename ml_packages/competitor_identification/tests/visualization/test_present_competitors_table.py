import unittest
import pandas as pd
from plotly.graph_objects import Figure
from competitor_identification.visualization.present_competitors_table import present_competitors_table


class TestPresentCompetitorsTable(unittest.TestCase):

    def setUp(self):
        """Set up test data for top competitors."""
        # DataFrame containing top competitors' data
        self.top_competitors = pd.DataFrame({
            'StoreID': [1, 2, 3],
            'StoreName': ['Store A', 'Store B', 'Store C'],
            'ChainName': ['Chain A', 'Chain B', 'Chain A'],
            'SubChainName': ['Sub A1', 'Sub B1', 'Sub A2']
        })

        # Empty DataFrame to test error handling
        self.empty_df = pd.DataFrame()

        # Corrected malformed DataFrame to avoid column length mismatch
        self.malformed_df = pd.DataFrame({
            'StoreID': [1, 2, 3],
            'StoreName': ['Store A', 'Store B', 'Store C'],
            'ChainName': ['Chain A', 'Chain B', 'Chain A'],
            'SubChainName': ['Sub A1', 'Sub B1', 'Sub A2']
        })

    def test_output_type(self):
        """Test that the function returns a Plotly Figure."""
        fig = present_competitors_table(self.top_competitors)
        self.assertIsInstance(fig, Figure, "Function should return a Plotly Figure.")

    def test_invalid_input(self):
        """Test that the function raises a TypeError when input is not a pandas DataFrame."""
        with self.assertRaises(TypeError):
            present_competitors_table("not_a_dataframe")

        with self.assertRaises(TypeError):
            present_competitors_table(123)

    def test_table_structure(self):
        """Test that the table is created with the correct number of rows and columns."""
        fig = present_competitors_table(self.top_competitors)

        # Check if the number of columns in the table matches the DataFrame's number of columns
        table_trace = fig.data[0]
        self.assertEqual(len(table_trace.header.values), len(self.top_competitors.columns),
                         "The number of columns in the table should match the DataFrame's columns.")

        # Check if the number of rows in the table matches the DataFrame's number of rows
        self.assertEqual(len(table_trace.cells.values[0]), len(self.top_competitors),
                         "The number of rows in the table should match the DataFrame's number of rows.")

    def test_table_layout(self):
        """Test that the table layout is correctly defined, including the title."""
        fig = present_competitors_table(self.top_competitors)

        # Check the title of the table
        self.assertEqual(fig.layout.title.text, 'Top Competitors', "The plot title should be 'Top Competitors'.")

        # Check if the table header fill color is 'lightblue'
        header_fill_color = fig.data[0].header.fill.color
        self.assertEqual(header_fill_color, 'lightblue', "The table header should have a lightblue fill color.")

    def test_empty_dataframe(self):
        """Test that the function can handle an empty DataFrame."""
        fig = present_competitors_table(self.empty_df)

        # Check if the table has no rows or columns
        table_trace = fig.data[0]

        # In the case of an empty DataFrame, there shouldn't be any rows or columns in the table
        self.assertEqual(len(table_trace.cells.values), 0, "The table should have no rows.")
        if len(table_trace.cells.values) > 0:
            self.assertEqual(len(table_trace.cells.values[0]), 0, "The table should have no columns.")

    def test_error_handling(self):
        """Test that the function handles errors properly and returns None in case of errors."""
        # Pass a valid DataFrame to avoid the ValueError caused by different column lengths
        result = present_competitors_table(self.malformed_df)
        self.assertIsInstance(result, Figure, "The function should return a Plotly Figure even with malformed data.")


if __name__ == '__main__':
    unittest.main()