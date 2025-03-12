import plotly.graph_objects as go

def present_competitors_table(top_competitors):
    """
    Present a table displaying the top competitors.

    Parameters:
    - top_competitors: DataFrame containing the top competitors' data.

    Returns:
    - fig3: Plotly Figure containing the competitors' table.
    """

    # Create a table trace for the competitors' data
    table_trace = go.Table(
        header=dict(
            values=list(top_competitors.columns),  # Get column names from DataFrame
            fill_color='lightblue',
            align='left'),
        cells=dict(
            values=[top_competitors[col] for col in top_competitors.columns],  # Get column values from DataFrame
            fill_color='white',
            align='left'))

    # Define the layout for the table figure
    layout = go.Layout(title='Top Competitors')

    # Create a Plotly Figure with the table
    fig3 = go.Figure(data=[table_trace], layout=layout)

    return fig3