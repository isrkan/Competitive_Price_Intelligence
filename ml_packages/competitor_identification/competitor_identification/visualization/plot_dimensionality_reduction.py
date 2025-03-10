import plotly.express as px
import plotly.graph_objects as go


def plot_dimensionality_reduction(combined_components_df, chosen_store_info):
    """
    Plot the dimensionality reduction results for price movement and price level.
    Highlights the input store and its subchain.

    Parameters:
    - combined_components_df: DataFrame containing the principal components (price movement and price level).
    - Store_input_data: DataFrame containing the store metadata for highlighting.

    Returns:
    - fig1: Plotly figure object with the scatter plot of the dimensionality reduction results.
    """

    # Create scatter plot of the principal components
    fig1 = px.scatter(combined_components_df,
                      x='PrincipalPriceMovement',
                      y='PrincipalPriceLevel',
                      color='SubChainName',
                      hover_data=['ChainName', 'SubChainName', 'StoreName'])

    # Add trace to highlight the input store (Store_input_data) with black border
    fig1.add_trace(go.Scatter(
        x=combined_components_df[combined_components_df['StoreID'] == chosen_store_info['StoreID'].iloc[0]]['PrincipalPriceMovement'],
        y=combined_components_df[combined_components_df['StoreID'] == chosen_store_info['StoreID'].iloc[0]]['PrincipalPriceLevel'],
        mode='markers',
        text=combined_components_df[combined_components_df['StoreID'] == chosen_store_info['StoreID'].iloc[0]]['SubChainName'] + '<br>' + combined_components_df[combined_components_df['StoreID'] == chosen_store_info['StoreID'].iloc[0]]['StoreName'],
        hovertemplate='%{text}',
        marker=dict(size=6, color='rgba(0,0,0,0)', line=dict(color='black', width=1.2)),
        showlegend=False
    ))

    # Bold the input subchain in the legend if its name matches
    fig1.for_each_trace(lambda t: t.update(name=t.name.replace(chosen_store_info['SubChainName'].iloc[0], '<b>' + chosen_store_info['SubChainName'].iloc[0] + '</b>')) if t.name == chosen_store_info['SubChainName'].iloc[0] else t)

    # Update layout and titles
    fig1.update_layout(
        title='Market Structure: Pricing Similarity in Terms of Price Level and Price Movement',
        legend=dict(title='Sub-Chain Name', yanchor="top", y=0.99, xanchor="left", x=1)
    )

    # Return the plotly figure
    return fig1