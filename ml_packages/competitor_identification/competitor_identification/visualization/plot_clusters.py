import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd

def plot_clusters(dr_components_df, dr_cluster_labels, store_dr_components_df, clustering_method):
    """
    Plots the clustering result based on the dimensionality reduction components.

    Parameters:
    - dr_components_df: DataFrame with the dimensionality reduction components and clustering results.
    - dr_cluster_labels: Array of cluster labels resulting from the clustering algorithm.
    - store_dr_components_df: DataFrame containing the selected store's dimensionality reduction components.

    Returns:
    - fig2: The plotly figure object with the clustering visualization.
    """
    try:
        # Validate input types
        if not isinstance(dr_components_df, pd.DataFrame):
            raise TypeError("dr_components_df must be a pandas DataFrame.")
        if not isinstance(store_dr_components_df, pd.DataFrame):
            raise TypeError("store_dr_components_df must be a pandas DataFrame.")
        if not isinstance(dr_cluster_labels, np.ndarray):
            raise TypeError("dr_cluster_labels must be a numpy array.")

        fig2 = go.Figure()

        # Add points of all clusters except the selected store's cluster and the same chain stores
        cluster_mask = ((dr_components_df['dr_cluster_labels'] != max(np.unique(dr_cluster_labels))) & (dr_components_df['dr_cluster_labels'] != store_dr_components_df['dr_cluster_labels'].iloc[0]))
        fig2.add_trace(go.Scatter(
            x=dr_components_df[cluster_mask]['P_M_1'],
            y=dr_components_df[cluster_mask]['P_L_1'],
            mode='markers',
            text='Chain: ' + dr_components_df[cluster_mask]['ChainName'] +
                 '<br> SubChain: ' + dr_components_df[cluster_mask]['SubChainName'] +
                 '<br> Store: ' + dr_components_df[cluster_mask]['StoreName'] +
                 '<br> Cluster: ' + dr_components_df[cluster_mask]['dr_cluster_labels'].astype(str),
            hovertemplate='%{text}',
            marker=dict(
                color=dr_cluster_labels[cluster_mask],
                colorscale='Rainbow',
                opacity=0.7
            ),
            showlegend=False
        ))

        # Add marks for stores from the same chain
        cluster_mask = dr_components_df['dr_cluster_labels'] == max(np.unique(dr_cluster_labels))
        fig2.add_trace(go.Scatter(
            x=dr_components_df[cluster_mask]['P_M_1'],
            y=dr_components_df[cluster_mask]['P_L_1'],
            mode='markers',
            text='Chain: ' + dr_components_df[cluster_mask]['ChainName'] +
                 '<br> SubChain: ' + dr_components_df[cluster_mask]['SubChainName'] +
                 '<br> Store: ' + dr_components_df[cluster_mask]['StoreName'] +
                 '<br> Cluster: Same Chain',
            hovertemplate='%{text}',
            marker=dict(color='gray', opacity=1, size=8),
            showlegend=False
        ))

        # Add bolder points for the specific cluster of the chosen store
        cluster_mask = dr_components_df['dr_cluster_labels'] == store_dr_components_df['dr_cluster_labels'].iloc[0]
        fig2.add_trace(go.Scatter(
            x=dr_components_df[cluster_mask]['P_M_1'],
            y=dr_components_df[cluster_mask]['P_L_1'],
            mode='markers',
            text='Chain: ' + dr_components_df[cluster_mask]['ChainName'] +
                 '<br> SubChain: ' + dr_components_df[cluster_mask]['SubChainName'] +
                 '<br> Store: ' + dr_components_df[cluster_mask]['StoreName'] +
                 '<br> Cluster: ' + dr_components_df[cluster_mask]['dr_cluster_labels'].astype(str),
            hovertemplate='%{text}',
            marker=dict(
                color=dr_cluster_labels[cluster_mask],
                colorscale='Rainbow',
                opacity=0.7,
                size=8,
                line=dict(color='black', width=1.5)
            ),
            showlegend=False
        ))

        # Add arrow annotation for the chosen store
        fig2.add_annotation(
            x=store_dr_components_df['P_M_1'].iloc[0],
            y=store_dr_components_df['P_L_1'].iloc[0],
            ax=-60,
            ay=60,
            text='The chosen store',
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor='black'
        )

        # Update the layout of the figure
        fig2.update_layout(
            xaxis_title='Principal Price Movement',
            yaxis_title='Principal Price Level',
            title=f'Store Clustering: Pricing Similarity using {clustering_method} Algorithm'
        )

        return fig2

    except Exception as e:
        raise TypeError(f"Error in plot clusters: {e}")