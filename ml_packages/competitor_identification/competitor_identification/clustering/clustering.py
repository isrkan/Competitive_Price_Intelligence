from .birch import birch_clustering
from .dbscan import dbscan_clustering
from .optics import optics_clustering
import numpy as np
import pandas as pd


def perform_clustering(dr_components_df, chosen_store_info, same_chain_dr_components_df, method, **kwargs):
    """
    Perform clustering on the data based on the specified method.

    Parameters:
    - dr_components_df: DataFrame with dimensionality reduction components.
    - chosen_store_info: DataFrame with store metadata for filtering.
    - same_chain_dr_components_df: DataFrame containing dimensionality reduction components for stores in the same chain.
    - method: The clustering method to apply ('birch', 'dbscan', 'optics').
    - kwargs: Additional parameters for the clustering algorithms.

    Returns:
    - clustered_dr_components_df: DataFrame with updated clustering labels.
    - cluster_labels: Array with the resulting cluster labels.
    - clustered_store_dr_components_df: DataFrame with the store's dimensionality reduction components.
    - clustered_competitors_dr_components_df: DataFrame with competitors' dimensionality reduction components.
    """

    # Perform clustering based on the method
    if method == 'birch':
        clustered_dr_components_df, cluster_labels = birch_clustering(dr_components_df, **kwargs)
    elif method == 'dbscan':
        clustered_dr_components_df, cluster_labels = dbscan_clustering(dr_components_df, **kwargs)
    elif method == 'optics':
        clustered_dr_components_df, cluster_labels = optics_clustering(dr_components_df, **kwargs)
    else:
        raise ValueError(f"Clustering method '{method}' not supported. Choose from ['birch', 'dbscan', 'optics']")

    # Reset index to make filtering easier
    clustered_dr_components_df = clustered_dr_components_df.reset_index()
    # Filter data for the chosen store
    clustered_store_dr_components_df = clustered_dr_components_df[clustered_dr_components_df['StoreID'] == chosen_store_info['StoreID'].iloc[0]]

    # Filter competitors: same cluster label but different SubChainID
    clustered_competitors_dr_components_df = clustered_dr_components_df[(clustered_dr_components_df['dr_cluster_labels'] == clustered_store_dr_components_df['dr_cluster_labels'].iloc[0]) & (clustered_dr_components_df['SubChainID'] != clustered_store_dr_components_df['SubChainID'].iloc[0])]

    # If no competitors are found, select stores in other chains
    if len(clustered_competitors_dr_components_df) == 0:
        clustered_competitors_dr_components_df = clustered_dr_components_df[
            clustered_dr_components_df['SubChainID'] != clustered_store_dr_components_df['SubChainID'].iloc[0]]

    same_chain_dr_components_df['dr_cluster_labels'] = max(np.unique(cluster_labels)) + 1

    clustered_dr_components_df = pd.concat([clustered_dr_components_df, same_chain_dr_components_df])
    cluster_labels = np.concatenate((cluster_labels, same_chain_dr_components_df['dr_cluster_labels'].to_numpy()))

    # Return the results: clustered dr_components_df, cluster labels, and filtered data
    return clustered_dr_components_df, cluster_labels, clustered_store_dr_components_df, clustered_competitors_dr_components_df