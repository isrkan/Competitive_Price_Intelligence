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


import numpy as np


def find_top_competitors(clustered_store_dr_components_df, clustered_competitors_dr_components_df, top_n=5):
    """
    Find the top N competitors based on Euclidean distance in dimensionality reduction space.

    Parameters:
    - clustered_store_dr_components_df: DataFrame containing the clustered dimensionality reduction components of the store.
    - clustered_competitors_dr_components_df: DataFrame containing the clustered dimensionality reduction components of the competitors.
    - top_n: The number of top competitors to return (default is 5).

    Returns:
    - top_competitors: DataFrame with the top N competitors based on Euclidean distance.
    """

    # Calculate Euclidean distance between the store and each competitor
    def euclidean_distance(row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    # Store dimensionality reduction components (assuming the store is the first row)
    store_row = clustered_store_dr_components_df.iloc[0][['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']]

    # Calculate Euclidean distance for each row in competitors data
    euclidean_distances = clustered_competitors_dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']].apply(lambda row: euclidean_distance(store_row, row), axis=1)

    # Add Euclidean distance column to the competitors DataFrame
    clustered_competitors_dr_components_df['EuclideanDistance'] = euclidean_distances

    # Sort competitors by the Euclidean distance (ascending order)
    competitors_pca_components_df = clustered_competitors_dr_components_df.sort_values(by=['EuclideanDistance'])

    # Select the top N competitors
    top_competitors = competitors_pca_components_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName']].head(top_n).reset_index(drop=True)

    return top_competitors