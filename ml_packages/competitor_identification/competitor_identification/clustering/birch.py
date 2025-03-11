from sklearn.cluster import Birch

def birch_clustering(dr_components_df, threshold=9, n_clusters=None):
    """
    Perform Birch clustering on the data.

    Parameters:
    - dr_components_df: DataFrame with dimensionality reduction components.
    - threshold: The threshold for the Birch algorithm (default 9).
    - n_clusters: The number of clusters (default None, meaning no fixed number of clusters).

    Returns:
    - dr_components_df: DataFrame with updated Birch clustering labels.
    - dr_birch_labels: Array with the resulting cluster labels.
    """
    # Define and fit the BIRCH model with external parameters
    birch = Birch(threshold=threshold, n_clusters=n_clusters)
    birch.fit(dr_components_df)

    # Get the cluster labels
    dr_birch_labels = birch.labels_
    dr_components_df['dr_labels'] = dr_birch_labels  # Assign the labels to the DataFrame

    return dr_components_df, dr_birch_labels