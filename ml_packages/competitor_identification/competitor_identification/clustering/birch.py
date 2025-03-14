from sklearn.cluster import Birch

def birch_clustering(dr_components_df, method_params):
    """
    Perform BIRCH clustering on the data.

    Parameters:
    - dr_components_df: DataFrame with dimensionality reduction components.
    - method_params: Dictionary containing the parameters for BIRCH.

    Returns:
    - dr_components_df: DataFrame with updated Birch clustering labels.
    - dr_birch_labels: Array with the resulting cluster labels.
    """
    # Define and fit the BIRCH model with external parameters
    birch = Birch(**method_params)
    birch.fit(dr_components_df)

    # Get the cluster labels
    dr_birch_labels = birch.labels_
    dr_components_df['dr_cluster_labels'] = dr_birch_labels  # Assign the labels to the DataFrame

    return dr_components_df, dr_birch_labels