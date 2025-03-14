from sklearn.cluster import OPTICS

def optics_clustering(dr_components_df, method_params):
    """
    Perform OPTICS clustering on the data.

    Parameters:
    - dr_components_df: DataFrame with PCA components.
    - method_params: Dictionary containing the parameters for OPTICS.

    Returns:
    - dr_components_df: DataFrame with updated OPTICS clustering labels.
    - optics_labels: Array with the resulting cluster labels.
    """
    # Define and fit the OPTICS model with external parameters
    optics = OPTICS(**method_params)
    optics.fit(dr_components_df)

    # Get the cluster labels
    optics_labels = optics.labels_
    dr_components_df['dr_cluster_labels'] = optics_labels  # Assign the labels to the DataFrame

    return dr_components_df, optics_labels