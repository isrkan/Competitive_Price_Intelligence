from sklearn.cluster import OPTICS

def optics_clustering(dr_components_df, min_samples=5, min_cluster_size=0.02, metric='minkowski'):
    """
    Perform OPTICS clustering on the data.

    Parameters:
    - dr_components_df: DataFrame with PCA components.
    - min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point (default 5).
    - min_cluster_size: The minimum size of a cluster (default 0.02).
    - metric: The distance metric to use for the OPTICS algorithm (default 'minkowski').

    Returns:
    - dr_components_df: DataFrame with updated OPTICS clustering labels.
    - optics_labels: Array with the resulting cluster labels.
    """
    # Define and fit the OPTICS model with external parameters
    optics = OPTICS(min_samples=min_samples, metric=metric, min_cluster_size=min_cluster_size)
    optics.fit(dr_components_df)

    # Get the cluster labels
    optics_labels = optics.labels_
    dr_components_df['dr_labels'] = optics_labels  # Assign the labels to the DataFrame

    return dr_components_df, optics_labels