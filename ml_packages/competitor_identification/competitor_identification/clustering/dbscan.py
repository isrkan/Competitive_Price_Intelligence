import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def find_optimal_eps(data, n_neighbors=2):
    """
    Find the optimal epsilon (eps) using the elbow method.

    Parameters:
    - data: DataFrame with the features for DBSCAN clustering.
    - n_neighbors: Number of neighbors for the distance calculation (default 2).

    Returns:
    - distances: Array of distances to nearest neighbors.
    - knee_point: The optimal eps (knee point).
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(data)

    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]  # Get the distance to the second nearest neighbor

    knee_locator = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    knee_point = knee_locator.knee

    return distances, knee_point


def dbscan_clustering(dr_components_df, min_samples=5, eps=None):
    """
    Perform DBSCAN clustering on the data.

    Parameters:
    - dr_components_df: DataFrame with dimensionality reduction components.
    - min_samples: Minimum number of samples required to form a cluster (default 5).
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood (default None).

    Returns:
    - dr_components_df: DataFrame with updated DBSCAN clustering labels.
    - dbscan_labels: Array with the resulting cluster labels.
    """
    if eps is None:
        # Find the optimal epsilon using the elbow method
        distances, knee_point = find_optimal_eps(dr_components_df)
        eps = distances[knee_point]

    # Define and fit the DBSCAN model with external parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(dr_components_df)

    # Get the cluster labels
    dbscan_labels = dbscan.labels_
    dr_components_df['dr_labels'] = dbscan_labels  # Assign the labels to the DataFrame

    return dr_components_df, dbscan_labels