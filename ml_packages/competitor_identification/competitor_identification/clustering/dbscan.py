import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def find_optimal_eps(data, n_neighbors):
    """
    Find the optimal epsilon (eps) using the elbow method.

    Parameters:
    - data: DataFrame with the features for DBSCAN clustering.
    - n_neighbors: Number of neighbors for the distance calculation.

    Returns:
    - distances: Array of distances to nearest neighbors.
    - knee_point: The optimal eps (knee point).
    """
    try:
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(data)

        distances, indices = neighbors_fit.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]  # Get the distance to the second nearest neighbor

        knee_locator = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        knee_point = knee_locator.knee
        return distances, knee_point

    except Exception as e:
        print(f"Error in find_optimal_eps: {e}")
        return None

def dbscan_clustering(dr_components_df, method_params):
    """
    Perform DBSCAN clustering on the data.

    Parameters:
    - dr_components_df: DataFrame with dimensionality reduction components.
    - method_params: Dictionary containing the parameters for DBSCAN.

    Returns:
    - dr_components_df: DataFrame with updated DBSCAN clustering labels.
    - dbscan_labels: Array with the resulting cluster labels.
    """
    try:
        # Validate input types
        if not isinstance(dr_components_df, pd.DataFrame):
            raise TypeError("dr_components_df must be a pandas DataFrame.")
        if not isinstance(method_params, dict):
            raise TypeError("method_params must be a dictionary.")

        # Attempt to extract and process 'eps' from method_params
        eps = method_params.get('eps')
        n_neighbors = method_params.get('n_neighbors', 2) # Default to 2 if not provided

        if eps is None:
            # If eps is not provided, calculate it using the elbow method
            print("Warning: 'eps' not found in method_params. Calculating optimal eps...")
            distances, knee_point = find_optimal_eps(dr_components_df, n_neighbors)
            eps = distances[knee_point]  # Use the optimal eps based on elbow method

        # Remove 'eps' and 'n_neighbors' from method_params to avoid passing it again
        method_params.pop('eps', None)
        method_params.pop('n_neighbors', None)

        # Now initialize DBSCAN with the updated method_params including eps
        dbscan = DBSCAN(eps=eps, **method_params)
    except Exception as e:
        # If there's any issue in the above try block, fallback to initializing DBSCAN without eps
        print(f"Error while processing 'eps': {e}")
        print("Falling back to DBSCAN with only the method_params.")
        # Initialize DBSCAN without eps, just using the other parameters
        dbscan = DBSCAN(**method_params)

    try:
        # Fit the DBSCAN model
        dbscan.fit(dr_components_df)

        # Get the cluster labels
        dbscan_labels = dbscan.labels_
        dr_components_df['dr_cluster_labels'] = dbscan_labels  # Assign the labels to the DataFrame

        return dr_components_df, dbscan_labels

    except Exception as e:
        print(f"Error during DBSCAN fitting: {e}")
        raise