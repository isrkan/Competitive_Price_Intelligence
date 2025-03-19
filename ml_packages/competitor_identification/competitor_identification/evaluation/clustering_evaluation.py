from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_clustering(cluster_labels, clustered_dr_components_df):
    """
    Evaluate the clustering results using silhouette score and davies-bouldin score.

    Parameters:
    - cluster_labels: Array of cluster labels from the clustering.
    - clustering_results_df: DataFrame containing the clustered dimensionality reduction components.

    Returns:
    - silhouette: The silhouette score of the clustering.
    - davies_bouldin: The Davies-Bouldin score of the clustering.
    """
    try:
        if len(cluster_labels) != len(clustered_dr_components_df):
            raise ValueError("The number of cluster labels must match the number of rows in dr_components_df.")

        # Compute silhouette score
        silhouette = silhouette_score(clustered_dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']], cluster_labels)

        # Compute Davies-Bouldin score
        davies_bouldin = davies_bouldin_score(clustered_dr_components_df[['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']], cluster_labels)

        # Return evaluation scores
        return silhouette, davies_bouldin

    except Exception as e:
        print(f"Error in evaluate_clustering: {e}")
        raise