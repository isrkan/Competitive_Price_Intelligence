from sklearn.cluster import OPTICS
import pandas as pd

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
    try:
        # Validate input types
        if not isinstance(dr_components_df, pd.DataFrame):
            raise TypeError("dr_components_df must be a pandas DataFrame.")
        if not isinstance(method_params, dict):
            raise TypeError("method_params must be a dictionary.")

        # Define and fit the OPTICS model with external parameters
        optics = OPTICS(**method_params)
        optics.fit(dr_components_df)

        # Get the cluster labels
        optics_labels = optics.labels_
        dr_components_df['dr_cluster_labels'] = optics_labels  # Assign the labels to the DataFrame

        return dr_components_df, optics_labels

    except Exception as e:
        print(f"Error in optics clustering: {e}")
        raise