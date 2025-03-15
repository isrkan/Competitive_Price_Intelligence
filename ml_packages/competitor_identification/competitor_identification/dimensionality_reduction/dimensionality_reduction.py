import pandas as pd
from .pca import perform_pca_on_price_data
from .t_sne import perform_tsne_on_price_data
from .umap import perform_umap_on_price_data

def perform_dimensionality_reduction_on_price_movement_and_level(price_movement_df, price_level_df, product_df, method, method_params):
    """
    Perform dimensionality reduction on both price movement and price level data and combine the results.

    Parameters:
    - price_movement_df: DataFrame with scaled price movement data.
    - price_level_df: DataFrame with scaled price level data.
    - product_df: DataFrame containing the product metadata.
    - method: The dimensionality reduction method to apply ('pca', 't-sne', 'umap').
    - method_params: Parameters specific to the dimensionality reduction method.

    Returns:
    - combined_components_df: DataFrame with combined dimensionality reduction results for price movement and price level.
    - price_movement_principal_pca: DataFrame with dimensionality reduction results for price movement.
    - price_level_principal_pca: DataFrame with dimensionality reduction results for price level.
    """
    try:
        # Validate input types
        if not isinstance(price_movement_df, pd.DataFrame):
            raise TypeError("price_movement_df must be a pandas DataFrame.")
        if not isinstance(price_level_df, pd.DataFrame):
            raise TypeError("price_level_df must be a pandas DataFrame.")
        if not isinstance(product_df, pd.DataFrame):
            raise TypeError("product_df must be a pandas DataFrame.")
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        if not isinstance(method_params, dict):
            raise TypeError("method_params must be a dictionary.")

        # Ensure DataFrames are not empty
        if price_movement_df.empty:
            raise ValueError("price_movement_df is empty. Cannot perform dimensionality reduction.")
        if price_level_df.empty:
            raise ValueError("price_level_df is empty. Cannot perform dimensionality reduction.")
        if product_df.empty:
            raise ValueError("product_df is empty. Cannot merge metadata.")

        # Validate method input
        valid_methods = {'pca', 't-sne', 'umap'}
        if method not in valid_methods:
            raise ValueError(f"Invalid method: '{method}'. Choose from {valid_methods}.")

        if method == 'pca':
            # Perform PCA for price movement
            price_movement_principal_df = perform_pca_on_price_data(price_movement_df, product_df, method_params)
            # Perform PCA for price level
            price_level_principal_df = perform_pca_on_price_data(price_level_df, product_df, method_params)
        elif method == 't-sne':
            # Perform PCA for price movement
            price_movement_principal_df = perform_tsne_on_price_data(price_movement_df, product_df, method_params)
            # Perform PCA for price level
            price_level_principal_df = perform_tsne_on_price_data(price_level_df, product_df, method_params)
        elif method == 'umap':
            # Perform PCA for price movement
            price_movement_principal_df = perform_umap_on_price_data(price_movement_df, product_df, method_params)
            # Perform PCA for price level
            price_level_principal_df = perform_umap_on_price_data(price_level_df, product_df, method_params)

        # Ensure principal components have at least one column
        if price_movement_principal_df.shape[1] < 1 or price_level_principal_df.shape[1] < 1:
            raise ValueError("Dimensionality reduction resulted in empty components.")

        # Combine the principal components from both PCA results
        combined_components_df = pd.concat([price_movement_principal_df.iloc[:, 0], price_level_principal_df.iloc[:, 0]], axis=1)
        combined_components_df.columns = ['PrincipalPriceMovement', 'PrincipalPriceLevel']

        # Merge the product metadata with the combined PCA results
        combined_components_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

        # Return the results
        return combined_components_df, price_movement_principal_df, price_level_principal_df

    except Exception as e:
        print(f"Error in perform_dimensionality_reduction_on_price_movement_and_level: {e}")
        raise