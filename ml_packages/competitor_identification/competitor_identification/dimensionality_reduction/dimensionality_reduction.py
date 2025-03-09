import pandas as pd
from .pca import perform_pca_on_price_data
from .t_sne import perform_tsne_on_price_data
from .umap import perform_umap_on_price_data

def perform_dimensionality_reduction_on_price_movement_and_level(price_movement_df, price_level_df, product_df, method):
    """
    Perform dimensionality reduction on both price movement and price level data and combine the results.

    Parameters:
    - price_movement_df: DataFrame with scaled price movement data.
    - price_level_df: DataFrame with scaled price level data.
    - product_df: DataFrame containing the product metadata.

    Returns:
    - combined_components_df: DataFrame with combined dimensionality reduction results for price movement and price level.
    - price_movement_principal_pca: DataFrame with dimensionality reduction results for price movement.
    - price_level_principal_pca: DataFrame with dimensionality reduction results for price level.
    """

    if method == 'pca':
        # Perform PCA for price movement
        price_movement_principal_df = perform_pca_on_price_data(price_movement_df, product_df)
        # Perform PCA for price level
        price_level_principal_df = perform_pca_on_price_data(price_level_df, product_df)
    elif method == 't-sne':
        # Perform PCA for price movement
        price_movement_principal_df = perform_tsne_on_price_data(price_movement_df, product_df)
        # Perform PCA for price level
        price_level_principal_df = perform_tsne_on_price_data(price_level_df, product_df)
    elif method == 'umap':
        # Perform PCA for price movement
        price_movement_principal_df = perform_umap_on_price_data(price_movement_df, product_df)
        # Perform PCA for price level
        price_level_principal_df = perform_umap_on_price_data(price_level_df, product_df)

    # Combine the principal components from both PCA results
    combined_components_df = pd.concat([price_movement_principal_df.iloc[:, 0], price_level_principal_df.iloc[:, 0]], axis=1)
    combined_components_df.columns = ['PrincipalPriceMovement', 'PrincipalPriceLevel']

    # Merge the product metadata with the combined PCA results
    combined_components_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    # Return the results
    return combined_components_df, price_movement_principal_df, price_level_principal_df