import umap.umap_ as umap
import pandas as pd

def perform_umap_on_price_data(df, product_df, n_neighbors=5, n_components=2):
    """
    Perform UMAP transformation on the given DataFrame and merge with product details.

    Parameters:
    - df: DataFrame containing the price data (either price movement or price level).
    - product_df: DataFrame containing the product metadata such as ChainID, StoreID, etc.
    - n_neighbors: The size of local neighborhood used for manifold approximation (default 5).
    - n_components: The number of dimensions to reduce to (default 2).

    Returns:
    - principal_df: DataFrame with the embedded components from the UMAP transformation.
    """
    # Perform UMAP transformation
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
    umap_components = umap_obj.fit_transform(df)
    principal_df = pd.DataFrame(data=umap_components)

    # Merge with product details (metadata)
    principal_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    return principal_df