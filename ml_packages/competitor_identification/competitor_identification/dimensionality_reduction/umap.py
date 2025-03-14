import umap.umap_ as umap
import pandas as pd

def perform_umap_on_price_data(df, product_df, method_params):
    """
    Perform UMAP transformation on the given DataFrame and merge with product details.

    Parameters:
    - df: DataFrame containing the price data (either price movement or price level).
    - product_df: DataFrame containing the product metadata such as ChainID, StoreID, etc.
    - method_params: Dictionary containing the parameters for UMAP.

    Returns:
    - principal_df: DataFrame with the embedded components from the UMAP transformation.
    """
    # Perform UMAP transformation
    umap_obj = umap.UMAP(**method_params)
    umap_components = umap_obj.fit_transform(df)
    principal_df = pd.DataFrame(data=umap_components)

    # Merge with product details (metadata)
    principal_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    return principal_df