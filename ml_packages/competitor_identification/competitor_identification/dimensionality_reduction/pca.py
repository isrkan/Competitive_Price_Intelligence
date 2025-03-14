from sklearn.decomposition import PCA
import pandas as pd


def perform_pca_on_price_data(df, product_df, method_params):
    """
    Perform PCA transformation on the given DataFrame and merge with product details.

    Parameters:
    - df: DataFrame containing the price data (either price movement or price level).
    - product_df: DataFrame containing the product metadata such as ChainID, StoreID, etc.
    - method_params: Dictionary containing the parameters for PCA.

    Returns:
    - principal_df: DataFrame with the principal components of the input price data.
    """
    # Perform PCA transformation
    pca = PCA(**method_params)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(data=principal_components)

    # Merge with product details (metadata)
    principal_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    return principal_df