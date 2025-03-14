from sklearn.manifold import TSNE
import pandas as pd


def perform_tsne_on_price_data(df, product_df, method_params):
    """
    Perform t-SNE transformation on the given DataFrame and merge with product details.

    Parameters:
    - df: DataFrame containing the price data (either price movement or price level).
    - product_df: DataFrame containing the product metadata such as ChainID, StoreID, etc.
    - method_params: Dictionary containing the parameters for t-SNE.

    Returns:
    - principal_df: DataFrame with the embedded components from the t-SNE transformation.
    """
    # Perform t-SNE transformation
    tsne = TSNE(**method_params)
    embedded_components = tsne.fit_transform(df)
    principal_df = pd.DataFrame(data=embedded_components)

    # Merge with product details (metadata)
    principal_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    return principal_df