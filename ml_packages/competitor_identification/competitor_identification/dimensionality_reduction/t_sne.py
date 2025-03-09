from sklearn.manifold import TSNE
import pandas as pd


def perform_tsne_on_price_data(df, product_df, n_components=2, perplexity=40, max_iter=300, method='exact', init='random', learning_rate=200):
    """
    Perform t-SNE transformation on the given DataFrame and merge with product details.

    Parameters:
    - df: DataFrame containing the price data (either price movement or price level).
    - product_df: DataFrame containing the product metadata such as ChainID, StoreID, etc.
    - n_components: The number of dimensions to reduce to (default 2).
    - perplexity: The perplexity parameter for t-SNE (default 40).
    - n_iter: The number of iterations for optimization (default 300).
    - method: The optimization method to use (default 'exact').
    - init: Initialization method for the t-SNE algorithm (default 'random').
    - learning_rate: The learning rate for t-SNE (default 200).

    Returns:
    - principal_df: DataFrame with the embedded components from the t-SNE transformation.
    """
    # Perform t-SNE transformation
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, method=method, init=init, learning_rate=learning_rate)
    embedded_components = tsne.fit_transform(df)
    principal_df = pd.DataFrame(data=embedded_components)

    # Merge with product details (metadata)
    principal_df[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']] = product_df.reset_index()[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    return principal_df