import pandas as pd


def get_store_and_chain_ids(Store_data, SubChainName, StoreName):
    """
    Extracts the store and chain IDs based on SubChainName and StoreName.

    Parameters:
    - Store_data: The DataFrame containing store-related information.
    - SubChainName: The name of the sub-chain.
    - StoreName: The name of the store.

    Returns:
    - The filtered DataFrame with the rows that match the given SubChainName and StoreName.
    """
    Store_input_data = Store_data[(Store_data['SubChainName'] == SubChainName) & (Store_data['StoreName'] == StoreName)]
    return Store_input_data


def filter_nans(category_df, nan_threshold):
    """
    Filters observations from category_df where the number of NaN values exceeds a given threshold.

    Parameters:
    - category_df: The DataFrame containing the category price data.
    - nan_threshold: The threshold for allowed NaN values. For example, a value of 0.75 means 25% NaN values are allowed.

    Returns:
    - The filtered DataFrame with rows containing fewer NaN values than the threshold.
    """
    # Calculate the threshold for non-NaN values
    threshold_value = int((1 - nan_threshold) * category_df.shape[1])

    # Filter out rows where NaN values exceed the threshold
    filtered_df = category_df.dropna(thresh=threshold_value, axis=0)

    return filtered_df

