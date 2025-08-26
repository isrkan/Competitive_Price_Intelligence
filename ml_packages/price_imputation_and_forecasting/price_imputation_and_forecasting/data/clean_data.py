import pandas as pd

def filter_missing_values_from_price_data(category_df, threshold_ratio):
    """
    Filters rows in the price data DataFrame based on the proportion of non-missing values.

    Parameters:
    - category_df (pd.DataFrame): Input price data.
    - threshold_ratio (float): Minimum ratio of non-NaN values required per row (between 0 and 1).

    Returns:
    - pd.DataFrame: Filtered DataFrame with rows meeting the threshold criteria.
    """
    if category_df.empty:
        raise ValueError("Input DataFrame 'category_df' is empty.")
    if not isinstance(category_df, pd.DataFrame):
        raise TypeError(f"Expected category_df to be a pandas DataFrame, got {type(category_df)} instead.")
    if not (0 < threshold_ratio <= 1):
        raise ValueError(f"threshold_ratio must be in (0,1], got {threshold_ratio}")

    num_cols = len(category_df.columns)
    thresh = int(threshold_ratio * num_cols)

    try:
        df_filtered = category_df.dropna(thresh=thresh, axis=0)
    except Exception as e:
        raise RuntimeError(f"Error filtering DataFrame with threshold {thresh}: {e}")
    return df_filtered