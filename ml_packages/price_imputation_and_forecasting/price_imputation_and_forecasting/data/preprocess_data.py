import pandas as pd
import numpy as np

def merge_encoded_store_data_with_price_data(store_data_with_dummies, price_data):
    """
    Merges one-hot encoded store data with price data using StoreID as the key.

    Parameters:
    - store_dummies (pd.DataFrame): One-hot encoded store metadata with StoreID as index.
    - price_data (pd.DataFrame): Price data with 'StoreID' as a column.

    Returns:
    - pd.DataFrame: Merged dataset.
    """
    print(price_data.index.names)
    # Check price_data is indexed by StoreID
    if 'StoreID' not in price_data.index.names:
        raise ValueError("`price_data` must be indexed by 'StoreID'.")
    # Check store_dummies is indexed by StoreID
    if store_data_with_dummies.index.name != 'StoreID':
        raise ValueError("`store_dummies` must be indexed by 'StoreID'.")
    # Validate inputs
    if not isinstance(store_data_with_dummies, pd.DataFrame) or not isinstance(price_data, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames.")

    try:
        # Merge store metadata with product price time series
        merged_df = store_data_with_dummies.merge(price_data, how='right', left_index=True, right_on='StoreID')
    except Exception as e:
        raise RuntimeError(f"Failed to merge store and price data: {e}")

    return merged_df


def convert_to_numpy_inputs(df_input, df_target):
    """
    Converts input and target DataFrames into NumPy arrays.

    Parameters:
    - df_input (pd.DataFrame): Input features (merged store + price data).
    - df_target (pd.DataFrame): Target values (imputed time series).

    Returns:
    - Tuple of (input_data, target_data): NumPy arrays for training.
    """
    if not isinstance(df_input, pd.DataFrame) or not isinstance(df_target, pd.DataFrame):
        raise TypeError("Both `df_input` and `df_target` must be pandas DataFrames.")

    try:
        input_data = df_input.values
        target_data = df_target.values

    except Exception as e:
        raise RuntimeError(f"Failed to convert dataframes to numpy arrays: {e}")

    return input_data, target_data


def replace_nan_with_predefined_value(input_data, fill_nan_value):
    """
    Replaces NaN values in a NumPy array with a specified value.

    Parameters:
    - input_data (np.ndarray): NumPy array with potential NaN values.
    - fill_nan_value (float): Value to use to replace NaNs (default: -1.0).

    Returns:
    - np.ndarray: Cleaned array with NaNs replaced.
    """
    if not isinstance(input_data, np.ndarray):
        raise TypeError("`input_data` must be a NumPy array.")
    # Check if array is numeric
    if not np.issubdtype(input_data.dtype, np.number):
        raise TypeError(f"`input_data` must be a numeric array, got dtype {input_data.dtype}.")

    try:
        # Identify NaN locations
        nan_locations = np.isnan(input_data)
        # Replace NaN with a specified value
        input_data[nan_locations] = fill_nan_value
    except Exception as e:
        raise RuntimeError(f"Error replacing NaN values in NumPy array: {e}")

    return input_data