import pandas as pd
import numpy as np

def encode_store_data(store_data):
    """
    Encodes categorical columns to dummies variable in the store data using one-hot encoding, using StoreID as the index.

    Parameters:
    - store_data (pd.DataFrame): Input store data.

    Returns:
    - pd.DataFrame: One-hot encoded store data with index set to the ID column.
    """
    # List of required columns for encoding
    required_columns = ['StoreID', 'ChainID', 'DistrictName', 'StoreType', 'LocationType']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in store_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in store data: {missing_columns}")

    try:
        # Keep only the relevant columns for encoding
        store_data_subset = store_data[required_columns]

        # Apply one-hot encoding (drop_first=True avoids dummy variable trap)
        store_data_encoded = pd.get_dummies(
            store_data_subset,
            columns=['ChainID', 'DistrictName', 'StoreType', 'LocationType'],
            drop_first=True
        )

        # Convert all boolean columns to integers (0/1)
        bool_cols = store_data_encoded.select_dtypes(include=['bool']).columns
        store_data_encoded[bool_cols] = store_data_encoded[bool_cols].astype(int)

        # Set StoreID as the index (needed for merging later)
        store_data_encoded = store_data_encoded.set_index('StoreID')
    except Exception as e:
        # Catch any unexpected error during encoding
        raise RuntimeError(f"Error encoding store data: {e}")

    return store_data_encoded

def scale_price_inputs(df_price_input, df_sim_price_input):
    """
    Apply log1p scaling (log(1 + x)) to both ground truth and simulated price inputs.
    Compresses large values, makes the distribution more symmetric and stabilizes variance.

    Parameters:
    - df_price_input (np.ndarray): Ground truth price values (NaN already filled).
    - df_sim_price_input (np.ndarray): Simulated price values (with artificial gaps).

    Returns:
    - df_price_input_scaled (np.ndarray)
    - df_sim_price_input_scaled (np.ndarray)
    """
    if not isinstance(df_price_input, np.ndarray) or not isinstance(df_sim_price_input, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")

    if df_price_input.shape != df_sim_price_input.shape:
        raise ValueError("df_price_input and df_sim_price_input must have the same shape.")

    try:
        # log1p transform (safe for zeros)
        df_price_input_scaled = np.log1p(df_price_input)
        df_sim_price_input_scaled = np.log1p(df_sim_price_input)
    except Exception as e:
        raise RuntimeError(f"Failed during log1p scaling: {e}")

    return df_price_input_scaled, df_sim_price_input_scaled


def inverse_scale_price_inputs_with_expm1(scaled_array):
    """
    Apply the inverse of log1p scaling (expm1) to recover original price values.

    Parameters:
    - scaled_array (np.ndarray): Array transformed with log1p.

    Returns:
    - original_array (np.ndarray): Back-transformed array with original scale.
    """
    if not isinstance(scaled_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    try:
        original_array = np.expm1(scaled_array)
    except Exception as e:
        raise RuntimeError(f"Failed during inverse expm1 scaling: {e}")

    return original_array


def round_prices_to_nearest_10_cents(prices_array):
    """
    Round each price in the DataFrame to the nearest 0.1 (10 cents).
    Prices less than 0.1 are set to 0.1 to avoid zeros.

    Parameters:
    - prices_array (np.ndarray): Array of prices.

    Returns:
    - rounded_prices (np.ndarray): Prices rounded to nearest 0.1, with minimum 0.1.
    """
    if not isinstance(prices_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    try:
        # Round to nearest 0.1
        rounded_prices = np.round(prices_array * 10) / 10.0
        # Replace any 0s with 0.1
        rounded_prices[rounded_prices == 0] = 0.1
    except Exception as e:
        raise RuntimeError(f"Failed during rounding prices. Details: {e}")

    return rounded_prices