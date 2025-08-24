import pandas as pd

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

        # Set StoreID as the index (needed for merging later)
        store_data_encoded = store_data_encoded.set_index('StoreID')
    except Exception as e:
        # Catch any unexpected error during encoding
        raise RuntimeError(f"Error encoding store data: {e}")

    return store_data_encoded