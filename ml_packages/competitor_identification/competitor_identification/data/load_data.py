import pandas as pd


def load_store_data(store_file_path, subchain_file_path):
    """
    Loads store data from a CSV file and merges it with subchain names from an Excel file.

    Parameters:
    - store_file_path (str): Path to the store data CSV file.
    - subchain_file_path (str): Path to the subchain names Excel file.

    Returns:
    - DataFrame: Merged store data with subchain English names.
    """
    # Load store data
    store_data = pd.read_csv(store_file_path)
    store_data = store_data[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]

    # Load subchain names
    subchain_names = pd.read_excel(subchain_file_path)

    # Merge store data with subchain names based on SubChainID and SubChainName
    store_data = store_data.merge(subchain_names[['SubChainID', 'SubChainName', 'EnglishName']], on=['SubChainID', 'SubChainName'], how='left')
    # Replace SubChainName with the English name from the subchain_names file
    store_data['SubChainName'] = store_data['EnglishName']
    # Drop the EnglishName column as it is no longer needed
    store_data = store_data.drop(columns=['EnglishName'])

    return store_data


def load_price_data(category, price_data_dir):
    """
    Loads price data for a specific category from a Parquet file.

    Parameters:
    - category (str): The category name (used to find the correct Parquet file).
    - price_data_dir (str): The directory where the Parquet files are stored.

    Returns:
    - DataFrame: The price data with the index set to ['category', 'ProductDescription', 'StoreID'].
    """
    # Build the path for the Parquet file
    price_data_path = f'{price_data_dir}/{category}.parquet'

    # Load the price data from the Parquet file
    category_df = pd.read_parquet(price_data_path)

    # Set the index as ['category', 'ProductDescription', 'StoreID']
    category_df = category_df.set_index(['category', 'ProductDescription', 'StoreID'])

    return category_df