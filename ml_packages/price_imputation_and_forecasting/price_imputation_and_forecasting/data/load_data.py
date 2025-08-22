import pandas as pd
import os

def load_store_data(store_file_path, subchain_file_path):
    """
    Loads store data from a CSV file and merges it with subchain names from an Excel file.

    Parameters:
    - store_file_path (str): Path to the store data CSV file.
    - subchain_file_path (str): Path to the subchain names Excel file.

    Returns:
    - DataFrame: Merged store data with subchain English names.
    """
    try:
        if not os.path.exists(store_file_path):
            raise FileNotFoundError(f"Store file not found: {store_file_path}")
        if not os.path.exists(subchain_file_path):
            raise FileNotFoundError(f"Subchain file not found: {subchain_file_path}")

        # Load store data
        try:
            store_data = pd.read_csv(store_file_path)
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing CSV file: {store_file_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading CSV file {store_file_path}: {e}")

        # Check required columns are present in store data
        required_store_columns = {'ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName', 'StoreType', 'LocationType'}
        if not required_store_columns.issubset(store_data.columns):
            missing_columns = required_store_columns - set(store_data.columns)
            raise ValueError(f"Missing columns in store data: {missing_columns}")
        store_data = store_data[list(required_store_columns)]

        # Load subchain names
        try:
            subchain_names = pd.read_excel(subchain_file_path)
        except ValueError:
            raise ValueError(f"Error reading Excel file: {subchain_file_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading Excel file {subchain_file_path}: {e}")

        # Check required columns in subchain file
        required_subchain_columns = {'SubChainID', 'SubChainName', 'EnglishName'}
        if not required_subchain_columns.issubset(subchain_names.columns):
            missing_columns = required_subchain_columns - set(subchain_names.columns)
            raise ValueError(f"Missing columns in subchain names file: {missing_columns}")

        # Merge store data with subchain names based on SubChainID and SubChainName
        try:
            store_data = store_data.merge(subchain_names[['SubChainID', 'SubChainName', 'EnglishName']], on=['SubChainID', 'SubChainName'], how='left')
        except KeyError as e:
            raise ValueError(f"Key error during merge: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error merging store data with subchain names: {e}")

        # Replace SubChainName with the English name from the subchain_names file
        store_data['SubChainName'] = store_data['EnglishName']
        # Drop the EnglishName column as it is no longer needed
        store_data = store_data.drop(columns=['EnglishName'])

        return store_data

    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"Error loading store data: {e}")
        raise


def load_price_data(category, price_data_dir):
    """
    Loads price data for a specific category from a Parquet file.

    Parameters:
    - category (str): The category name (used to find the correct Parquet file).
    - price_data_dir (str): The directory where the Parquet files are stored.

    Returns:
    - DataFrame: The price data with the index set to ['category', 'ProductDescription', 'StoreID'].
    """
    try:
        # Validate price_data_dir path
        price_data_path = os.path.join(price_data_dir, f"{category}.parquet")
        if not os.path.exists(price_data_path):
            raise FileNotFoundError(f"Price data file not found: {price_data_path}")

        # Load the price data from the Parquet file
        try:
            category_df = pd.read_parquet(price_data_path)
        except ValueError:
            raise ValueError(f"Error reading Parquet file: {price_data_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading Parquet file {price_data_path}: {e}")

        # Set the index as ['category', 'ProductDescription', 'StoreID']
        category_df = category_df.set_index(['category', 'ProductDescription', 'StoreID'])

        return category_df

    except (FileNotFoundError, ValueError, NotADirectoryError, RuntimeError) as e:
        print(f"Error loading price data: {e}")
        raise