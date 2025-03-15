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
    try:
        if not isinstance(Store_data, pd.DataFrame):
            raise TypeError("Store_data must be a pandas DataFrame.")
        if not isinstance(SubChainName, str) or not isinstance(StoreName, str):
            raise TypeError("SubChainName and StoreName must be strings.")

        chosen_store_info = Store_data[(Store_data['SubChainName'] == SubChainName) & (Store_data['StoreName'] == StoreName)]
        return chosen_store_info

    except Exception as e:
        print(f"Error extracting store and chain IDs: {e}")
        raise


def filter_nans(category_df, nan_threshold):
    """
    Filters observations from category_df where the number of NaN values exceeds a given threshold.

    Parameters:
    - category_df: The DataFrame containing the category price data.
    - nan_threshold: The threshold for allowed NaN values. For example, a value of 0.75 means 25% NaN values are allowed.

    Returns:
    - The filtered DataFrame with rows containing fewer NaN values than the threshold.
    """
    try:
        if not isinstance(category_df, pd.DataFrame):
            raise TypeError("category_df must be a pandas DataFrame.")
        if not isinstance(nan_threshold, (float, int)) or not (0 <= nan_threshold <= 1):
            raise ValueError("nan_threshold must be a float between 0 and 1.")
        if category_df.empty:
            raise ValueError("category_df DataFrame is empty.")

        # Calculate the threshold for non-NaN values
        threshold_value = int((1 - nan_threshold) * category_df.shape[1])

        # Filter out rows where NaN values exceed the threshold
        filtered_df = category_df.dropna(thresh=threshold_value, axis=0)
        return filtered_df

    except Exception as e:
        print(f"Error filtering NaN values: {e}")
        raise


def filter_by_specific_product_and_add_store_details(linear_no_cb_fill_df, product_description, Store_data):
    """
    Filter the product data for a specific product description and merge it with store data.

    Parameters:
    - linear_no_cb_fill_df: The DataFrame containing the price data.
    - product_description: The product description to filter the data by.
    - Store_data: The DataFrame containing the store data.

    Returns:
    - A DataFrame with filtered product data merged with store data.
    """
    try:
        if not isinstance(linear_no_cb_fill_df, pd.DataFrame):
            raise TypeError("linear_no_cb_fill_df must be a pandas DataFrame.")
        if not isinstance(product_description, str):
            raise TypeError("product_description must be a string.")
        if not isinstance(Store_data, pd.DataFrame):
            raise TypeError("Store_data must be a pandas DataFrame.")
        if linear_no_cb_fill_df.empty:
            raise ValueError("linear_no_cb_fill_df DataFrame is empty.")
        if Store_data.empty:
            raise ValueError("Store_data DataFrame is empty.")

        # Reset the index to filter data correctly
        linear_no_cb_fill_df.reset_index(inplace=True)

        # Filter the data by product description
        product_df = linear_no_cb_fill_df[linear_no_cb_fill_df['ProductDescription'] == product_description].copy()

        if product_df.empty:
            raise ValueError("No matching product found in linear_no_cb_fill_df.")

        # Ensure StoreID is in the correct type for merging
        product_df['StoreID'] = product_df['StoreID'].astype('int64')
        # Merge with the store data
        product_df = pd.merge(product_df, Store_data, how="left", on="StoreID")

        return product_df

    except Exception as e:
        print(f"Error filtering product data and merging store details: {e}")
        raise

def filter_by_geographic_region(product_df, chosen_store_info, Geographic):
    """
    Filter the product data based on geographic criteria such as City, County, or District.

    Parameters:
    - product_df: The DataFrame containing the filtered product data.
    - Store_input_data: A DataFrame containing the store-related data for comparison (City, SubDistrict, etc.).
    - Geographic: A string representing the geographic region type (City, County, or District).

    Returns:
    - A DataFrame with further filtered data based on the geographic region.
    """
    # Filter by region (City, County, District)
    try:
        if not isinstance(product_df, pd.DataFrame):
            raise TypeError("product_df must be a pandas DataFrame.")
        if not isinstance(chosen_store_info, pd.DataFrame):
            raise TypeError("chosen_store_info must be a pandas DataFrame.")
        if not isinstance(Geographic, str):
            raise TypeError("Geographic must be a string.")
        if product_df.empty:
            raise ValueError("product_df is empty.")
        if chosen_store_info.empty:
            raise ValueError("chosen_store_info is empty.")

        if Geographic == 'City':
            if len(product_df[product_df['CityName'] == chosen_store_info['CityName'].iloc[0]]) > 10:
                product_df = product_df[product_df['CityName'] == chosen_store_info['CityName'].iloc[0]]
            else:
                Geographic = 'County'

        if Geographic == 'County':
            if len(product_df[product_df['SubDistrictName'] == chosen_store_info['SubDistrictName'].iloc[0]]) > 10:
                product_df = product_df[product_df['SubDistrictName'] == chosen_store_info['SubDistrictName'].iloc[0]]
            else:
                Geographic = 'District'

        if Geographic == 'District':
            if len(product_df[product_df['DistrictName'] == chosen_store_info['DistrictName'].iloc[0]]) > 10:
                product_df = product_df[product_df['DistrictName'] == chosen_store_info['DistrictName'].iloc[0]]

        # Set the index as required
        product_df.set_index(['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName', 'category', 'ProductDescription'], inplace=True)
        return product_df

    except Exception as e:
        print(f"Error filtering by geographic region: {e}")
        raise


def data_preparation_clustering(price_movement_principal, price_level_principal, chosen_store_info, product_df):
    """
    Prepare the data for clustering by combining dimensionality reduction components for price movement and price level, and filtering based on the chosen store and chain.

    Parameters:
    - price_movement_principal_dr: DataFrame containing the principal components for price movement.
    - price_level_principal_dr: DataFrame containing the principal components for price level.
    - chosen_store_info: DataFrame containing the metadata of the store to be used for filtering.
    - product_df: DataFrame containing the product metadata, including ChainID, StoreID, etc.

    Returns:
    - dr_components_df: DataFrame containing combined dimensionality reduction components for price movement and price level with metadata.
    - store_dr_components_df: DataFrame with dimensionality reduction components for the input store.
    - same_chain_dr_components_df: DataFrame with dimensionality reduction components for stores in the same chain as the input store.
    """
    try:
        if not all(isinstance(df, pd.DataFrame) for df in [price_movement_principal, price_level_principal, chosen_store_info, product_df]):
            raise TypeError("All inputs must be pandas DataFrames.")
        if any(df.empty for df in [price_movement_principal, price_level_principal, chosen_store_info, product_df]):
            raise ValueError("None of the input DataFrames can be empty.")

        # Combine dimensionality reduction components for price movement and price level
        dr_components_df = pd.concat([price_movement_principal.iloc[:, 0:2], price_level_principal.iloc[:, 0:2]], axis=1)
        dr_components_df.columns = ['P_M_1', 'P_M_2', 'P_L_1', 'P_L_2']

        # Add relevant columns from store metadata
        dr_components_df = dr_components_df.reset_index()
        required_columns = ['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']
        for col in required_columns:
            if col not in product_df.reset_index().columns:
                raise KeyError(f"Missing required column in product_df: {col}")
        dr_components_df[required_columns] = product_df.reset_index()[required_columns]

        # Filter for the input store based on StoreID and ChainID
        store_dr_components_df = dr_components_df[dr_components_df['StoreID'] == chosen_store_info['StoreID'].iloc[0]]
        # Filter for stores from the same chain, excluding the input store
        same_chain_dr_components_df = dr_components_df[(dr_components_df['ChainID'] == chosen_store_info['ChainID'].iloc[0]) & (dr_components_df['StoreID'] != chosen_store_info['StoreID'].iloc[0])]
        # Filter for stores from other chains
        other_chains_dr_components_df = dr_components_df[dr_components_df['ChainID'] != chosen_store_info['ChainID'].iloc[0]]

        # Combine the data: input store and stores from other chains
        dr_components_df = pd.concat([store_dr_components_df, other_chains_dr_components_df])
        # Set index again for the final output
        dr_components_df.set_index(['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName'], inplace=True)

        return dr_components_df, store_dr_components_df, same_chain_dr_components_df

    except Exception as e:
        print(f"Error in data preparation for clustering: {e}")
        raise