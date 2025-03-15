from sklearn.preprocessing import StandardScaler
import pandas as pd


def scale_data(product_df):
    """
    Scales the product price data for both price movement (changes) and price level (absolute prices).

    Parameters:
    - product_df: DataFrame containing the product price data, where columns represent time periods and rows represent products.

    Returns:
    - price_movement_df: DataFrame with scaled price movement data (normalized changes in price).
    - price_level_df: DataFrame with scaled price level data (normalized absolute prices).
    """
    try:
        if not isinstance(product_df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if product_df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Check for numeric columns
        if not all(pd.api.types.is_numeric_dtype(product_df[col]) for col in product_df.columns):
            raise ValueError("DataFrame contains non-numeric data, which cannot be scaled.")

        # Scale the data for price movement (transpose and scale)
        try:
            std_scaler = StandardScaler()
            price_movement_df = pd.DataFrame(std_scaler.fit_transform(product_df.T.to_numpy()), columns=product_df.T.columns).T
            price_movement_df.columns = product_df.columns
        except Exception as e:
            raise ValueError(f"Error during price movement scaling: {e}")

        # Scale the data for price level (scale directly)
        try:
            std_scaler = StandardScaler()
            price_level_df = pd.DataFrame(std_scaler.fit_transform(product_df.to_numpy()), columns=product_df.columns)
            price_level_df = price_level_df.set_index(product_df.index)
        except Exception as e:
            raise ValueError(f"Error during price level scaling: {e}")

        return price_movement_df, price_level_df

    except Exception as e:
        print(f"Error during data scaling: {e}")
        raise