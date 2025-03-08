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
    # Scale the data for price movement (transpose and scale)
    std_scaler = StandardScaler()
    price_movement_df = pd.DataFrame(std_scaler.fit_transform(product_df.T.to_numpy()), columns=product_df.T.columns).T
    price_movement_df.columns = product_df.columns

    # Scale the data for price level (scale directly)
    std_scaler = StandardScaler()
    price_level_df = pd.DataFrame(std_scaler.fit_transform(product_df.to_numpy()), columns=product_df.columns)
    price_level_df = price_level_df.set_index(product_df.index)

    return price_movement_df, price_level_df