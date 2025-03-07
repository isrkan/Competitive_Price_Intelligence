import pandas as pd


def impute_missing_data(filtered_df):
    """
    Performs linear interpolation followed by two types of imputation:
    NOCB (Next Observation Carried Backward) and LOCB (Last Observation Carried Forward).

    Parameters:
    - filtered_df: The DataFrame containing the price data with potential missing values (NaNs).

    Returns:
    - A DataFrame with the missing data imputed.
    """
    # Interpolation - linear
    linear_fill_df = filtered_df.T.interpolate(method='linear').T

    # Imputation - first NOCB (Backward fill), then LOCB (Forward fill)
    no_cb_fill_df = linear_fill_df.T.bfill().T  # Backward fill (NOCB)
    linear_no_cb_fill_df = no_cb_fill_df.T.ffill().T  # Forward fill (LOCB)

    return linear_no_cb_fill_df