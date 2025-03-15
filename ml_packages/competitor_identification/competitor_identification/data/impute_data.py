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
    try:
        # Check if the input is a pandas DataFrame
        if not isinstance(filtered_df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        # Check if the DataFrame is empty
        if filtered_df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Check for numeric columns (since interpolation requires numeric data)
        if not all(pd.api.types.is_numeric_dtype(filtered_df[col]) for col in filtered_df.columns):
            raise ValueError("DataFrame contains non-numeric data, which cannot be interpolated.")

        # Interpolation - linear
        linear_fill_df = filtered_df.T.interpolate(method='linear').T

        # Imputation - first NOCB (Backward fill), then LOCB (Forward fill)
        no_cb_fill_df = linear_fill_df.T.bfill().T  # Backward fill (NOCB)
        linear_no_cb_fill_df = no_cb_fill_df.T.ffill().T  # Forward fill (LOCB)

        return linear_no_cb_fill_df

    except Exception as e:
        print(f"Unexpected error during data imputation: {e}")
        raise