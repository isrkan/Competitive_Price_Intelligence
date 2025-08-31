import os
import pandas as pd
from .data import load_data, feature_engineering, preprocess_data
from .imputation import imputation_model_utils
from .forecasting import forecasting_model_utils
from .imputation import predict as imputation_predict
from .forecasting import predict as forecasting_predict
from .visualization import plot_predicted_time_series
from .config.config import Config


def run_imputation_forecasting_predict_pipeline(config, data_directory_path, product_category_name, store_ids, product_description):
    """
    Run a prediction pipeline: load data → impute missing values → forecast future prices
    for the given product category and store(s).

    Parameters:
    - config (Config): Configuration object with pipeline settings.
    - data_directory_path (str): Root directory for data files.
    - product_category_name (str): Name of the product category (e.g., 'rice').
    - store_ids (list[str|int]): List of store IDs to include in prediction.
    - product_description (str): Specific product description to filter

    Returns:
    - filtered_category_df (pd.DataFrame): Original filtered price data (with NaNs)
    - df_price_with_imputed_data (pd.DataFrame): Price data with missing values imputed
    - df_price_with_imputed_forecasted_data (pd.DataFrame): Price data with imputation + forecast
    - figs (dict): Plotly figure objects for each row (store-product combination)
    """
    try:
        # Validate inputs
        if not isinstance(config, Config):
            raise TypeError("Error: 'config' must be an instance of Config.")
        if not isinstance(product_category_name, str) or not product_category_name.strip():
            raise ValueError("Error: 'product_category_name' must be a non-empty string.")
        if not isinstance(store_ids, (list, tuple)) or not store_ids:
            raise ValueError("Error: 'store_ids' must be a non-empty list or tuple.")
        if not os.path.exists(data_directory_path):
            raise RuntimeError(f"Error: Data directory '{data_directory_path}' does not exist.")

        # Load configuration parameters
        price_data_dir = config.get("price_data_dir").replace("<USER_DIRECTORY_TOKEN>", data_directory_path)
        store_file_path = config.get("store_file_path").replace("%price_data_dir%", price_data_dir)
        subchain_file_path = config.get("subchain_file_path").replace("%price_data_dir%", price_data_dir)

        fill_nan_value = config.get("fill_nan_value")  # Placeholder value for missing prices
        lookback = config.get("lookback")  # Number of past timesteps for forecasting input

        # Directories where trained models are saved
        model_save_dir_imputation = config.get("model_save_dir_imputation")
        model_save_dir_forecasting = config.get("model_save_dir_forecasting")

        # Load category price data
        category_df = load_data.load_price_data(product_category_name, price_data_dir)
        if category_df.empty:
            raise ValueError(f"Error: No price data found for product category '{product_category_name}'.")

        # Load store data
        store_data = load_data.load_store_data(store_file_path, subchain_file_path)
        if store_data.empty:
            raise ValueError("Error: Loaded store data is empty.")

        # One-hot encode categorical store attributes (e.g., ChainID, DistrictName)
        store_data_with_dummies = feature_engineering.encode_store_data(store_data)

        # Filter only selected product and stores
        filtered_store_data = store_data_with_dummies.loc[store_ids]
        if filtered_store_data.empty:
            raise ValueError(f"Error: No store data found for given store_ids {store_ids}.")
        filtered_category_df = category_df[(category_df.index.get_level_values('StoreID').isin(store_ids)) & (category_df.index.get_level_values('ProductDescription').isin([product_description]))]
        if filtered_category_df.empty:
            raise ValueError(f"Error: No price data found for given store_ids {store_ids}.")

        # Align encoded store data rows with filtered product price data
        aligned_store_data_with_dummies = preprocess_data.align_encoded_store_data_with_price_data(filtered_store_data, filtered_category_df)
        # Convert pandas DataFrames into NumPy arrays for ML inputs
        df_price_input, df_store_input = preprocess_data.convert_to_numpy_inputs(filtered_category_df, aligned_store_data_with_dummies)

        # Replace NaNs with a predefined filler value and generate mask of observed vs. missing values
        df_replaced_nans_price_input, matrix_masked_price_inputs = preprocess_data.make_mask_and_replace_nan_with_predefined_value(df_price_input, fill_nan_value)

        # Scale price inputs (e.g., log the price)
        df_scaled_price_input, _ = feature_engineering.scale_price_inputs(df_replaced_nans_price_input, df_replaced_nans_price_input)

        # --- Imputation ---
        # Load pre-trained imputation model
        imputation_model_path = model_save_dir_imputation + product_category_name
        imputation_model = imputation_model_utils.load_imputation_model(imputation_model_path)

        # Prepare inputs for imputation model
        X_sequence, X_static = preprocess_data.prepare_imputation_model_inputs_for_predictions(df_scaled_price_input, matrix_masked_price_inputs, df_store_input)

        # Run imputation predictions
        y_imputation_pred = imputation_predict.run_model_prediction(imputation_model, X_sequence, X_static)
        # Fill missing entries with model predictions
        df_price_with_imputed_data = imputation_predict.replace_missing_with_predictions(df_scaled_price_input, y_imputation_pred, fill_nan_value)

        # --- Forecasting ---
        # Load pre-trained forecasting model
        forecasting_model_path = model_save_dir_forecasting + product_category_name
        forecasting_model = forecasting_model_utils.load_forecasting_model(forecasting_model_path)

        # Prepare inputs for forecasting model (last input_lookback timesteps)
        X_sequence = preprocess_data.prepare_forecasting_model_inputs_for_predictions(df_price_with_imputed_data, lookback)

        # Run forecasting predictions
        y_forecast_pred = forecasting_predict.run_forecasting_prediction(forecasting_model, X_sequence)
        # Append forecasts to imputed data
        df_price_with_imputed_forecasted_data = forecasting_predict.append_forecast_to_imputed_data(df_price_with_imputed_data, y_forecast_pred)

        # Build forecasts dates: original time index + future horizon dates
        original_dates = pd.to_datetime(filtered_category_df.columns)
        forecast_dates = pd.date_range(start=original_dates[-1] + pd.Timedelta(days=1),periods=y_forecast_pred.shape[1],freq="D")
        all_dates = original_dates.append(forecast_dates)  # Complete timeline including forecast horizon

        # Apply the inverse of scaling to recover original price values - both for imputed values
        df_price_with_imputed_data = feature_engineering.inverse_scale_price_inputs_with_expm1(df_price_with_imputed_data)
        # Round prices to nearest 10 cents
        df_price_with_imputed_data = feature_engineering.round_prices_to_nearest_10_cents(df_price_with_imputed_data)
        df_price_with_imputed_data = pd.DataFrame(df_price_with_imputed_data, index=filtered_category_df.index, columns=filtered_category_df.columns)

        # Apply the inverse of scaling to recover original price values - for forecasted values
        df_price_with_imputed_forecasted_data = feature_engineering.inverse_scale_price_inputs_with_expm1(df_price_with_imputed_forecasted_data)
        # Round prices to nearest 10 cents
        df_price_with_imputed_forecasted_data = feature_engineering.round_prices_to_nearest_10_cents(df_price_with_imputed_forecasted_data)
        df_price_with_imputed_forecasted_data = pd.DataFrame(df_price_with_imputed_forecasted_data, index=filtered_category_df.index, columns=all_dates)

        # Visualize the results for each row: Original price data with NaNs, imputed price data, Imputed + forecasted price data. Each row is visualized individually
        figs = plot_predicted_time_series.visualize_imputation_forecasting_results(filtered_category_df, df_price_with_imputed_data, df_price_with_imputed_forecasted_data)

        # Return all relevant datasets and figures
        return filtered_category_df, df_price_with_imputed_data, df_price_with_imputed_forecasted_data, figs
    except Exception as e:
        print(f"Error in run_imputation_forecasting_predict_pipeline: {e}")
        raise