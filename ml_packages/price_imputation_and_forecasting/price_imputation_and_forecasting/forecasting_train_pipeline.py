import os
from .data import load_data, clean_data, feature_engineering, preprocess_data
from .imputation import imputation_model_utils, predict
from .forecasting import train, forecasting_model_utils
from .evaluation import forecasting_evaluation
from .config.config import Config


def run_forecasting_train_pipeline(config, data_directory_path, product_category_name):
    """
    Run the forecasting training pipeline with the provided config.

    Parameters:
    - config (Config): Configuration object with pipeline settings.
    - data_directory_path (str): Root directory for data files (replaces <USER_DIRECTORY_TOKEN>).
    - product_category_name (str): Name of the product category (e.g., 'rice').

    Returns:
    - results (dict): evaluation results of model training.
    """
    try:
        # Validate inputs
        if not isinstance(config, Config):
            raise TypeError("Error: 'config' must be an instance of Config.")
        if not isinstance(product_category_name, str) or not product_category_name.strip():
            raise ValueError("Error: 'product_category_name' must be a non-empty string.")
        if not os.path.exists(data_directory_path):
            raise RuntimeError(f"Error: Data directory '{data_directory_path}' does not exist.")

        # Load data from config
        try:
            price_data_dir = config.get("price_data_dir").replace("<USER_DIRECTORY_TOKEN>", data_directory_path)  # Replace user token in price data directory path
            # Build file paths for store and subchain data (relative to price_data_dir)
            store_file_path = config.get("store_file_path").replace("%price_data_dir%", price_data_dir)
            subchain_file_path = config.get("subchain_file_path").replace("%price_data_dir%", price_data_dir)

            # Preprocessing hyperparameters
            nan_threshold_ratio = config.get("nan_threshold_ratio")
            fill_nan_value = config.get("fill_nan_value")

            test_size = config.get("test_size")
            val_size = config.get("val_size")
            random_state = config.get("random_state")

            # Sliding window params
            lookback = config.get("lookback")
            horizon = config.get("horizon")
            min_stride = config.get("min_stride")
            max_stride = config.get("max_stride")

            forecasting_model = config.get("forecasting_model")
            forecasting_model_params = config.get("forecasting_model_params")

            model_save_dir_forecasting = config.get("model_save_dir_forecasting")
        except KeyError as e:
            raise KeyError(f"Error: Missing key in configuration: {e}")

        # Load category price data
        category_df = load_data.load_price_data(product_category_name, price_data_dir)
        if category_df.empty:
            raise ValueError(f"Error: No price data found for product category '{product_category_name}'.")

        # Load store data
        store_data = load_data.load_store_data(store_file_path, subchain_file_path)
        if store_data.empty:
            raise ValueError("Error: Loaded store data is empty.")

        ### Preprocessing steps
        # Remove products with too many missing values based on nan_threshold_ratio
        missing_filtered_price_data = clean_data.filter_missing_values_from_price_data(category_df, nan_threshold_ratio)
        # One-hot encode categorical store attributes (e.g., ChainID, DistrictName)
        store_data_with_dummies = feature_engineering.encode_store_data(store_data)
        # Align encoded store data rows with filtered product price data
        aligned_store_data_with_dummies = preprocess_data.align_encoded_store_data_with_price_data(store_data_with_dummies, missing_filtered_price_data)
        # Convert pandas DataFrames into NumPy arrays for ML inputs
        df_price_input, df_store_input = preprocess_data.convert_to_numpy_inputs(missing_filtered_price_data, aligned_store_data_with_dummies)

        # Replace NaNs with a predefined filler value and generate mask of observed vs. missing values
        df_price_input, matrix_masked_price_inputs = preprocess_data.make_mask_and_replace_nan_with_predefined_value(df_price_input, fill_nan_value)

        # Scale price inputs (e.g., log the price)
        df_scaled_price_input, _ = feature_engineering.scale_price_inputs(df_price_input, df_price_input)

        # Impute missing values using pre-trained imputation model
        model_path = config.get("model_save_dir_imputation") + product_category_name
        imputation_model = imputation_model_utils.load_imputation_model(model_path)
        # Prepare inputs for imputation model
        X_sequence, X_static = preprocess_data.prepare_imputation_model_inputs_for_predictions(df_scaled_price_input, matrix_masked_price_inputs, df_store_input)
        # Run imputation predictions
        y_pred = predict.run_model_prediction(imputation_model, X_sequence, X_static)
        # Fill missing entries with model predictions
        df_price_with_imputed_data = predict.replace_missing_with_predictions(df_scaled_price_input, y_pred, fill_nan_value)

        # Chronological split: Train / Val / Test
        train_data, val_data, test_data = preprocess_data.split_chronologically_train_val_test_for_forecasting(
            df_price_with_imputed_data, test_size, val_size
        )

        # Create sliding windows for each split
        X_train, y_train, ids_train, t0s_train = preprocess_data.create_sliding_windows(
            train_data, lookback, horizon, min_stride, max_stride, random_state
        )
        X_val, y_val, ids_val, t0s_val = preprocess_data.create_sliding_windows(
            val_data, lookback, horizon, min_stride, max_stride, random_state
        )
        X_test, y_test, ids_test, t0s_test = preprocess_data.create_sliding_windows(
            test_data, lookback, horizon, min_stride, max_stride, random_state
        )

        # Train the forecasting model
        model, history = train.train_forecasting_model(
            forecasting_model, forecasting_model_params, X_train, y_train, X_val, y_val
        )

        # Save trained model and training metrics
        save_path = model_save_dir_forecasting + product_category_name
        forecasting_model_utils.save_forecasting_model(model, history, save_path)

        # Evaluate forecasting model on test data
        evaluation_metrics = forecasting_evaluation.evaluate_forecasting_model(model, X_test, y_test)

        return evaluation_metrics  # Return evaluation results dictionary
    except Exception as e:
        print(f"Error in run_forecasting_train_pipeline: {e}")
        raise