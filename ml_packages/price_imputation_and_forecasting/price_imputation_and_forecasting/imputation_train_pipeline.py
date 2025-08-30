import os
from .data import load_data, clean_data, feature_engineering, preprocess_data
from .imputation import train, imputation_model_utils
from .evaluation import imputation_evaluation
from .config.config import Config


def run_imputation_train_pipeline(config, data_directory_path, product_category_name):
    """
    Run the imputation training pipeline with the provided config.

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
            nan_threshold_ratio = config.get("nan_threshold_ratio")  # Ratio threshold to drop products with too many NaNs
            fill_nan_value = config.get("fill_nan_value")  # Predefined filler for missing values
            gap_prob = config.get("gap_prob")  # Probability of simulating a missing gap
            max_gap = config.get("max_gap")  # Maximum length of simulated missing gap

            test_size = config.get("test_size")
            val_size = config.get("val_size")
            random_state = config.get("random_state")
            imputation_model = config.get('imputation_model')
            imputation_model_params = config.get('imputation_model_params')  # dict of model hyperparams

            model_save_dir_imputation = config.get("model_save_dir_imputation")  # Directory where trained model + history will be saved
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

        # Simulate random missing gaps for training (gap imputation task)
        df_sim_price_input, matrix_sim_masked_price_inputs, matrix_target_mask = preprocess_data.simulate_missing_gaps(
            df_price_input, matrix_masked_price_inputs, gap_prob, max_gap
        )

        # Scale price inputs (e.g., log the price)
        df_scaled_price_input, df_scaled_sim_price_input = feature_engineering.scale_price_inputs(df_price_input, df_sim_price_input)

        # Train/val/test split
        (
            X_sequence_train, X_static_train, target_mask_train, y_train,
            X_sequence_val,   X_static_val,   target_mask_val,   y_val,
            X_sequence_test,  X_static_test,  target_mask_test,  y_test
        ) = preprocess_data.split_train_val_test_for_imputation(
            df_scaled_sim_price_input,
            df_scaled_price_input,
            matrix_sim_masked_price_inputs,
            df_store_input,
            matrix_target_mask,
            test_size,
            val_size,
            random_state
        )

        # Train the imputation model
        model, history = train.train_imputation_model(
            imputation_model, imputation_model_params,  # model choice + hyperparams
            X_sequence_train, X_static_train, target_mask_train, y_train,
            X_sequence_val,   X_static_val,   target_mask_val,   y_val
        )

        # Save trained model and training metrics
        model_save_path = model_save_dir_imputation + product_category_name
        imputation_model_utils.save_imputation_model(model, history, model_save_path)

        # Evaluate trained model on test data
        evaluation_metrics = imputation_evaluation.evaluate_imputation_model(model, X_sequence_test, X_static_test, y_test, target_mask_test)

        return evaluation_metrics  # Return evaluation results dictionary
    except Exception as e:
        print(f"Error in run_imputation_train_pipeline: {e}")
        raise