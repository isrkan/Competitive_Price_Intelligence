import argparse
import yaml
from price_imputation_and_forecasting import imputation_forecasting_predict_pipeline
from price_imputation_and_forecasting import price_imputation_and_forecasting

def main():
    """
    CLI entry point for running the imputation + forecasting prediction pipeline.
    Reads input parameters from a YAML configuration file.
    """
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(description="Run the imputation + forecasting prediction pipeline.")
    # Command-line arguments
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to the YAML input file containing pipeline parameters."
    )

    # Parse CLI arguments
    args = parser.parse_args()

    # Load parameters from YAML file
    with open(args.input_file, 'r', encoding='utf-8') as file:
        input_params = yaml.safe_load(file)

    # Extract required parameters from YAML
    category = input_params.get('category')  # Product category to predict
    data_directory_path = input_params.get('data_directory_path')  # Root directory for data
    store_ids = input_params.get('store_ids')  # List of store IDs to include in prediction
    product_description = input_params.get('product_description')  # Specific product description to filter
    config_path = input_params.get('config_path')  # Path to pipeline configuration YAML

    # Check if all required values are available
    if not all([category, data_directory_path, store_ids, product_description, config_path]):
        print("Error: Missing required values in YAML file. Ensure 'category', 'data_directory_path', 'store_ids', 'product_description', and 'config_path' are provided.")
        return

    # Initialize the price imputation and forecasting with optional custom config path
    if config_path != "/price_imputation_and_forecasting/config/birnn_gru_config.yaml":
        price_pipeline = price_imputation_and_forecasting.PriceImputationForecasting(custom_config_path=config_path)
    else:
        price_pipeline = price_imputation_and_forecasting.PriceImputationForecasting()

    try:
        # Run the imputation + forecasting prediction pipeline
        filtered_category_df, df_imputed, df_forecasted, figs = imputation_forecasting_predict_pipeline.run_imputation_forecasting_predict_pipeline(
            config=price_pipeline.config,
            data_directory_path=data_directory_path,
            product_category_name=category,
            store_ids=store_ids,
            product_description=product_description
        )

        # Output summary of results
        print(f"Prediction pipeline completed for category '{category}' and product '{product_description}'.")
        print(f"Figures generated: {len(figs)}")

        # Present the figures interactively
        for idx, fig in figs.items():
            print(f"Showing prediction plot for {idx}...")
            fig.show()

    except Exception as e:
        print(f"Error running imputation and forecasting prediction pipeline: {e}")


if __name__ == "__main__":
    main()