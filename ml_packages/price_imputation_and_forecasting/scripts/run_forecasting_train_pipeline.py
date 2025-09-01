import argparse
import yaml
from price_imputation_and_forecasting import forecasting_train_pipeline
from price_imputation_and_forecasting import price_imputation_and_forecasting

def main():
    """
    CLI entry point for running the forecasting training pipeline.
    Reads input parameters from a YAML configuration file.
    """
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(description="Run the forecasting training pipeline.")
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
    category = input_params.get('category')  # Product category to train on
    data_directory_path = input_params.get('data_directory_path')  # Root directory for data
    config_path = input_params.get('config_path')  # Path to pipeline configuration YAML

    # Check if all required values are available
    if not all([category, data_directory_path, config_path]):
        print("Error: Missing required values in YAML file. Ensure 'category', 'data_directory_path', and 'config_path' are provided.")
        return

    # Initialize the price imputation and forecasting with optional custom config path
    if config_path != "/price_imputation_and_forecasting/config/birnn_gru_config.yaml":
        price_forecaster = price_imputation_and_forecasting.PriceImputationForecasting(custom_config_path=config_path)
    else:
        price_forecaster = price_imputation_and_forecasting.PriceImputationForecasting()

    try:
        # Run the forecasting training pipeline
        evaluation_metrics = forecasting_train_pipeline.run_forecasting_train_pipeline(
            config=price_forecaster.config,
            data_directory_path=data_directory_path,
            product_category_name=category
        )

        # Print out the evaluation results
        print(f"Forecasting training completed for category '{category}'.")
        print("Evaluation metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"Error running forecasting training pipeline: {e}")


if __name__ == "__main__":
    main()