import argparse
import mlflow
import yaml
import sys
import os

# Add parent directory to find price_imputation_and_forecasting module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from price_imputation_and_forecasting import forecasting_train_pipeline
from price_imputation_and_forecasting import price_imputation_and_forecasting


def run_pipeline(input_file_path: str):
    """
    Run the forecasting training pipeline and log the results to MLflow.
    """

    # Absolute path to YAML parameter file
    param_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", input_file_path))

    # Load input parameters
    with open(param_file_path, "r", encoding="utf-8") as f:
        input_data = yaml.safe_load(f)

    config_path = input_data.get("config_path")
    data_directory_path = input_data.get("data_directory_path")
    category = input_data.get("category")

    # Check required parameters
    if not all([category, data_directory_path, config_path]):
        print("Error: Missing required values in YAML file. Ensure 'category', 'data_directory_path', and 'config_path' are provided.")
        return
    
    # Resolve config_path to an absolute path if it is not already absolute.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if config_path:
        # remove leading slashes/backslashes to avoid absolute-root issues on Windows
        cfg_rel = config_path.lstrip("/\\")
        if not os.path.isabs(cfg_rel):
            candidate = os.path.join(project_root, cfg_rel)
            if os.path.exists(candidate):
                config_path = candidate
            else:
                # fall back to original string
                config_path = config_path

    # Initialize PriceImputationForecasting object (loads config internally)
    price_forecaster = price_imputation_and_forecasting.PriceImputationForecasting(custom_config_path=config_path)

    # Extract config parameters
    forecasting_model = price_forecaster.config.get("forecasting_model")

    # Run MLflow logging
    with mlflow.start_run() as run:
        # Set input file parameters as run tags (metadata)
        mlflow.set_tag("category", category)

        # Add run description as a tag
        run_description = f"Forecasting training for product category '{category}' using model={forecasting_model}."
        mlflow.set_tag("mlflow.note.content", run_description)

        # Log config params
        mlflow.log_param("forecasting_model", forecasting_model)

        # Run pipeline
        evaluation_metrics = forecasting_train_pipeline.run_forecasting_train_pipeline(
            config=price_forecaster.config,
            data_directory_path=data_directory_path,
            product_category_name=category
        )

        # Log evaluation metrics
        for metric, value in evaluation_metrics.items():
            mlflow.log_metric(metric, float(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the YAML input file with pipeline parameters", required=True)
    args = parser.parse_args()

    run_pipeline(args.input_file)