import argparse
from competitor_identification import competitor_identification
from competitor_identification import pipeline
import yaml


def main():
    """
    Main function to run the competitor identification pipeline.
    Fetches parameters from a YAML configuration file.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run the competitor identification pipeline for prediction.")

    # Command-line arguments
    parser.add_argument('--input_file', type=str, required=True, help="Path to the YAML input file with pipeline parameters.")

    # Parse arguments
    args = parser.parse_args()

    # Read configuration from the YAML file
    with open(args.input_file, 'r', encoding='utf-8') as file:
        input_params = yaml.safe_load(file)
    # Extract the config path and the data directory path from the YAML file
    config_path = input_params.get('config_path')
    data_directory_path = input_params.get('data_directory_path')

    # Extract other parameters from the YAML file
    category = input_params.get('category')
    subchain_name = input_params.get('subchain_name')
    store_name = input_params.get('store_name')
    product_description = input_params.get('product_description')
    geographic = input_params.get('geographic')

    # Check if all required values are available
    if not all([category, subchain_name, store_name, product_description, geographic]):
        print("Error: Missing required values in configuration file.")
        return

    # Initialize the churn predictor with optional custom config path
    if config_path:
        competitor_identifier = competitor_identification.CompetitorIdentification(config_path=args.config_path)
    else:
        competitor_identifier = competitor_identification.CompetitorIdentification()

    try:
        # Run the prediction pipeline
        top_competitors, fig1, fig2, fig3 = pipeline.run_pipeline(config=competitor_identifier.config,
                                                                  data_directory_path=data_directory_path,
                                                                  category=category,
                                                                  SubChainName=subchain_name,
                                                                  StoreName=store_name,
                                                                  product_description=product_description,
                                                                  Geographic=geographic)

        # Output the results
        print("Top Competitors Identified:")
        print(top_competitors)

    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()