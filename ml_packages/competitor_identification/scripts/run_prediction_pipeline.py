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

    # Check if 'include_clustering_evaluation' exists in input_params and convert it to boolean if found
    if 'include_clustering_evaluation' in input_params:
        include_clustering_evaluation = input_params['include_clustering_evaluation'].lower() in ['true', '1', 't', 'y', 'yes']
    else:
        include_clustering_evaluation = None  # If it doesn't exist, don't pass it

    # Initialize the competitor identification with optional custom config path
    if config_path!="/competitor_identification/config/pca_birch_config.yaml":
        competitor_identifier = competitor_identification.CompetitorIdentification(custom_config_path=config_path)
    else:
        competitor_identifier = competitor_identification.CompetitorIdentification()

    try:
        # Run the pipeline
        if include_clustering_evaluation is not None:
            # Run the prediction pipeline
            top_competitors, fig1, fig2, fig3, silhouette_score, davies_bouldin_score = pipeline.run_pipeline(config=competitor_identifier.config,
                                                                                                              data_directory_path=data_directory_path,
                                                                                                              category=category,
                                                                                                              SubChainName=subchain_name,
                                                                                                              StoreName=store_name,
                                                                                                              product_description=product_description,
                                                                                                              Geographic=geographic,
                                                                                                              include_clustering_evaluation=include_clustering_evaluation)
        else:  # If the include_clustering_evaluation flag is not provided, skip evaluation
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

        # Optionally print silhouette score and Davies-Bouldin score if evaluation was done
        if include_clustering_evaluation is not None:
            print(f"Silhouette Score: {silhouette_score}")
            print(f"Davies-Bouldin Score: {davies_bouldin_score}")

    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()