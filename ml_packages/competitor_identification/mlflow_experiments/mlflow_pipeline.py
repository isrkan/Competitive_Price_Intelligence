import argparse
import mlflow
import yaml
import sys
import os

# Add parent directory to path to find competitor_identification module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from competitor_identification import competitor_identification
from competitor_identification import pipeline

def run_pipeline(input_file_path):
    """
    Runs the competitor identification pipeline and logs the results to MLflow.
    """

    # Absolute path to the YAML parameter file
    param_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", input_file_path))

    with open(param_file_path, "r", encoding="utf-8") as f:
        input_data = yaml.safe_load(f)

    config_path = input_data.get("config_path")
    data_directory_path = input_data.get("data_directory_path")

    category = input_data.get("category")
    subchain_name = input_data.get("subchain_name")
    store_name = input_data.get("store_name")
    product_description = input_data.get("product_description")
    geographic = input_data.get("geographic")

    # Check if all required values are available
    if not all([category, subchain_name, store_name, product_description, geographic, config_path, data_directory_path]):
        print("Error: Missing required values in configuration file.")
        return
        
    # Initialize CompetitorIdentification object (loads config internally)
    competitor_identifier = competitor_identification.CompetitorIdentification(custom_config_path=config_path)

    # Extract config parameters
    dr_method = competitor_identifier.config.get("dimensionality_reduction_method")
    clustering_method = competitor_identifier.config.get("clustering_method")
    # top_n_competitors = competitor_identifier.config.get("top_n_competitors")

    # Define custom run name
    run_name = f"{dr_method}_{clustering_method}_{category}_{subchain_name}_{store_name}_{product_description}_{geographic}"

    with mlflow.start_run() as run:
        # Set input file parameters as run tags (metadata)
        mlflow.set_tag("category", category)
        mlflow.set_tag("subchain_name", subchain_name)
        mlflow.set_tag("store_name", store_name)
        mlflow.set_tag("product_description", product_description)
        mlflow.set_tag("geographic", geographic)

        # Add run description as a tag
        run_description = (
            f"Run for identifying competitors of '{product_description}' in '{store_name}' ({subchain_name}), "
            f"geographic region '{geographic}' using {dr_method} + {clustering_method}."
        )
        mlflow.set_tag("mlflow.note.content", run_description)

        # Log config parameters
        mlflow.log_param("dimensionality_reduction_method", dr_method)
        mlflow.log_param("clustering_method", clustering_method)
        # mlflow.log_param("top_n_competitors", top_n_competitors)

        # Run pipeline
        results = pipeline.run_pipeline(
            config=competitor_identifier.config,
            data_directory_path=data_directory_path,
            category=category,
            SubChainName=subchain_name,
            StoreName=store_name,
            product_description=product_description,
            Geographic=geographic,
            include_clustering_evaluation=True,
        )
        top_competitors, fig1, fig2, fig3, silhouette_score, davies_bouldin_score = results

        # Log metrics
        mlflow.log_metric("silhouette_score", silhouette_score)
        mlflow.log_metric("davies_bouldin_score", davies_bouldin_score)

        # Log artifacts
        competitors_csv_path = "competitors.csv"
        top_competitors.to_csv(competitors_csv_path, index=False)
        mlflow.log_artifact(competitors_csv_path)

        # Log plots
        mlflow.log_figure(fig1, "dimensionality_reduction_plot.png")
        mlflow.log_figure(fig2, "clusters_plot.png")
        mlflow.log_figure(fig3, "top_competitors_table.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the YAML input file with pipeline parameters", required=True)
    args = parser.parse_args()

    # Run the pipeline
    run_pipeline(args.input_file)