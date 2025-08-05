import mlflow
from mlflow.tracking import MlflowClient
import os
import yaml

if __name__ == "__main__":
    # Set URI to the mlflow experiments directory (where MLproject file is located)
    project_uri = os.path.abspath(os.path.dirname(__file__))

    # Define experiment details
    experiment_name = "competitor-identification"
    experiment_description = "Pipeline for identifying top competitors based on pricing data"
    experiment_tags = {
        "developer": "Israel",
        "environment": "dev",
        "project": "Competitive Price Intelligence"
    }

    # Create MLflow client
    client = MlflowClient()

    # Get or create experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        # Set tags and description only after creation
        client.set_experiment_tag(experiment_id, "mlflow.note.content", experiment_description)
        for key, value in experiment_tags.items():
            client.set_experiment_tag(experiment_id, key, value)
    else:
        experiment_id = experiment.experiment_id
        # Optionally update tags even if it exists (safe to do)
        client.set_experiment_tag(experiment_id, "mlflow.note.content", experiment_description)
        for key, value in experiment_tags.items():
            client.set_experiment_tag(experiment_id, key, value)

    # Define parameter file path (relative to project root)
    param_file_path = "param_values.yaml"
    full_param_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", param_file_path))

    # Extract values from YAML for run_name
    with open(full_param_file_path, "r", encoding="utf-8") as f:
        input_data = yaml.safe_load(f)

    category = input_data.get("category")
    subchain = input_data.get("subchain_name")
    store = input_data.get("store_name")
    product = input_data.get("product_description")
    geo = input_data.get("geographic")
    config_path = input_data.get("config_path")

    # Slice config filename from path
    config_key = config_path[36:-5]  # Assumes fixed path prefix and .yaml suffix

    if config_key == "default_config":
        dr_method = "pca"
        clustering_method = "birch"
    elif config_key == "tsne_dbscan_config":
        dr_method = "tsne"
        clustering_method = "dbscan"
    elif config_key == "umap_optics_config":
        dr_method = "umap"
        clustering_method = "optics"
    else:
        raise ValueError(f"Unknown config key: {config_key}")


    # Build the run name
    run_name = f"{dr_method}_{clustering_method}_{category}_{subchain}_{store}_{product}_{geo}"

    # Define the parameters dictionary
    parameters = {
        "input_file": param_file_path
    }

    # Run the MLflow project using the API
    mlflow.projects.run(
        uri=project_uri,  # The URI of the project - current directory
        entry_point="main",  # The entry point to run
        parameters=parameters,
        experiment_id=experiment_id,
        run_name=run_name
    )