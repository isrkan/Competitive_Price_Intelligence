import mlflow
from mlflow.tracking import MlflowClient
import os
import yaml

if __name__ == "__main__":
    # Set URI to the mlflow experiments directory (where MLproject file is located)
    project_uri = os.path.abspath(os.path.dirname(__file__))

    # Define experiment details
    experiment_name = "price-forecasting"
    experiment_description = "Pipeline for training forecasting models on time-series pricing data"
    experiment_tags = {
        "developer": "Israel",
        "environment": "dev",
        "project": "Competitive Price Intelligence"
    }

    # Create MLflow client to manage experiments
    client = MlflowClient()

    # Get or create experiment
    experiment = client.get_experiment_by_name(experiment_name)
    # Check if the experiment already exists, otherwise create it
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        # Set tags and description only after creation
        client.set_experiment_tag(experiment_id, "mlflow.note.content", experiment_description)
        for k, v in experiment_tags.items():
            client.set_experiment_tag(experiment_id, k, v)
    else:
        # Use existing experiment ID and optionally update tags
        experiment_id = experiment.experiment_id
        client.set_experiment_tag(experiment_id, "mlflow.note.content", experiment_description)
        for k, v in experiment_tags.items():
            client.set_experiment_tag(experiment_id, k, v)

    # Define parameter file path (relative to project root)
    param_file_path = "train_param_values.yaml"
    full_param_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", param_file_path))

    # Extract values from YAML for run_name
    with open(full_param_file_path, "r", encoding="utf-8") as f:
        input_data = yaml.safe_load(f)

    category = input_data.get("category")
    config_path = ".." + input_data.get("config_path")

    # Slice config filename to extract forecasting model name
    forecasting_model = config_path[40:-12]  # Assumes fixed path prefix and .yaml suffix

    # Define run name combining relevant info name
    run_name = f"{forecasting_model}_{category}"

    # Define the parameters dictionary
    parameters = {
        "input_file": param_file_path
    }

    # Run the MLflow project using the API
    mlflow.projects.run(
        uri=project_uri,  # The URI of the project - current directory
        entry_point="forecasting",  # The entry point to run - defined in MLproject file
        parameters=parameters,  # Parameters to send to the entry point
        experiment_id=experiment_id,  # Track under this experiment
        run_name=run_name  # Name for this run
    )