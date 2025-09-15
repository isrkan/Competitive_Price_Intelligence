import mlflow
from mlflow.tracking import MlflowClient
import os
import yaml

if __name__ == "__main__":
    # Set URI to the mlflow experiments directory (where MLproject file is located)
    project_uri = os.path.abspath(os.path.dirname(__file__))

    # Define experiment details
    experiment_name = "price-imputation"
    experiment_description = "Pipeline for training imputation models on time-series pricing data"
    experiment_tags = {
        "developer": "Israel",
        "environment": "dev",
        "project": "Competitive Price Intelligence"
    }

    # Create MLflow client to manage experiments
    client = MlflowClient()

    # Create or get experiment
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

    # Slice config filename from path to extract imputation model name
    imputation_model = os.path.splitext(os.path.basename(config_path))[0].split("_")[0]

    # Define run name combining relevant info name
    run_name = f"{imputation_model}_{category}"

    # Define the parameters dictionary
    parameters = {
        "input_file": param_file_path
    }

    # Run the MLflow project using the API
    mlflow.projects.run(
        uri=project_uri,  # The URI of the project - current directory
        entry_point="imputation",  # The entry point to run - defined in MLproject file
        parameters=parameters,  # Parameters to send to the entry point
        experiment_id=experiment_id,  # Track under this experiment
        run_name=run_name  # Name for this run
    )