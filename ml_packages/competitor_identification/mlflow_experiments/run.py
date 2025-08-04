import mlflow
import os

if __name__ == "__main__":
    # Absolute path to the YAML parameter file
    param_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "param_values.yaml"))

    # Set URI to the mlflow experiments directory (where MLproject file is located)
    project_uri = os.path.abspath(os.path.dirname(__file__))

    # Define the parameters dictionary
    parameters = {
        "input_file": param_file_path
    }

    # Run the MLflow project using the API
    mlflow.projects.run(
        uri=project_uri,  # The URI of the project - current directory
        entry_point="main",  # The entry point to run
        parameters=parameters,
        experiment_name="competitor-identification"
    )