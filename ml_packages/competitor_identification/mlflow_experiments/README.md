# Competitor Identification MLflow Project

This MLflow project enables structured experimentation for identifying top competitors based on pricing data, using dimensionality reduction and clustering techniques. It tracks parameters, metrics, and artifacts, and logs all results to a DagsHub MLflow tracking server.

## Clone the package
Clone only the relevant module from the repository:
```bash
git clone --no-checkout https://github.com/isrkan/Competitive_Price_Intelligence.git
cd Competitive_Price_Intelligence
git sparse-checkout init --cone
git sparse-checkout set "ml_packages/competitor_identification"
git fetch
git pull
git read-tree -mu HEAD
cd "ml_packages/competitor_identification/mlflow_experiments"
```

## Prerequisites

- **Conda**: Required for managing the environment specified in `conda.yaml`.
- **MLflow**: install MLflow via Conda:
  ```bash
  conda install mlflow
  ```

### DagsHub configuration
Before running experiments, set the DagsHub credentials for MLflow tracking (Windows PowerShell example):
```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/USERNAME/REPO_NAME.mlflow"
$env:MLFLOW_TRACKING_USERNAME="DAGSHUB_USERNAME"
$env:MLFLOW_TRACKING_PASSWORD="DAGSHUB_TOKEN"
```

Replace `USERNAME`, `REPO_NAME`, and `DAGSHUB_TOKEN` with your actual DagsHub credentials and token (from DagsHub > Settings > Tokens).

## Running the MLflow project
This MLflow project follows the standard structure with:
* **`MLproject`** – defines the project entry point (`mlflow_pipeline.py`) and input parameters
* **`conda.yaml`** – specifies the Python environment and dependencies for reproducible execution
* **`mlflow_pipeline.py`** – the main pipeline script that runs the competitor identification analysis, and logs outputs to MLflow
* **`run.py`** – a wrapper script that loads parameters, initializes the experiment, and runs the project

### 1. Configure `param_values.yaml`
Create a `param_values.yaml` in the parent directory of `mlflow_experiments` to ensure relative path resolution works as expected. This file defines the inputs required for your experiment. Example:
```yaml
category: "rice"
subchain_name: "AM:PM"
store_name: "אבן גבירול (פינת ז'בוטינסקי)"
product_description: "אורז בסמטי טילדה 1 קילו"
geographic: "City"

# Path to the directory containing the price data
data_directory_path: "/absolute/path/to/data"

# Path to the DR + clustering config file
config_path: "/competitor_identification/config/pca_birch_config.yaml"

# Include clustering evaluation in the output
include_clustering_evaluation: "True"
```

### 2. Run the experiment
Ensure you are in the `mlflow_experiments` directory, then execute:
```bash
python run.py
```

## View experiment results
Visit the MLflow tracking interface on DagsHub:
```
https://dagshub.com/isrkan/Competitive_Price_Intelligence/experiments
```

There, compare runs across different dimensionality reduction and clustering methods, visualize metrics, browse artifacts and analyze parameters to evaluate what drives performance.