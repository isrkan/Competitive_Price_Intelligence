# Price Imputation and Forecasting MLflow Project

This MLflow project enables structured experimentation for imputation and forecasting models of time-series pricing data. It tracks parameters, metrics, and artifacts, and logs all results to a DagsHub MLflow tracking server.

The project includes two pipelines:
* Imputation training pipeline – trains models for imputing missing values in historical price data.
* Forecasting training pipeline – trains models to predict future prices based on historical (and imputed) series.

## Clone the package
Clone only the relevant module from the repository:

```bash
git clone --no-checkout https://github.com/isrkan/Competitive_Price_Intelligence.git
cd Competitive_Price_Intelligence
git sparse-checkout init --cone
git sparse-checkout set "ml_packages/price_imputation_and_forecasting"
git fetch
git pull
git read-tree -mu HEAD
cd "ml_packages/price_imputation_and_forecasting/mlflow_experiments"
```

## Prerequisites
* **Conda**: Required for managing the environment specified in `conda.yaml`.
* **MLflow**: Install MLflow via Conda:
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
This MLflow project follows the standard MLflow structure with:
* **`MLproject`** – defines the project entry points (`mlflow_imputation_pipeline.py` and `mlflow_forecasting_pipeline.py`) and their input parameters.
* **`conda.yaml`** – specifies the Python environment and dependencies for reproducible execution.
* **`mlflow_imputation_pipeline.py`** – runs the imputation training pipeline and logs outputs to MLflow.
* **`mlflow_forecasting_pipeline.py`** – runs the forecasting training pipeline and logs outputs to MLflow.
* **`run_imputation.py`** – wrapper script to load parameters and launch the imputation experiment.
* **`run_forecasting.py`** – wrapper script to load parameters and launch the forecasting experiment.

### 1. Configure parameter files
Create a `train_param_values.yaml` in the parent directory of `mlflow_experiments` to ensure relative path resolution works as expected. This file defines the inputs required for your experiment. Example (for imputation and forecasting):

```yaml
category: "rice"

# Path to the directory containing the price data
data_directory_path: "/absolute/path/to/data"

# Path to the imputation and forecasting config file
config_path: "/price_imputation_and_forecasting/config/birnn_gru_config.yaml"
```

### 2. Run the experiments
Navigate to the `mlflow_experiments` directory and run one of the pipelines:

#### Run the imputation training experiment:
```bash
python run_imputation.py
```

#### Run the forecasting training experiment:
```bash
python run_forecasting.py
```

## View experiment results
Visit the MLflow tracking interface on DagsHub:
```
https://dagshub.com/isrkan/Competitive_Price_Intelligence/experiments
```

There, you can:
* Compare runs across different models and categories.
* Visualize metrics.
* Browse artifacts (plots, configs).
* Analyze parameters to evaluate what drives performance.