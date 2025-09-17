# Price Imputation and Forecasting Package

The `price_imputation_and_forecasting` Python package offers a complete solution for handling missing data and forecasting future prices for retail products. It leverages deep learning models, including RNN, LSTM and GRU networks, to first impute missing price points in a time series and then to forecast future prices. The package is designed for easy integration into larger data processing and analytics workflows and supports both API and command-line usage.

### Overview
In the competitive retail landscape, strategic pricing is paramount for success. However, real-world retail price data is frequently incomplete due to issues like stockouts, data collection errors, or temporary promotions. Missing data can significantly hinder the accuracy of any subsequent analysis, especially for crucial tasks like price forecasting. Inaccurate forecasts can lead to poor business decisions, such as suboptimal pricing, missed revenue opportunities, or ineffective inventory management.

This package provides an integrated solution to these challenges by combining price imputation and forecasting into a two-stage process:
1.  **Price imputation**: The first step is to address the missing data. This package uses advanced deep learning models to perform robust price imputation. By accurately filling in the gaps, we create a complete and reliable historical price series. This ensures that our forecasts are based on a comprehensive and accurate view of the past.
2.  **Price forecasting**: With a complete dataset, the package then employs another set of powerful deep learning models to forecast future price movements. Price forecasting is essential for a wide range of strategic business activities, such as anticipating market trends and proactively adjusting prices to maximize profitability, determining the best timing and pricing for promotional campaigns, and understanding how competitors might price their products in the future.

By providing robust imputation and forecasting capabilities, this package enables businesses to transform incomplete data into actionable insights, enabling more intelligent, data-driven pricing and strategic planning.


### Features
* **Data preprocessing**: Includes functionalities for cleaning, scaling, and preparing time-series data for deep learning models.
* **Imputation models**: Implements bidirectional deep learning models (Bi-RNN, Bi-LSTM, Bi-GRU) to capture temporal dependencies from both past and future data points, leading to more accurate imputations.
* **Forecasting models**: Utilizes time-series forecasting models (RNN, LSTM, GRU) to predict future prices.
* **Three Pipelines**:
    1. **Imputation model training**: To train and save a model for filling missing values.
    2. **Forecasting model training**: To train and save a model for price prediction, using imputed data.
    3. **Price prediction**: To apply the trained models to impute and then forecast prices for specific products and stores.
* **Visualization**: Generates interactive plots to visualize the original, imputed, and forecasted price series, making it easy to assess the results.
* **Configuration flexibility**: Allows users to define model architectures, hyperparameters, and other settings through YAML configuration files.


### Data
This package utilizes pricing data collected from Israeli supermarkets, made publicly available under the 'Price Transparency Law'. The data has been carefully cleaned and preprocessed to enable efficient analysis. The package uses the following data sources:
*   **Pricing data:** This dataset contains information about product prices across different stores and chains. The data is expected to be in a CSV format and should include store identifiers, product details and time series of prices.
*   **Store data:** This file contains information about each store, such as its location and other attributes.
*   **Sub-chain data:** This file contains information about each sub-chain.

Users must specify the `data_directory_path` parameter, which should point to a parent directory structured as follows:
`<data_directory_path>/retail_pricing/competitor_recognition_data/`. All required input files (pricing data, store metadata and sub-chain metadata) must be placed within the `competitor_recognition_data` subdirectory.

## Installation
Install directly from GitHub using `pip`.
```
pip install "git+https://github.com/isrkan/Competitive_Price_Intelligence.git#subdirectory=ml_packages/price_imputation_and_forecasting"
```

To verify that the package has been installed correctly, use the following command:
```
pip show price_imputation_and_forecasting
```

## Usage
The package can be used via its Python API or its command-line interface (CLI). There are three separate pipelines, each serving a different purpose.

### Using the API

#### 1. Training the imputation model
This pipeline trains a model to fill in missing price data for a given product category.

```python
from price_imputation_and_forecasting import price_imputation_and_forecasting, imputation_train_pipeline

# Define pipeline parameters
data_directory_path = "<path_to_your_data_directory>"
config_path = "/price_imputation_and_forecasting/config/<imputation_model_forecasting_model_config>.yaml"
product_category_name = "<product_category>"

# Initialize the main class with a specific configuration
price_imputer_forecaster = price_imputation_and_forecasting.PriceImputationForecasting(custom_config_path=config_path)

# Run the imputation training pipeline
evaluation_metrics = imputation_train_pipeline.run_imputation_train_pipeline(
    config=price_imputer_forecaster.config,
    data_directory_path=data_directory_path,
    product_category_name=product_category_name
)

# Print the evaluation results
print("Imputation model training evaluation:")
print(evaluation_metrics)
```

**Note:** The trained model, along with its training history, will be saved in the directory specified by the `model_save_dir_imputation` parameter in your configuration file. The saved model will be named after the `product_category_name`. For example, using the default configuration, the model for the "milk" category would be saved at `./saved_models/imputation/milk`.

#### 2. Training the forecasting model
This pipeline trains a model to forecast future prices. It uses a pre-trained imputation model to handle missing data before training.

```python
from price_imputation_and_forecasting import price_imputation_and_forecasting, forecasting_train_pipeline

# Define pipeline parameters
data_directory_path = "<path_to_your_data_directory>"
config_path = "/price_imputation_and_forecasting/config/<imputation_model_forecasting_model_config>.yaml"
product_category_name = "<product_category>"

# Initialize the main class with a specific configuration
price_imputer_forecaster = price_imputation_and_forecasting.PriceImputationForecasting(custom_config_path=config_path)

# Run the forecasting training pipeline
evaluation_metrics = forecasting_train_pipeline.run_forecasting_train_pipeline(
    config=price_imputer_forecaster.config,
    data_directory_path=data_directory_path,
    product_category_name=product_category_name
)

# Print the evaluation results
print("Forecasting model training evaluation:")
print(evaluation_metrics)
```

**Note:** The trained forecasting model is saved in the directory specified by `model_save_dir_forecasting` in the configuration file. The model file is named after the `product_category_name`. For example, with the default settings, the forecasting model for the "milk" category would be saved at `./saved_models/forecasting/milk`.

#### 3. Predicting prices (imputation + forecasting)
This pipeline applies pre-trained imputation and forecasting models to predict future prices for a specific product in a list of stores.

```python
from price_imputation_and_forecasting import price_imputation_and_forecasting, imputation_forecasting_predict_pipeline

# Define pipeline parameters
data_directory_path = "<path_to_your_data_directory>"
config_path = "/price_imputation_and_forecasting/config/<imputation_model_forecasting_model_config>.yaml"
product_category_name = "<product_category>"
store_ids = [1001, 1234]  # Example store IDs
product_description = "<product_description>"  # Specific product for prediction

# Initialize the main class with a specific configuration
price_imputer_forecaster = price_imputation_and_forecasting.PriceImputationForecasting(custom_config_path=config_path)

# Run the prediction pipeline
original_data, imputed_data, forecasted_data, figures = imputation_forecasting_predict_pipeline.run_imputation_forecasting_predict_pipeline(
    config=price_imputer_forecaster.config,
    data_directory_path=data_directory_path,
    product_category_name=product_category_name,
    store_ids=store_ids,
    product_description=product_description
)

# Display the interactive plots
for name, fig in figures.items():
    fig.show()
```


### Using the CLI
The package also provides a command-line interface (CLI) for running all the pipelines, which is ideal for automation and integration into production workflows.

#### 1. **Clone the repository**
To use the CLI, start by cloning the relevant part of the repository:
```bash
git clone --no-checkout https://github.com/isrkan/Competitive_Price_Intelligence.git
cd Competitive_Price_Intelligence
git sparse-checkout init --cone
git sparse-checkout set "ml_packages/price_imputation_and_forecasting"
git fetch
git pull
git read-tree -mu HEAD
cd "ml_packages/price_imputation_and_forecasting"
```

#### Next Step: Choose a pipeline to run
After cloning the repository, choose one of the following options based on what you would like to do:

#### 1. Training the imputation model

1.  **Create a YAML input file** (e.g., `imputation_train_input.yaml`):
    ```yaml
    data_directory_path: "<path_to_your_data_directory>"
    config_path: "/price_imputation_and_forecasting/config/<imputation_model_forecasting_model_config>.yaml"
    category: "<product_category>"
    ```
2.  **Run the pipeline from the command line:**
    ```bash
    python -m scripts.run_imputation_train_pipeline --input_file "imputation_train_input.yaml"
    ```

    * `--input_file`: Specifies the path to your YAML configuration file.

#### 2. Training the forecasting model

1.  **Create a YAML input file** (e.g., `forecasting_train_input.yaml`):
    ```yaml
    data_directory_path: "<path_to_your_data_directory>"
    config_path: "/price_imputation_and_forecasting/config/<imputation_model_forecasting_model_config>.yaml"
    category: "<product_category>"
    ```
2.  **Run the pipeline from the command line:**
    ```bash
    python -m scripts.run_forecasting_train_pipeline --input_file "forecasting_train_input.yaml"
    ```

    * `--input_file`: Specifies the path to your YAML configuration file.

#### 3. Predicting prices (imputation + forecasting)

1.  **Create a YAML input file** (e.g., `prediction_input.yaml`):
    ```yaml
    data_directory_path: "<path_to_your_data_directory>"
    config_path: "/price_imputation_and_forecasting/config/<imputation_model_forecasting_model_config>.yaml"
    category: "<product_category>"
    store_ids: [1001, 1234]  # Example store IDs
    product_description: "<product_description>"  # Specific product for prediction
    ```
2.  **Run the pipeline from the command line:**
    ```bash
    python -m scripts.run_imputation_forecasting_predict_pipeline --input_file "prediction_input.yaml"
    ```

    * `--input_file`: Specifies the path to your YAML configuration file.

All results will be printed to the console.

### Configuration files
The behavior of the pipelines is controlled by YAML configuration files located in `price_imputation_and_forecasting/config/`. The package includes the following default configurations:
*   `birnn_gru_config.yaml`: Bidirectional RNN for imputation and GRU for forecasting.
*   `bilstm_lstm_config.yaml`: Bidirectional LSTM for imputation and LSTM for forecasting.
*   `bigru_rnn_config.yaml`: Bidirectional GRU for imputation and RNN for forecasting.

You can create your own custom configuration files and pass their path to the `PriceImputationForecasting` class to experiment with different models and hyperparameters.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).