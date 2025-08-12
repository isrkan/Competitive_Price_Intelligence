# Competitor Identification Package

The `competitor_identification` Python package provides a comprehensive solution for identifying competing retail stores offering the same product (at the barcode level) on pricing data. It applies unsupervised machine learning techniques, including dimensionality reduction and clustering, to analyze product-level pricing data and group similar stores together. This allows businesses to gain insights into their competitive landscape for a given product, understand pricing strategies of other players and make data-driven decisions. Its design supports integration into pricing intelligence and decision-support systems.

### Overview
In the competitive retail industry, accurately identifying competitors and understanding how they price their products is essential for making informed pricing and positioning decisions. Manually identifying competitor stores is labor-intensive and error-prone at scale.

This package automates competitor store identification by analyzing product-level price data at the barcode level. Through dimensionality reduction and clustering algorithms, it detects competitive store groups for each product, helping businesses monitor their competitive environment and optimize pricing strategies.

### Features
*   **Data preprocessing:** Handles missing values and scales the data for optimal model performance.
*   **Dimensionality reduction:** Reduces the dimensionality of the data using methods like PCA, t-SNE, and UMAP to visualize high-dimensional data and improve clustering performance.
*   **Clustering:** Groups similar products together using various clustering algorithms like Birch, DBSCAN, and OPTICS.
*   **Competitor identification:** Identifies the top competitors for a given product based on the clustering results.
*   **Evaluation:** Evaluates the performance of the clustering models using metrics like Silhouette Score and Davies-Bouldin Score.
*   **Visualization:** Generates plots to visualize the clusters and the results of the dimensionality reduction.

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
pip install "git+https://github.com/isrkan/Competitive_Price_Intelligence.git#subdirectory=ml_packages/competitor_identification"
```

To verify that the package has been installed correctly, use the following command:
```
pip show competitor_identification
```

## Usage

### Using the API
To integrate the competitor identification pipeline programmatically, you can use the `CompetitorIdentification` class and the `pipeline` function. Below is an example demonstrating how to use the API:
```python
from competitor_identification import competitor_identification
from competitor_identification import pipeline

# Define pipeline parameters
data_directory_path = "<path_to_your_data_directory>"
config_path = "/competitor_identification/config/<dr_method_clustering_method_config>.yaml"
category = "<product_category>"
subchain_name = "<subchain_name>"
store_name = "<store_name>"
product_description = "<product_description>"
geographic = "<geographic_level>"
include_clustering_evaluation = True

# Initialize CompetitorIdentification with default configuration
competitor_identifier = competitor_identification.CompetitorIdentification()

# Run the pipeline
top_competitors, fig1, fig2, fig3, silhouette_score, davies_bouldin_score = pipeline.run_pipeline(
    config=competitor_identifier.config,
    data_directory_path=data_directory_path,
    category=category,
    SubChainName=subchain_name,
    StoreName=store_name,
    product_description=product_description,
    Geographic=geographic,
    include_clustering_evaluation=include_clustering_evaluation
)

# Print the results
print("Top Competitors Identified:")
print(top_competitors)

# Print evaluation scores
print(f"Silhouette Score: {silhouette_score}")
print(f"Davies-Bouldin Score: {davies_bouldin_score}")
```

#### **Parameters explained**
* `data_directory_path`: The path to the directory containing the pricing and metadata files.
* `config_path`: Path to the configuration file. The package includes predefined configuration files for different dimensionality reduction and clustering methods:
  * **PCA + Birch clustering**: `"pca_birch_config.yaml"`
  * **t-SNE + DBSCAN clustering**: `"tsne_dbscan_config.yaml"`
  * **UMAP + OPTICS clustering**: `"umap_optics_config.yaml"`
* `category`: Product category.
* `subchain_name`: Name of the sub-chain in the dataset.
* `store_name`: Store name for which competitors are being identified.
* `product_description`: A description of the product.
* `geographic`: Geographic level of analysis.
* `include_clustering_evaluation`: Set to `True` to include evaluation metrics for clustering.

#### **Output**
The `run_pipeline` function returns:
* Top competitors: A list of the top competitor stores identified for the product.
* Visualizations: Plots of the dimensionality reduction and clustering results (`fig1`, `fig2`, `fig3`).
* Evaluation metrics (if requested): Silhouette score and Davies-Bouldin score for clustering evaluation.

### Using command line
The package also provides a command-line interface (CLI) for running the competitor identification pipeline, which is ideal for automation and integration into production workflows.
1. **Clone the repository** - To use the CLI, start by cloning the relevant part of the repository:
    ```bash
    git clone --no-checkout https://github.com/isrkan/Competitive_Price_Intelligence.git
    cd Competitive_Price_Intelligence
    git sparse-checkout init --cone
    git sparse-checkout set "ml_packages/competitor_identification"
    git fetch
    git pull
    git read-tree -mu HEAD
    cd "ml_packages/competitor_identification"
    ```
2.  **Create a YAML input file** (e.g., `input.yaml`) with the pipeline parameters:
    ```yaml
    data_directory_path: "<path_to_your_data_directory>"
    config_path: "/competitor_identification/config/<dr_method_clustering_method_config>.yaml"
    category: "<product_category>"
    subchain_name: "<subchain_name>"
    store_name: "<store_name>"
    product_description: "<product_description>"
    geographic: "<geographic_level>"
    include_clustering_evaluation: "True"
    ```
3.  **Run the pipeline via CLI:**
    ```bash
    python -m scripts.run_prediction_pipeline --input_file "<path_to_your_input_file>.yaml"
    ```

    * `--input_file`: Specifies the path to your YAML configuration file.

    The results will be printed to the console.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).