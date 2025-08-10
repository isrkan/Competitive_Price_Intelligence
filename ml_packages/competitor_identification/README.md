# Competitor Identification Package

The `competitor_identification` Python package provides a comprehensive solution for identifying competing retail stores offering the same product (at the barcode level) on pricing data. It applies unsupervised machine learning techniques, including dimensionality reduction and clustering, to analyze product-level pricing data and group similar stores together. This allows businesses to gain insights into their competitive landscape for a given product, understand pricing strategies of other players and make data-driven decisions. Its design supports integration into pricing intelligence and decision-support systems.

### Problem statement
In the competitive retail industry, accurately identifying competitors and understanding how they price their products is essential for making informed pricing and positioning decisions. Manually identifying competitor stores is labor-intensive and error-prone at scale.

This package automates competitor store identification by analyzing product-level price data at the barcode level. Through dimensionality reduction and clustering algorithms, it detects competitive store groups for each product, helping businesses monitor their competitive environment and optimize pricing strategies.

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