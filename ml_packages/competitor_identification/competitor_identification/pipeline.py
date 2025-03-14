from .config.config import Config
from .data.load_data import load_store_data, load_price_data
from .data.preprocess_data import get_store_and_chain_ids, filter_nans, filter_by_specific_product_and_add_store_details, filter_by_geographic_region, data_preparation_clustering
from .data.impute_data import impute_missing_data
from .data.scale_data import scale_data
from .dimensionality_reduction.dimensionality_reduction import perform_dimensionality_reduction_on_price_movement_and_level
from .clustering.clustering import perform_clustering, find_top_competitors
from .visualization.plot_dimensionality_reduction import plot_dimensionality_reduction
from .visualization.plot_clusters import plot_clusters
from .visualization.present_competitors_table import present_competitors_table
import os


def run_pipeline(config, category, SubChainName, StoreName, product_description, Geographic):
    """
    Run the complete pipeline for competitor identification with the provided config.

    Parameters:
    - config (Config): Configuration object with pipeline settings.
    - category (str): Category of the product.
    - SubChainName (str): SubChain name.
    - StoreName (str): Store name.
    - product_description (str): Description of the product.
    - Geographic (str): Geographic region for filtering.

    Returns:
    - top_competitors (list): The top competitors identified.
    - fig1, fig2, fig3 (figures): Figures for visualization.
    """
    # Fetch the user directory token
    user_directory_token = os.getenv('USER_DIRECTORY_TOKEN')

    # Load data from config
    price_data_dir = config.get('price_data_dir').replace('<USER_DIRECTORY_TOKEN>', user_directory_token)
    store_file_path = config.get('store_file_path').replace('%price_data_dir%', price_data_dir)
    subchain_file_path = config.get('subchain_file_path').replace('%price_data_dir%', price_data_dir)
    nan_threshold = config.get('nan_threshold')
    dimensionality_reduction_method = config.get('dimensionality_reduction_method')
    dimensionality_reduction_method_params = config.get('dimensionality_reduction_method_params')
    clustering_method = config.get('clustering_method')
    clustering_method_params = config.get('clustering_method_params')
    top_n_competitors = config.get('top_n_competitors')

    # Load store data
    store_data = load_store_data(store_file_path, subchain_file_path)
    # Load price data
    price_data = load_price_data(category, price_data_dir)

    # Extract the chosen store and chain information
    chosen_store_info = get_store_and_chain_ids(store_data, SubChainName, StoreName)

    # Filter NaN observations with a custom threshold (e.g., 25% NaN values allowed)
    filtered_price_data = filter_nans(price_data, nan_threshold)

    # Impute missing data
    imputed_data = impute_missing_data(filtered_price_data)

    # Filter the product data for a specific product
    filtered_product_df = filter_by_specific_product_and_add_store_details(imputed_data, product_description, store_data)

    # Filter by geographic criteria
    product_regional_df = filter_by_geographic_region(filtered_product_df, chosen_store_info, Geographic)

    # Scaling the price movement and price level data
    scaled_price_movement_df, scaled_price_level_df = scale_data(product_regional_df)

    # Perform dimensionality reduction on both price movement and price level data
    combined_components_df, price_movement_principal_dr, price_level_principal_dr = perform_dimensionality_reduction_on_price_movement_and_level(scaled_price_movement_df, scaled_price_level_df, product_regional_df,dimensionality_reduction_method, dimensionality_reduction_method_params)

    # Plot the dimensionality reduction results for price movement and price level
    fig1 = plot_dimensionality_reduction(combined_components_df, chosen_store_info)

    # Prepare the data for clustering
    dr_components_df, store_dr_components_df, same_chain_dr_components_df = data_preparation_clustering(price_movement_principal_dr, price_level_principal_dr, chosen_store_info, product_regional_df)

    # Perform the clustering
    clustered_dr_components_df, cluster_labels, clustered_store_dr_components_df, clustered_competitors_dr_components_df = perform_clustering(dr_components_df, chosen_store_info, same_chain_dr_components_df, clustering_method, clustering_method_params)

    # Plot the clusters
    fig2 = plot_clusters(clustered_dr_components_df, cluster_labels, clustered_store_dr_components_df, clustering_method.upper())

    # Find the top N competitors based on Euclidean distance
    top_competitors = find_top_competitors(clustered_store_dr_components_df, clustered_competitors_dr_components_df, top_n=top_n_competitors)

    # Present a table displaying the top competitors
    fig3 = present_competitors_table(top_competitors)

    return top_competitors, fig1, fig2, fig3