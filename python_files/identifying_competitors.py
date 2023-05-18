import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
readRDS = robjects.r['readRDS']


def identify_competitors(category,product_description,SubChainName,StoreName):

    # Import the store data
    Store_data = pd.read_csv('Store_data_git.csv')
    Store_data = Store_data[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']]
    # Merge with Subchain names
    Subchain_names = pd.read_excel('SubChainNameEnglish.xlsx')
    Store_data = Store_data.merge(Subchain_names, on=['SubChainID','SubChainName'], how='left')
    Store_data['SubChainName'] = Store_data['EnglishName']
    Store_data = Store_data.drop(columns=['EnglishName'])

    # Save the store and chain IDs
    Store_input_data = Store_data[(Store_data['SubChainName']==SubChainName) & (Store_data['StoreName']==StoreName)]

    # Import the price data in one category and convert it to wide data
    category_df = pd.read_parquet(str(category+'.parquet'))
    category_df = category_df.set_index(['category','ProductDescription','StoreID'])

    # Filter observations with more than 25% nan values
    filtered_df = category_df.dropna(thresh=int(0.75*731), axis=0)

    # Interpolation - linear
    linear_fill_df = filtered_df.T.interpolate(method='linear').T
    # Imputation - first NOCB and then LOCB
    no_cb_fill_df = linear_fill_df.T.fillna(method='bfill').T
    linear_no_cb_fill_df = no_cb_fill_df.T.fillna(method='ffill').T

    ### Choose one item and filter the data
    linear_no_cb_fill_df.reset_index(inplace=True)
    product_df = linear_no_cb_fill_df[linear_no_cb_fill_df['ProductDescription'] == product_description]
    # Merge with store data
    product_df['StoreID'] = product_df['StoreID'].astype('int64')
    product_df = pd.merge(product_df, Store_data, how="left", left_on="StoreID", right_on="StoreID")
    product_df.set_index(['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','category','ProductDescription'], inplace=True)
    
    ### Scale the data for price movement
    std_scaler = StandardScaler()
    price_movement_df = pd.DataFrame(std_scaler.fit_transform(product_df.T.to_numpy()), columns=product_df.T.columns).T
    price_movement_df.columns = product_df.columns

    ### Scale the data for price level
    std_scaler = StandardScaler()
    price_level_df = pd.DataFrame(std_scaler.fit_transform(product_df.to_numpy()), columns=product_df.columns)
    price_level_df = price_level_df.set_index(product_df.index)

    #### Dimensionality reduction #####

    # Create a dictionary that allocates unique color to unique subschain
    subchain_names = product_df.reset_index()['SubChainName'].unique()
    color_map = dict(zip(subchain_names, cm.rainbow(np.linspace(0, 1, len(subchain_names)))))
    # Create a list of patches (colored rectangles) for the legend
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]

    ###### PCA #####
    ### Price movement ###
    pca = PCA()
    price_movement_principal_components  = pca.fit_transform(price_movement_df)
    price_movement_principal_df = pd.DataFrame(data = price_movement_principal_components)
    price_movement_principal_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']]
    price_movement_principal_pca = price_movement_principal_df.copy()

    ### Price level ###
    pca = PCA()
    price_level_principal_components  = pca.fit_transform(price_level_df)
    price_level_principal_df = pd.DataFrame(data = price_level_principal_components)
    price_level_principal_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']]
    price_level_principal_pca = price_level_principal_df.copy()

    ### Combination - price level and Price movement ###
    combined_components_df = pd.concat([price_movement_principal_df.iloc[:, 0],price_level_principal_df.iloc[:, 0]], axis=1)
    combined_components_df.columns = ['PrincipalPriceMovement','PrincipalPricelevel']
    combined_components_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']]


    ## plot the principal components
    plt.scatter(combined_components_df['PrincipalPriceMovement'], combined_components_df['PrincipalPricelevel'], c=combined_components_df['SubChainName'].map(color_map))
    plt.title('PCA - Price level vs. Price movement')

    plt.legend(loc='best', handles=legend_patches, bbox_to_anchor=(1, 1), title='Sub-Chain Name')
    plt.show()

    #### Cluster analysis #####

    # Data preparation for clustering 

    # Combine PCA components for price movement and price level
    pca_components_df = pd.concat([price_movement_principal_pca.iloc[:, 0:2],price_level_principal_pca.iloc[:, 0:2]], axis=1)
    pca_components_df.columns = ['P_M_1','P_M_2','P_L_1','P_L_2']
    # Add relevant columns
    pca_components_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']]
    pca_components_df.set_index(['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName'], inplace=True)

    # Filter out stores from the same chain
    pca_components_df = pca_components_df.reset_index()
    store_pca_components_df = pca_components_df[(pca_components_df['StoreID']==Store_input_data['StoreID'].iloc[0])]
    other_chains_pca_components_df = pca_components_df[pca_components_df['ChainID']!=Store_input_data['ChainID'].iloc[0]]
    pca_components_df = pd.concat([store_pca_components_df,other_chains_pca_components_df])
    pca_components_df.set_index(['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName'], inplace=True)

    # Define and fit the BIRCH model
    birch = Birch(threshold=9, n_clusters=None)
    birch.fit(pca_components_df)
    # Get the cluster labels
    pca_birch_labels = birch.labels_

    # Plot BIRCH clusters on PCA components
    plt.scatter(pca_components_df['P_M_1'], pca_components_df['P_L_1'], c=pca_birch_labels)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('BIRCH Clusters on PCA')
    plt.show()

    # Calculate Euclidean distance
    def euclidean_distance(row1, row2):
        return np.sqrt(np.sum((row1 - row2)**2))

    # Assign labels and filter data
    pca_components_df['pca_birch_labels'] = pca_birch_labels
    pca_components_df = pca_components_df.reset_index()
    store_pca_components_df = pca_components_df[(pca_components_df['StoreID']==Store_input_data['StoreID'].iloc[0])]
    competitors_pca_components_df = pca_components_df[(pca_components_df['pca_birch_labels']==store_pca_components_df['pca_birch_labels'].iloc[0]) & (pca_components_df['SubChainID']!=store_pca_components_df['SubChainID'].iloc[0])]

    # Calculate Euclidean distance for each row in competitors data
    store_row = store_pca_components_df.iloc[0][['P_M_1','P_M_2','P_L_1','P_L_2']]
    euclidean_distances = competitors_pca_components_df[['P_M_1','P_M_2','P_L_1','P_L_2']].apply(lambda row: euclidean_distance(store_row, row), axis=1)
    competitors_pca_components_df['EuclideanDistance'] = euclidean_distances
    competitors_pca_components_df = competitors_pca_components_df.sort_values(by=['EuclideanDistance'])
    # Select top 5 competitors
    top5competitors = competitors_pca_components_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']][:5].reset_index(drop=True)

    return top5competitors
