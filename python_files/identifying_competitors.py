import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch


def preprocess_data(category, product_description, SubChainName, StoreName, Geographic=None):
    # Import the store data
    Store_data = pd.read_csv('competitor_recognition_data/Store_data_git.csv')
    Store_data = Store_data[['ChainID', 'ChainName', 'SubChainID', 'SubChainName', 'StoreID', 'StoreName', 'DistrictName', 'SubDistrictName', 'CityName']]
    
    # Merge with subchain names
    subchain_names = pd.read_excel('competitor_recognition_data/SubChainNameEnglish.xlsx')
    Store_data = Store_data.merge(subchain_names, on=['SubChainID', 'SubChainName'], how='left')
    Store_data['SubChainName'] = Store_data['EnglishName']
    Store_data = Store_data.drop(columns=['EnglishName'])

    # Take the store and chain IDs
    Store_input_data = Store_data[(Store_data['SubChainName'] == SubChainName) & (Store_data['StoreName'] == StoreName)]

    # Import the price data in one category
    category_df = pd.read_parquet(f'competitor_recognition_data/{category}.parquet')
    category_df = category_df.set_index(['category', 'ProductDescription', 'StoreID'])
    # Filter observations with more than 25% nan values
    filtered_df = category_df.dropna(thresh=int(0.75 * 731), axis=0)
    
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
    
    # Filter by region
    if Geographic=='City':
        if len(product_df[product_df['CityName']==Store_input_data['CityName'].iloc[0]])>10:
            product_df = product_df[product_df['CityName']==Store_input_data['CityName'].iloc[0]]
        else:
            Geographic=='County'

    if Geographic=='County':
        if len(product_df[product_df['SubDistrictName']==Store_input_data['SubDistrictName'].iloc[0]])>10:
            product_df = product_df[product_df['SubDistrictName']==Store_input_data['SubDistrictName'].iloc[0]]
        else:
            Geographic=='District'

    if Geographic=='District':
        if len(product_df[product_df['DistrictName']==Store_input_data['DistrictName'].iloc[0]])>10:
            product_df = product_df[product_df['DistrictName']==Store_input_data['DistrictName'].iloc[0]]
            
    product_df.set_index(['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName','category','ProductDescription'], inplace=True)
    return product_df, Store_input_data


def scale_data(product_df):
    ### Scale the data for price movement
    std_scaler = StandardScaler()
    price_movement_df = pd.DataFrame(std_scaler.fit_transform(product_df.T.to_numpy()), columns=product_df.T.columns).T
    price_movement_df.columns = product_df.columns
    ### Scale the data for price level
    std_scaler = StandardScaler()
    price_level_df = pd.DataFrame(std_scaler.fit_transform(product_df.to_numpy()), columns=product_df.columns)
    price_level_df = price_level_df.set_index(product_df.index)

    return price_movement_df, price_level_df


def pca(price_movement_df, price_level_df, product_df):
    ### Price movement ###
    pca = PCA()
    price_movement_principal_components  = pca.fit_transform(price_movement_df)
    price_movement_principal_df = pd.DataFrame(data = price_movement_principal_components)
    price_movement_principal_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']]
    price_movement_principal_pca = price_movement_principal_df.copy()

    ### Price level ###
    pca = PCA()
    price_level_principal_components  = pca.fit_transform(price_level_df)
    price_level_principal_df = pd.DataFrame(data = price_level_principal_components)
    price_level_principal_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']]
    price_level_principal_pca = price_level_principal_df.copy()

    ### Combination - price level and Price movement ###
    combined_components_df = pd.concat([price_movement_principal_df.iloc[:, 0],price_level_principal_df.iloc[:, 0]], axis=1)
    combined_components_df.columns = ['PrincipalPriceMovement','PrincipalPricelevel']
    combined_components_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']]

    return combined_components_df, price_movement_principal_pca, price_level_principal_pca


def plot_dimensionality_reduction(combined_components_df, Store_input_data):
    ## plot the principal components
    fig1 = px.scatter(combined_components_df, x='PrincipalPriceMovement', y='PrincipalPricelevel', color='SubChainName', hover_data=['ChainName','SubChainName', 'StoreName'])
    # Update the color of points matching the store
    fig1.add_trace(go.Scatter(
        x=combined_components_df[combined_components_df['StoreID'] == Store_input_data['StoreID'].iloc[0]]['PrincipalPriceMovement'],
        y=combined_components_df[combined_components_df['StoreID'] == Store_input_data['StoreID'].iloc[0]]['PrincipalPricelevel'],
        mode='markers',
        text=combined_components_df[combined_components_df['StoreID'] == Store_input_data['StoreID'].iloc[0]]['SubChainName'] + '<br>' + combined_components_df[combined_components_df['StoreID'] == Store_input_data['StoreID'].iloc[0]]['StoreName'],
        hovertemplate='%{text}',
        marker=dict(size=6,color='rgba(0,0,0,0)',line=dict(color='black', width=1.2)),
        showlegend=False
    ))
    # Bold the input subchain
    fig1.for_each_trace(lambda t: t.update(name=t.name.replace(Store_input_data['SubChainName'].iloc[0], '<b>' + Store_input_data['SubChainName'].iloc[0] + '</b>')) if t.name == Store_input_data['SubChainName'].iloc[0] else t)
    fig1.update_layout(
        title='Market Structure: Pricing Similarity in Terms of Price Level and Price Movement',
        legend=dict(title='Sub-Chain Name',yanchor="top",y=0.99,xanchor="left",x=1)
    )
    #fig.show()
    pio.write_html(fig1, 'static/market_structure_analysis.html')
    return fig1


def data_preparation_clustering(price_movement_principal_pca, price_level_principal_pca, Store_input_data, product_df):
    # Combine PCA components for price movement and price level
    pca_components_df = pd.concat([price_movement_principal_pca.iloc[:, 0:2],price_level_principal_pca.iloc[:, 0:2]], axis=1)
    pca_components_df.columns = ['P_M_1','P_M_2','P_L_1','P_L_2']
    # Add relevant columns
    pca_components_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']] = product_df.reset_index()[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName']]
    pca_components_df.set_index(['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName'], inplace=True)

    # Filter out stores from the same chain
    pca_components_df = pca_components_df.reset_index()
    store_pca_components_df = pca_components_df[(pca_components_df['StoreID']==Store_input_data['StoreID'].iloc[0])]
    same_chain_pca_components_df = pca_components_df[(pca_components_df['ChainID']==Store_input_data['ChainID'].iloc[0]) & (pca_components_df['StoreID']!=Store_input_data['StoreID'].iloc[0])]
    other_chains_pca_components_df = pca_components_df[pca_components_df['ChainID']!=Store_input_data['ChainID'].iloc[0]]
    pca_components_df = pd.concat([store_pca_components_df,other_chains_pca_components_df])
    pca_components_df.set_index(['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName','DistrictName','SubDistrictName','CityName'], inplace=True)
    
    return pca_components_df, store_pca_components_df, same_chain_pca_components_df


def birch_clustering(pca_components_df, Store_input_data, same_chain_pca_components_df):
    
    # Define and fit the BIRCH model
    birch = Birch(threshold=9, n_clusters=None)
    birch.fit(pca_components_df)
    # Get the cluster labels
    pca_birch_labels = birch.labels_
    pca_components_df['pca_birch_labels'] = pca_birch_labels
    # Filter data
    pca_components_df = pca_components_df.reset_index()
    store_pca_components_df = pca_components_df[(pca_components_df['StoreID']==Store_input_data['StoreID'].iloc[0])]
    competitors_pca_components_df = pca_components_df[(pca_components_df['pca_birch_labels']==store_pca_components_df['pca_birch_labels'].iloc[0]) & (pca_components_df['SubChainID']!=store_pca_components_df['SubChainID'].iloc[0])]
    if len(competitors_pca_components_df)==0:
        competitors_pca_components_df = pca_components_df[pca_components_df['SubChainID']!=store_pca_components_df['SubChainID'].iloc[0]]
    same_chain_pca_components_df['pca_birch_labels'] = max(np.unique(pca_birch_labels))+1
    
    pca_components_df = pd.concat([pca_components_df,same_chain_pca_components_df])
    pca_birch_labels = np.concatenate((pca_birch_labels, same_chain_pca_components_df['pca_birch_labels'].to_numpy() ))
    
    return pca_components_df, pca_birch_labels, store_pca_components_df, competitors_pca_components_df


def plot_clusters(pca_components_df, pca_birch_labels, store_pca_components_df):
    fig2 = go.Figure()
    # Add points of all clusters
    cluster_mask = ((pca_components_df['pca_birch_labels'] != max(np.unique(pca_birch_labels))) & (pca_components_df['pca_birch_labels'] != store_pca_components_df['pca_birch_labels'].iloc[0]))
    fig2.add_trace(go.Scatter(
        x=pca_components_df[cluster_mask]['P_M_1'],
        y=pca_components_df[cluster_mask]['P_L_1'],
        mode='markers',
        text='Chain: ' + pca_components_df[cluster_mask]['ChainName'] 
                + '<br> SubChain: ' + pca_components_df[cluster_mask]['SubChainName'] 
                + '<br> Store: ' + pca_components_df[cluster_mask]['StoreName'] 
                + '<br> Cluster: ' + pca_components_df[cluster_mask]['pca_birch_labels'].astype(str),
        hovertemplate='%{text}',
        marker=dict(
            color=pca_birch_labels[cluster_mask],
            colorscale='Rainbow',
            opacity=0.7
        ),
        showlegend=False
    ))

    # Add marks on stores from the same chain
    cluster_mask = pca_components_df['pca_birch_labels'] == max(np.unique(pca_birch_labels))
    fig2.add_trace(go.Scatter(
        x=pca_components_df[cluster_mask]['P_M_1'],
        y=pca_components_df[cluster_mask]['P_L_1'],
        mode='markers',
        text='Chain: ' + pca_components_df[cluster_mask]['ChainName'] 
                + '<br> SubChain: ' + pca_components_df[cluster_mask]['SubChainName'] 
                + '<br> Store: ' + pca_components_df[cluster_mask]['StoreName'] 
                + '<br> Cluster: Same Chain',
        hovertemplate='%{text}',
        marker=dict(color='gray', opacity=1, size=8),
        showlegend=False
    ))

    # Add bolder points for the specific cluster
    cluster_mask = pca_components_df['pca_birch_labels'] == store_pca_components_df['pca_birch_labels'].iloc[0]
    fig2.add_trace(go.Scatter(
        x=pca_components_df[cluster_mask]['P_M_1'],
        y=pca_components_df[cluster_mask]['P_L_1'],
        mode='markers',
        text='Chain: ' + pca_components_df[cluster_mask]['ChainName'] 
                + '<br> SubChain: ' + pca_components_df[cluster_mask]['SubChainName'] 
                + '<br> Store: ' + pca_components_df[cluster_mask]['StoreName'] 
                + '<br> Cluster: ' + pca_components_df[cluster_mask]['pca_birch_labels'].astype(str),
        hovertemplate='%{text}',
        marker=dict(color=pca_birch_labels[cluster_mask], colorscale='Rainbow', opacity=0.7, size=8, line=dict(color='black', width=1.5)),
        showlegend=False
    ))

    # Add arrow annotation
    fig2.add_annotation(
        x=store_pca_components_df['P_M_1'].iloc[0],
        y=store_pca_components_df['P_L_1'].iloc[0],
        ax=-60,
        ay=60,
        text='The chosen store',
        showarrow=True,
        arrowhead=2,
        arrowwidth=2,
        arrowcolor='black'
    )

    fig2.update_layout(
        xaxis_title='Principal Price Movement',
        yaxis_title='Principal Price Level',
        title='Store Clustering: Pricing Similarity using BIRCH Algorithm'
    )
    #fig.show()
    pio.write_html(fig2, 'static/store_clustering.html')
    return fig2


def find_top_competitors(store_pca_components_df, competitors_pca_components_df):
    # Calculate Euclidean distance
    def euclidean_distance(row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    # Calculate Euclidean distance for each row in competitors data
    store_row = store_pca_components_df.iloc[0][['P_M_1','P_M_2','P_L_1','P_L_2']]
    euclidean_distances = competitors_pca_components_df[['P_M_1','P_M_2','P_L_1','P_L_2']].apply(lambda row: euclidean_distance(store_row, row), axis=1)
    competitors_pca_components_df['EuclideanDistance'] = euclidean_distances
    competitors_pca_components_df = competitors_pca_components_df.sort_values(by=['EuclideanDistance'])
    # Select top 5 competitors
    top5competitors = competitors_pca_components_df[['ChainID','ChainName','SubChainID','SubChainName','StoreID','StoreName']][:5].reset_index(drop=True)
    
    return top5competitors


def present_competitors_table(top5competitors):
    # Present the table
    table_trace = go.Table(
        header=dict(values=list(top5competitors.columns),
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=[top5competitors[col] for col in top5competitors.columns],
                fill_color='white',
                align='left'))

    layout = go.Layout(title='Top 5 Competitors')
    fig3 = go.Figure(data=[table_trace], layout=layout)
    #fig.show()
    pio.write_html(fig3, 'static/top_competitors.html')
    return fig3


def identify_competitors(category, product_description, SubChainName, StoreName, Geographic=None):
    product_df, Store_input_data = preprocess_data(category, product_description, SubChainName, StoreName, Geographic)
    price_movement_df, price_level_df = scale_data(product_df)
    combined_components_df, price_movement_principal_pca, price_level_principal_pca = pca(price_movement_df, price_level_df, product_df)
    fig1 = plot_dimensionality_reduction(combined_components_df, Store_input_data)
    pca_components_df, store_pca_components_df, same_chain_pca_components_df = data_preparation_clustering(price_movement_principal_pca, price_level_principal_pca, Store_input_data, product_df)
    pca_components_df, pca_birch_labels, store_pca_components_df, competitors_pca_components_df = birch_clustering(pca_components_df, Store_input_data, same_chain_pca_components_df)
    fig2 = plot_clusters(pca_components_df, pca_birch_labels, store_pca_components_df)
    top5competitors = find_top_competitors(store_pca_components_df, competitors_pca_components_df)
    fig3 = present_competitors_table(top5competitors)
    
    return fig1, fig2, fig3, top5competitors