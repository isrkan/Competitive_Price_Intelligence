import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objects as go

def imputation(input_data,category):
    
    # Handle nan values
    nan_locations = np.isnan(input_data)
    input_data[nan_locations] = 0

    # Load the scaler
    with open('trained_models/time_series_imputation/'+str(category)+'/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Load the trained model
    loaded_model = tf.saved_model.load('trained_models/time_series_imputation/'+str(category))
    inference = loaded_model.signatures["serving_default"]

    # Convert the input data to tensors
    input_tensor = tf.convert_to_tensor(input_data_scaled, dtype=tf.float32)

    # Extract the key based on the pattern 'dense' + number
    output_dict = inference(input_tensor)
    output_key = next(key for key in output_dict.keys() if key.startswith('dense'))
    # Make predictions
    predictions_scaled = inference(input_tensor)[output_key].numpy()
    # Inverse scale the predictions
    predictions = scaler.inverse_transform(predictions_scaled)
    # Integrate the imputations to the data and round the values
    zero_indices = np.where(input_data == 0)
    input_data[zero_indices] = predictions[zero_indices]
    input_data = np.round(input_data, decimals=1)

    return input_data


def forecasting(competitor_product_imputed_df,category):
    
    input_forecasting_data = competitor_product_imputed_df.values
    # Define the window size
    sequence_length = 90
    input_data = input_forecasting_data[:, -sequence_length:]

    # Load the trained model
    loaded_forecasting_model = tf.saved_model.load('trained_models/time_series_forecasting/'+str(category))
    
    # Define the number of periods to predict
    num_periods = 10

    # Make predictions for each store
    all_predictions = []
    for row in input_data:
        # Get the values to make predictions
        input_sequence = row
        # Reshape the input sequence
        input_sequence = np.reshape(input_sequence, (1, input_sequence.shape[0], 1))

        # Process  the model
        inference = loaded_forecasting_model.signatures["serving_default"]
        # Convert the input data to tensors
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)

        # Make predictions for the next X periods 
        predictions = []
        for _ in range(num_periods):
            # Extract the key based on the pattern 'dense' + number
            output_dict = inference(input_tensor)
            output_key = next(key for key in output_dict.keys() if key.startswith('dense'))
            # Predict the next value
            next_prediction = inference(input_tensor)[output_key].numpy()

            # Append the prediction
            predictions.append(next_prediction)
            # Update the input sequence for the next prediction
            current_sequence = np.append(input_sequence[:, 1:, :], [next_prediction], axis=1)
            input_tensor = tf.convert_to_tensor(current_sequence, dtype=tf.float32)

        # Append the predictions to the overall predictions
        all_predictions.append(predictions)

    # Convert the predictions to a numpy array and round the values
    all_predictions = np.array(all_predictions)
    forecasts_data = np.reshape(all_predictions, (all_predictions.shape[0], all_predictions.shape[1]))
    forecasts_data = np.round(forecasts_data, decimals=1)

    return forecasts_data


def imputation_forecasting(category, top5competitors, product_description):
    
    # Import the price data in one category
    category_df = pd.read_parquet('competitor_recognition_data/'+str(category+'.parquet'))
    category_df = category_df.set_index(['category','ProductDescription','StoreID'])

    # Convert the data frames to numpy arrays
    input_imputation_data = category_df.copy().values
    # Impute the data
    imputed_data = imputation(input_imputation_data, category)
    
    # Filter the specific data of the product and the competitors
    category_df = category_df.reset_index()
    product_df = category_df[category_df['ProductDescription']==product_description]
    competitor_product_df = product_df[product_df['StoreID'].isin(top5competitors['StoreID'])]
    # Filter the specific imputed data of the product and the competitors
    imputed_data_df = pd.DataFrame(imputed_data)
    competitor_product_imputed_df = imputed_data_df.loc[competitor_product_df.index].reset_index(drop=True)
    # Forecast the data
    forecasting_data = forecasting(competitor_product_imputed_df, category)
    competitor_product_forecast_df = pd.DataFrame(forecasting_data)
    competitor_product_imputed_forecast_df = pd.concat([competitor_product_imputed_df, competitor_product_forecast_df], axis=1)
    
    # Reindex and rename columns
    competitor_product_df= competitor_product_df.reset_index(drop=True)
    competitor_product_imputed_df = pd.concat([competitor_product_df[['category', 'ProductDescription', 'StoreID']], competitor_product_imputed_df], axis=1).set_index(['category','ProductDescription','StoreID'])
    competitor_product_imputed_df.columns = competitor_product_df.columns[3:]
    
    competitor_product_imputed_forecast_df = pd.concat([competitor_product_df[['category', 'ProductDescription', 'StoreID']], competitor_product_imputed_forecast_df], axis=1).set_index(['category','ProductDescription','StoreID'])
    existing_dates = pd.to_datetime(competitor_product_df.columns[3:])
    additional_dates = pd.date_range(existing_dates[-1], periods=11, freq='D')[-10:]
    all_dates = existing_dates.union(additional_dates).strftime('%Y-%m-%d')
    competitor_product_imputed_forecast_df.columns = all_dates
    
    competitor_product_df = competitor_product_df.set_index(['category','ProductDescription','StoreID'])
    
    # Create plots for the each store from the competitors
    for _, row in top5competitors.iterrows():
        store_id = row['StoreID']
        store_name = row['StoreName']

        # Filter the dataframes based on the store ID
        filtered_competitor_product_df = competitor_product_df.loc[competitor_product_df.index.get_level_values('StoreID') == store_id]
        filtered_competitor_product_imputed_df = competitor_product_imputed_df.loc[competitor_product_imputed_df.index.get_level_values('StoreID') == store_id]
        filtered_competitor_product_imputed_forecast_df = competitor_product_imputed_forecast_df.loc[competitor_product_imputed_forecast_df.index.get_level_values('StoreID') == store_id]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_competitor_product_imputed_forecast_df.columns[3:],
                                y=filtered_competitor_product_imputed_forecast_df.iloc[0, 3:],
                                name='Imputed Forecast Data'))
        fig.add_trace(go.Scatter(x=filtered_competitor_product_imputed_df.columns[3:],
                                y=filtered_competitor_product_imputed_df.iloc[0, 3:],
                                name='Imputed Data'))
        fig.add_trace(go.Scatter(x=filtered_competitor_product_df.columns[3:],
                                y=filtered_competitor_product_df.iloc[0, 3:],
                                name='Data'))

        fig.update_layout(title=f'Predictions for Store ID {store_id}',
                        xaxis_title='Date',
                        yaxis_title='Price')

        # Save the plot
        fig.write_html(f'static/imputation_forecast_plot_store_{store_id}.html')

    return competitor_product_df, competitor_product_imputed_df, competitor_product_imputed_forecast_df