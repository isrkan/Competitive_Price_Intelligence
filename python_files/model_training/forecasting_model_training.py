import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Masking, SimpleRNN, LSTM, GRU
from keras.models import Model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle


def preprocess_data(category):
    # Import the price data in one category
    category_df = pd.read_parquet('retail_pricing/competitor_recognition_data/'+str(category+'.parquet'))
    category_df = category_df.set_index(['category','ProductDescription','StoreID'])

    # Filter observations with more than 25% nan values
    df_missing = category_df.dropna(thresh=int(0.75*731), axis=0)

    # Import the store data
    store_data = pd.read_csv('retail_pricing/competitor_recognition_data/Store_data_git.csv')
    store_data = store_data[['StoreID','ChainID','DistrictName','StoreType','LocationType']]
    store_data_with_dummies = pd.get_dummies(store_data, columns=['ChainID','DistrictName','StoreType','LocationType'], drop_first=True)
    store_data_with_dummies = store_data_with_dummies.set_index('StoreID')

    df_missing = store_data_with_dummies.merge(df_missing, how='right', left_index=True, right_on='StoreID')

    # Convert the data frames to numpy arrays
    input_data = df_missing.values
    # Preprocess the input data to handle NaN values
    nan_locations = np.isnan(input_data)
    input_data[nan_locations] = -1

    # Load the scaler
    with open('retail_pricing/trained_models/time_series_imputation/'+str(category)+'/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Scale the input data using the loaded scaler
    input_data_scaled = input_data.copy()
    input_data_scaled[:, 42:773] = scaler.transform(input_data[:, 42:773])
    
    # Load the imputation model
    loaded_model = tf.saved_model.load('retail_pricing/trained_models/time_series_imputation/'+str(category))
    inference = loaded_model.signatures["serving_default"]
    # Convert the input data to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data_scaled, dtype=tf.float32)
    # Make predictions
    predictions_scaled = inference(input_tensor)[list(inference(input_tensor).keys())[0]].numpy()
    # Inverse scale the predictions
    predictions = input_data.copy()
    predictions[:, 42:773] = scaler.inverse_transform(predictions_scaled)
    
    # Find the indices where input_data is -1
    minus_one_indices = np.where(input_data == -1)

    # Replace the zero values with the corresponding values from predictions
    input_data[minus_one_indices] = predictions[minus_one_indices]
    
    return input_data


def prepare_data_for_training(input_data, sequence_length):
    # Scale the input data
    scaler = MinMaxScaler()
    input_data_scaled = input_data[:, 42:773].copy()
    #input_data_scaled = scaler.fit_transform(input_data[:, 42:773])

    # Split the data into input sequences and corresponding labels
    X = []
    y = []
    for j, row in enumerate(input_data_scaled):
        for i in range(len(row) - sequence_length):
            X.append(np.concatenate((input_data[j:j+1, 0:42][0], row[i:i+sequence_length]), axis=None))
            y.append(row[i+sequence_length])
    X = np.array(X)
    y = np.array(y)

    # Sample examples for training
    random_numbers = np.round(np.random.uniform(1, len(X), size=20000)).astype(int)
    X_sample = X[random_numbers]
    y_sample = y[random_numbers]
    #X_sample = X
    #y_sample = y

    # Split the data into training, validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val


def lstm_model(X_train, X_val, y_train, y_val, sequence_length, category):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(sequence_length+42, 1)))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, shuffle=True, batch_size=32, verbose=1)
    
    # Save the model in SavedModel format
    tf.saved_model.save(model, 'retail_pricing/trained_models/time_series_forecasting/'+str(category))
    
    return model

def main():
    categories = ["rice","pasta","legumes and cereals","biscuits", "waffles", "chocolates", "black coffee", "toilet paper", "yogurt","cheese","canned food","wine","beer","water","non alcoholic carbonated beverages","non alcoholic non carbonated beverages","toothpaste","bread", "flour and semolina","gums and candies","honey and date honey", "jam and confiture", "ketchup, mayonnaise and mustard", "pastrami, sausage and kabanos", "pickled or smoked fish","deodorant","shower gel","shampoo"]
    
    sequence_length = 90  # Define the sequence length for input sequences
    
    for category in categories:
        print('Category:', category)
        input_data = preprocess_data(category)
        
        X_train, X_val, y_train, y_val =  prepare_data_for_training(input_data, sequence_length)

        model = lstm_model(X_train, X_val, y_train, y_val, sequence_length, category)

if __name__ == "__main__":
    main()