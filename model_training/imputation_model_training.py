import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle


def preprocess_data(category):
    # Import the price data in one category
    category_df = pd.read_parquet('../competitor_recognition_data/'+str(category+'.parquet'))
    category_df = category_df.set_index(['category','ProductDescription','StoreID'])
    
    # Filter observations with more than 25% nan values
    df_missing = category_df.dropna(thresh=int(0.75*731), axis=0)
    
    # Interpolation - linear
    lfill_df = df_missing.T.interpolate(method='linear').T
    # Imputation - first NOCB and then LOCB
    nlfill_df = lfill_df.T.fillna(method='bfill').T
    df_imputed = nlfill_df.T.fillna(method='ffill').T
    
    # Import the store data
    store_data = pd.read_csv('../competitor_recognition_data/Store_data_git.csv')
    store_data = store_data[['StoreID','ChainID','DistrictName','StoreType','LocationType']]
    store_data_with_dummies = pd.get_dummies(store_data, columns=['ChainID','DistrictName','StoreType','LocationType'], drop_first=True)
    store_data_with_dummies = store_data_with_dummies.set_index('StoreID')

    df_missing = store_data_with_dummies.merge(df_missing, how='right', left_index=True, right_on='StoreID')
    
    # Convert the data frames to numpy arrays
    input_data = df_missing.values
    target_data = df_imputed.values
    
    # Preprocess the input data to handle NaN values
    nan_locations = np.isnan(input_data)
    input_data[nan_locations] = -1
    
    return df_missing, df_imputed, input_data, target_data, store_data_with_dummies

def prepare_data_for_training(input_data,target_data):
    
    # Split the data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Create a scaler object
    scaler = MinMaxScaler()
    # Fit the scaler on the training data and transform the data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, 42:773] = scaler.fit_transform(X_train[:, 42:773])
    X_val_scaled[:, 42:773] = scaler.transform(X_val[:, 42:773])
    X_test_scaled[:, 42:773] = scaler.transform(X_test[:, 42:773])
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)
    # Scale the input data using the trained scaler
    input_data_scaled = input_data.copy()
    input_data_scaled[:, 42:773] = scaler.transform(input_data[:, 42:773])
    
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, input_data_scaled

def ffnn_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, category, scaler, input_data, target_data):
    
    # Define the model architecture
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_data.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(target_data.shape[1]))  # Output layer with the same number of features

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_scaled, y_train_scaled, validation_data=(X_val_scaled, y_val_scaled), epochs=100, batch_size=32, shuffle=True, verbose=0)
    
    # Save the model in SavedModel format
    tf.saved_model.save(model, '../trained_models/time_series_imputation/'+str(category))
    
    # Save the scaler
    with open('../trained_models/time_series_imputation/'+str(category)+'/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model

def plot_imputed_vs_predicted_series(model, input_data_scaled, scaler, target_data, input_data, store_data_with_dummies):
    
    #plot
    predictions_scaled = model.predict(input_data_scaled)
    predictions = scaler.inverse_transform(predictions_scaled)
    # Select a specific series index to plot
    series_index = 1
    # Get the imputed series from the target_data
    imputed_series = target_data[series_index, :]
    # Get the corresponding predicted series
    predicted_series = predictions[series_index, :]
    missing_series = input_data[series_index, store_data_with_dummies.shape[1]:]
    # Create a time axis (assuming the columns represent dates or time steps)
    time_axis = np.arange(target_data.shape[1])
    # Plot the imputed series and predicted series
    plt.plot(time_axis, imputed_series, label='Imputed Series')
    plt.plot(time_axis, predicted_series, label='Predicted Series')
    plt.plot(time_axis, missing_series, label='Missing Series')
    plt.title('Imputed vs Predicted Series')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
def main():
    categories = ["rice","pasta","legumes and cereals","biscuits", "waffles", "chocolates", "black coffee", "toilet paper", "yogurt","cheese","canned food","wine","beer","water","non alcoholic carbonated beverages","non alcoholic non carbonated beverages","toothpaste","bread", "flour and semolina","gums and candies","honey and date honey", "jam and confiture", "ketchup, mayonnaise and mustard", "pastrami, sausage and kabanos", "pickled or smoked fish","deodorant","shower gel","shampoo"]
    
    for category in categories:
        print('Category:', category)
        df_missing, df_imputed, input_data, target_data, store_data_with_dummies = preprocess_data(category)

        scaler, X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, input_data_scaled = prepare_data_for_training(input_data,target_data)

        model = ffnn_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, category, scaler, input_data, target_data)

        plot_imputed_vs_predicted_series(model, input_data_scaled, scaler, target_data, input_data, store_data_with_dummies)

if __name__ == "__main__":
    main()
