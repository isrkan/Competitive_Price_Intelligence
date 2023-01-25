import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Masking, SimpleRNN, LSTM, GRU
from keras.models import Model
from keras.layers import Bidirectional, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
readRDS = robjects.r['readRDS']


def Bi_RNN_imputation(X_train, y_train, X_test, y_test):
    
    # create and fit the bidirectional RNN network
    
    # Define the model architecture
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.reshape(-1, 1).shape[1], 1)))
    model.add(Bidirectional(SimpleRNN(32, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    model.add(Bidirectional(SimpleRNN(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(X_train.shape[1], activation='linear'))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=1, shuffle=True, verbose=0)
    
    # Make predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    
    return trainPredict, y_train, testPredict, y_test


def Encoder_decoder_forecasting(trainX, trainY, testX, testY,look_back):
    
    # create and fit the encoder-decoder network
    
    # Define the model architecture
    encoder = Sequential()
    encoder.add(LSTM(8, input_shape=(1, look_back), return_sequences=True))
    encoder.add(SimpleRNN(4))

    decoder = Sequential()
    decoder.add(RepeatVector(look_back))
    decoder.add(LSTM(8, return_sequences=True))
    decoder.add(Dense(1))

    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    # Train the model
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # Generate the input for future predictions
    future_input = np.array(testX[-1]).reshape((1, 1, look_back))
    # Initialize an empty list to store the future predictions
    future_predictions = []
    # Loop through the number of steps you want to predict in the future
    for i in range(10):
        # Get the next prediction
        next_prediction = model.predict(future_input)[0]
        # Append the prediction to the list
        future_predictions.append(next_prediction)
        # Update the input for the next prediction
        next_prediction = next_prediction.reshape(1, 1, 1)
        future_input = np.concatenate((future_input, next_prediction))
    
    return trainPredict, trainY, testPredict, testY, future_predictions
    

    
def imputation(data, model_name):
    
    # Interpolate missing values with linear method
    data['ImputedFinalPrice'] = data['FinalPrice'].interpolate(method='linear')
    # Impute missing values with the last non-null value
    data['ImputedFinalPrice'] = data['ImputedFinalPrice'].fillna(method='ffill')
    # Impute missing values with the next non-null value
    data['ImputedFinalPrice'] = data['ImputedFinalPrice'].fillna(method='backfill')
    
    # Split into train and test sets
    train_size = int(len(data['FinalPrice']) * 0.67)
    test_size = len(data['FinalPrice']) - train_size
    X_train, X_test = data['FinalPrice'][0:train_size], data['FinalPrice'][train_size:len(data['FinalPrice'])]
    y_train, y_test = data['ImputedFinalPrice'][0:train_size], data['ImputedFinalPrice'][train_size:len(data['ImputedFinalPrice'])]
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.values.reshape(-1, 1))
    X_test = scaler.transform(X_test.values.reshape(-1, 1))
    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))

    # Reshape the data to (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))

    # Replace the 'nan' values with 0
    X_train = np.nan_to_num(X_train, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Train imputation model
    if model_name == 'Bi_RNN':
        trainPredict, y_train, testPredict, y_test = Bi_RNN_imputation(X_train, y_train, X_test, y_test)
    
    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    orig_trainY = scaler.inverse_transform(y_train.reshape(len(y_train), 1))
    testPredict = scaler.inverse_transform(testPredict)
    orig_testY = scaler.inverse_transform(y_test.reshape(len(y_test), 1))
    
    # Add the series to the data
    data['BiRNNImputedFinalPrice'] = pd.concat([pd.DataFrame(trainPredict),pd.DataFrame(testPredict)]).reset_index(drop=True)
    # Integrate the predicted values to the 'nan' values only
    data['FinalPriceFilled'] = data['FinalPrice'].fillna(data['BiRNNImputedFinalPrice'])
    
    return data



def forecasting(data, model_name):
    
    # Scale the data
    scaler = MinMaxScaler()
    data['FinalPriceFilled_s'] = scaler.fit_transform(data[['FinalPriceFilled']])
    
    # Split into train and test sets
    train_size = int(len(data['FinalPriceFilled_s']) * 0.67)
    test_size = len(data['FinalPriceFilled_s']) - train_size
    train, test = data['FinalPriceFilled_s'][0:train_size], data['FinalPriceFilled_s'][train_size:len(data['FinalPriceFilled_s'])]
    
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train.reset_index(drop=True), look_back)
    testX, testY = create_dataset(test.reset_index(drop=True), look_back)
    
    # reshape input to be [number of samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, look_back))
    testX = np.reshape(testX, (testX.shape[0], 1, look_back))
    
    # Convert the data to tensors
    trainX = tf.convert_to_tensor(trainX, dtype=tf.float32)
    trainY = tf.convert_to_tensor(trainY, dtype=tf.float32)
    testX = tf.convert_to_tensor(testX, dtype=tf.float32)
    testY = tf.convert_to_tensor(testY, dtype=tf.float32)
    
    # Train forecasting model
    if model_name == 'Encoder_Decoder':
        trainPredict, y_train, testPredict, y_test, future_predictions = Encoder_decoder_forecasting(trainX, trainY, testX, testY, look_back)
    
    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
    orig_trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
    orig_testY = scaler.inverse_transform([testY])
    
    # Invert the out of given series predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Add the predicted series to the data
    data['EncoderDecoderPredictedFinalPrice'] = pd.concat([pd.DataFrame(trainPredict),pd.DataFrame([np.nan]),pd.DataFrame(testPredict),pd.DataFrame([np.nan])]).reset_index(drop=True)
    data['FinalPriceFilled'] = data['FinalPriceFilled'].astype(np.float64)
    data['EncoderDecoderPredictedFinalPrice'] = data['EncoderDecoderPredictedFinalPrice'].astype(np.float64)
    data['EncoderDecoderPredictedFinalPrice'].fillna(data['FinalPriceFilled'], inplace=True)
    
    # return also predictions out of the sample
    future_predictions_data = pd.DataFrame(future_predictions, columns=['EncoderDecoderPredictedFinalPrice'])
    
    return data, future_predictions_data


def plot_forecast(imputed_series_data, future_predictions_data):
    
    # Insert 'nan' values to existing values in the imputed series
    imputed_series_data['JustImputedFinalPrice'] = imputed_series_data['BiRNNImputedFinalPrice'].values
    non_nan_indices = np.ravel(np.array(np.where(np.isnan(imputed_series_data['FinalPrice']) == False)))
    imputed_series_data.loc[non_nan_indices, 'JustImputedFinalPrice'] = np.nan

    # Construct the predicted future series
    predicted_series = pd.concat([imputed_series_data['FinalPrice'],future_predictions_data]).reset_index(drop=True)
    predicted_series.columns=['FinalPrice','PredictedFinalPrice']

    # Define date range for the plot
    dates = pd.date_range(start='2019-01-01', end='2021-03-10')
    dates_without_time = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in dates]

    # Plot the series
    plt.plot(imputed_series_data['FinalPriceFilled'], color='blue',linewidth=2)
    plt.plot(imputed_series_data['JustImputedFinalPrice'], color='red',linewidth=3)
    plt.plot(predicted_series['PredictedFinalPrice'], color='orange',linewidth=3)
    plt.legend(['Original Series', 'Imputed Series','Predicted Series'], loc='upper left', bbox_to_anchor=(1,1))
    plt.xticks(np.arange(0, len(dates), 60), dates_without_time[::60], rotation=90)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    plt.savefig('forecast.png')
    
    return 'forecast.png'



def time_series_imputation_prediction(category,store_id,barcode):
    # Import the data
    data = readRDS('G:/retail data/rds data/data by store/'+category+'/'+store_id+'.rds')
    data = pandas2ri.rpy2py_dataframe(data)

    data['Date'] = pd.to_datetime(data['Date'], utc=True, unit='d').apply(lambda x: x.strftime('%Y-%m-%d'))
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce').sort_values()
    # Filter by barcode
    data = data[data['Barcode']==barcode]
    
    # group the data by date and calculate the mean of each group
    data_agg = data.groupby('Date', as_index = False).agg({'FinalPrice':'mean', 'Barcode':'first', 'ProductDescription':'first', 'Manufacturer':'first', 'Category':'first', 'ChainID':'first', 'ChainName':'first', 'SubChainID':'first', 'SubChainName':'first','StoreID':'first', 'StoreName':'first', 'Price':'mean'})
    
    ##### check for missing dates in the data

    # create a dataframe with the date range
    date_range = pd.date_range('2019-01-01 00:00:00', '2021-02-28 00:00:00')
    date_df = pd.DataFrame({'Date': date_range})

    # merge the dataframe with the date range
    merged_data = date_df.merge(data_agg, on='Date', how='left')

    # check for missing dates
    missing_dates = merged_data[merged_data['FinalPrice'].isnull()]['Date']
    
    ### Imputation ###
    imputed_series_data = imputation(merged_data, 'Bi_RNN')
    
    ### forecasting ###
    predicted_series_data, future_predictions_data = forecasting(imputed_series_data, 'Encoder_Decoder')
    
    ### plotting the series
    series_plot = plot_forecast(imputed_series_data, future_predictions_data)
    
    return series_plot
