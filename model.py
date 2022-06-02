import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras


def get_data():
    # Import Data
    dataset_raw = pd.read_csv('data/training_data_AAPL.csv')

    # Preprocess Data: Drop timestamp column
    dataset = dataset_raw.drop(['timestamp'], axis=1)
    return dataset


def normalize_data(dataset):
    # Normalize Data: Scale to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)
    return dataset_scaled


def split_train_test(dataset_scaled, split_index):
    training_data = dataset_scaled[0: split_index]
    testing_data = dataset_scaled[split_index: -1]
    return training_data, testing_data


# Reformat Data as time series
def reformat_data_as_time_series(training_data, step):
    x_train = []
    y_train = []
    for i in range(step, len(training_data)):
        x_train.append(training_data[i - step: i, 0])
        y_train.append(training_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


# Build Model
def train_model(params, training_data):
    units = params["rnn_units"]
    step = params["step"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]

    x_train, y_train = reformat_data_as_time_series(training_data, step)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(units, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(units))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1))

    # Train
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save model
    print('SAVING')
    model.save('model')
    return history, model


#PREDICT = True
#EPOCHS = 50
#TRAIN_TEST_SPLIT = 0.8
#TRAINING_STEPS = 60
#
#dataset = get_data()
#dataset_scaled = normalize_data(dataset)
#
## Split into training and testing data
#split_index = int(dataset.shape[0] * TRAIN_TEST_SPLIT)
#
#training_data, testing_data = split_train_test(dataset_scaled, split_index)
#
#params = {
#    "rnn_units": 50,
#    "step": TRAINING_STEPS,
#    "optimizer": "adam",
#    "learning_rate": 0.8,
#    "batch_size": 32,
#    "epochs": EPOCHS,
#}
#
#train_model(params, training_data)
#
## Test
#x_test = []
#for i in range(TRAINING_STEPS, len(testing_data)):
#    x_test.append(testing_data[i - TRAINING_STEPS:i, 0])
#x_test = np.array(x_test)
#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
#if PREDICT:
#    model = keras.models.load_model('model')
#
#    print('PREDICTING')
#
#    predictions = model.predict(x_test)
#    predictions = scaler.inverse_transform(predictions)
#
#    # Plot performance
#    real_stock_price = scaler.inverse_transform(testing_data)
#    real_stock_price = real_stock_price[TRAINING_STEPS - 1: -1]  # Chop of the first 60 entries to make them the same length as predictions
#
#    plt.plot(real_stock_price, color='black', label='Real Stock Price')
#    plt.plot(predictions, color='green', label='Predicted Stock Price')
#    plt.title('Stock Price Prediction')
#    plt.xlabel('Time')
#    plt.ylabel('Stock Price')
#    plt.legend()
#    plt.show()
#
