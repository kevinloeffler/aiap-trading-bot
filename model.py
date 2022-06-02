import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf

def get_data():
    # Import Data
    dataset_raw = pd.read_csv('data/training_data_AAPL.csv') #TODO final with ETH

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
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    dropout = params["dropout"]

    x_train, y_train = reformat_data_as_time_series(training_data, step)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.LSTM(units, return_sequences=True))
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.LSTM(units))
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    # Train
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save model
    print('SAVING')
    model.save('model')
    return history, model

