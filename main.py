import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras

train_test_split = 0.8
plt.rcParams["figure.figsize"] = (10, 6)  # make matplotlib figures bigger

# Import Data
dataset_raw = pd.read_csv('data/training_data.csv')

# Preprocess Data: Drop timestamp column
dataset = dataset_raw.drop(['timestamp'], axis=1)

# Normalize Data: Scale to range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# Split into training and testing data
training_data = dataset_scaled[0: int(dataset.shape[0] * train_test_split)]
testing_data = dataset_scaled[int(dataset.shape[0] * train_test_split): -1]

# Reformat Data as time series
x_train = []
y_train = []
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60: i, 0])
    y_train.append(training_data[i, 0])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build Model
model = keras.models.Sequential()

model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.LSTM(50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.LSTM(50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(1))
