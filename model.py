import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras

PREDICT = True
EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8
TRAINING_STEPS = 60

# Import Data
dataset_raw = pd.read_csv('data/training_data.csv')

# Preprocess Data: Drop timestamp column
dataset = dataset_raw.drop(['timestamp'], axis=1)

# Normalize Data: Scale to range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# Split into training and testing data
training_data = dataset_scaled[0: int(dataset.shape[0] * TRAIN_TEST_SPLIT)]
testing_data = dataset_scaled[int(dataset.shape[0] * TRAIN_TEST_SPLIT): -1]


# Reformat Data as time series
x_train = []
y_train = []
for i in range(TRAINING_STEPS, len(training_data)):
    x_train.append(training_data[i - TRAINING_STEPS: i, 0])
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

model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(1))

# Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)

# Save model
print('SAVING')
model.save('model')

# Test
x_test = []
for i in range(TRAINING_STEPS, len(testing_data)):
    x_test.append(testing_data[i - TRAINING_STEPS:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

if PREDICT:
    model = keras.models.load_model('model')

    print('PREDICTING')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot performance
    real_stock_price = scaler.inverse_transform(testing_data)
    real_stock_price = real_stock_price[TRAINING_STEPS - 1: -1]  # Chop of the first 60 entries to make them the same length as predictions

    plt.plot(real_stock_price, color='black', label='Real Stock Price')
    plt.plot(predictions, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
