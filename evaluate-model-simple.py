from model import get_data, normalize_data, reformat_data_as_time_series, train_model, split_train_test
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras


PREDICT = True
EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8
TRAINING_STEPS = 60

dataset = get_data()
dataset_scaled = normalize_data(dataset)

# Split into training and testing data
split_index = int(dataset.shape[0] * TRAIN_TEST_SPLIT)

training_data, testing_data = split_train_test(dataset_scaled, split_index)

params = {
    "rnn_units": 50,
    "step": TRAINING_STEPS,
    "optimizer": "adam",
    "learning_rate": 0.8, # not considered
    "batch_size": 32,
    "epochs": EPOCHS,
}

train_model(params, training_data)

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
    scaler = MinMaxScaler(feature_range=(0, 1))
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
