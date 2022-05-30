import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

model = keras.models.load_model('model')

test_data = []
start_price = 160.

for i in range(60):
    test_data.append(start_price + i / 2)

test_data = np.array(test_data).reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(test_data)

x_test = np.reshape(scaled_data, (1, 60, 1))

prediction_raw = model.predict(x_test)
prediction = scaler.inverse_transform(prediction_raw)

print(prediction)
