import math
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prediction_to_action(x: float) -> (float, str):
    print('x before:', x)
    AGGRESSIVENESS: int = 2  # a higher value means bigger movements on each action
    x: float = (x - 1) * AGGRESSIVENESS + 1

    if x > 2:
        print('Pred to Action: x bigger than 2')
        x = 2
    elif x < 0:
        print('Pred to Action: x smaller than 0')
        x = 0

    print('x after:', x)

    if x > 1:
        a, b, side = 1, 0.5, 'buy'
    else:
        a, b, side = 0.5, 0.75, 'sell'

    return (a / (1 + math.e ** (-100 * (x - 1))) + b), side


def predict(prices: list) -> (float, str):
    model: keras.models.Model = keras.models.load_model('model')
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    last_price: int = prices[-1]
    # Prepare data
    prices_array = np.array(prices).reshape(-1, 1)
    scaled_prices = scaler.fit_transform(prices_array)
    prices_array = np.reshape(scaled_prices, (1, 30, 1))
    # Predict
    prediction_raw = model.predict(prices_array)
    prediction = scaler.inverse_transform(prediction_raw)
    print('Prediction:', prediction)
    # Translate to action
    price_difference = prediction / last_price
    return prediction_to_action(x=price_difference)
