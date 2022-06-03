""" A list of all available stocks and cryptos """

SYMBOLS = {
    'AAPL': {
        'is_crypto': False,
        'precision': 5,
        'path': 'data/training_data_AAPL.csv',
    },
    'BTCUSD': {
        'is_crypto': True,
        'precision': 4,
        'path': 'data/training_data_BTCUSD.csv',
    },
    'ETHUSD': {
        'is_crypto': True,
        'precision': 3,
        'path': 'data/training_data_ETHUSD.csv',
    },
}

PARAMETERS = {
    "rnn_units": 50,
    "step": 30,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "dropout": 0.2,
}
