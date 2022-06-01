""" A list of all available stocks and cryptos """

symbols = {
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
