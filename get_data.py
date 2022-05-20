from alpaca_trade_api.rest import REST, TimeFrame
import csv

api: REST = REST()
SYMBOL: str = 'AAPL'

START_DATE = '2022-05-01T09:00:00Z'
END_DATE = '2022-05-10T23:59:59Z'

data = {
    'training_size': 0,
    'training': [],
    'training_ts': [],
    ##################
    'testing_size': 0,
    'testing': [],
    'testing_ts': [],
}

bars = api.get_bars_iter(SYMBOL, TimeFrame.Minute, START_DATE, END_DATE, adjustment='raw')

# CSV
with open('data/training_data.csv', 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['average-price', 'timestamp'])

    for bar in bars:
        csv_writer.writerow([bar._raw['vw'], bar._raw['t']])
