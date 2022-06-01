from symbols import symbols
from alpaca_trade_api.rest import REST, TimeFrame
import csv

# Select the target symbol:
TARGET = 'ETHUSD'
# Select the time period
START_DATE = '2022-04-01T09:00:00Z'
END_DATE = '2022-04-30T23:59:59Z'


api: REST = REST()
SYMBOL: str = TARGET
SAVE_LOCATION: str = symbols[TARGET]['path']


if symbols[TARGET]['is_crypto']:
    bars = api.get_crypto_bars(TARGET, TimeFrame.Minute, START_DATE, END_DATE)
else:
    bars = api.get_bars_iter(SYMBOL, TimeFrame.Minute, START_DATE, END_DATE, adjustment='raw')


# CSV
with open(SAVE_LOCATION, 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['average-price', 'timestamp'])

    for bar in bars:
        csv_writer.writerow([bar._raw['vw'], bar._raw['t']])
