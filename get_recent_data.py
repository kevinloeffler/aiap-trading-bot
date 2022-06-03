from config import SYMBOLS
from alpaca_trade_api.rest import REST, TimeFrame
import datetime as dt

# Select the target symbol:
TARGET = 'ETHUSD'


def format_datetime(datetime: dt.datetime) -> str:
    return f'{datetime.year}-{str(datetime.month).rjust(2, "0")}-{str(datetime.day).rjust(2, "0")}' \
           f'T{str(datetime.hour).rjust(2, "0")}:{str(datetime.minute).rjust(2, "0")}:59Z'


def get_last_prices(length: int) -> [float]:
    now = dt.datetime.now(tz=dt.timezone.utc)
    start_date = format_datetime(datetime=now - dt.timedelta(hours=2))
    end_date = format_datetime(datetime=now)

    api: REST = REST()
    prices = []

    if not SYMBOLS[TARGET]['is_crypto']:
        print('ERROR: Stocks are not implemented')

    bars = api.get_crypto_bars(TARGET, timeframe=TimeFrame.Minute, start=start_date, end=end_date)

    for bar in bars:
        prices.append(bar._raw['vw'])

    prices.reverse()
    prices = prices[0: length]

    return prices
