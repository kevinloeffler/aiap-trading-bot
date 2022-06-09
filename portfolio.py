from alpaca_trade_api.rest import REST
import dotenv
dotenv.load_dotenv()

KEY_ID = dotenv.get_key(dotenv_path='.env', key_to_get='APCA_API_KEY_ID')
SECRET_KEY = dotenv.get_key(dotenv_path='.env', key_to_get='APCA_API_SECRET_KEY')


api: REST = REST(key_id=KEY_ID, secret_key=SECRET_KEY)


def trade_stock(symbol: str, quantity: float) -> bool:
    pass


def trade_crypto(symbol: str, quantity: float, side: str) -> bool:
    print(f'ORDER: {side} {quantity}')
    order = api.submit_order(symbol, qty=quantity, side=side)
    return True if order.status == 'accepted' else False


def get_position(symbol: str) -> float:
    position = api.get_position(symbol)
    return float(position.qty)

