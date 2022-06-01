from alpaca_trade_api.rest import REST

api: REST = REST()


def trade_stock(symbol: str, quantity: float) -> bool:
    pass


def trade_crypto(symbol: str, quantity: float, side: str) -> bool:
    order = api.submit_order(symbol, qty=quantity, side=side)
    return True if order.status == 'accepted' else False


def get_position(symbol: str) -> float:
    position = api.get_position(symbol)
    return float(position.qty)

