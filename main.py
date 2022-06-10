from config import SYMBOLS, PARAMETERS
from predict import predict
from get_recent_data import get_last_prices
import portfolio
import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()

TARGET_SYMBOL = 'ETHUSD'
IS_CRYPTO = SYMBOLS[TARGET_SYMBOL]['is_crypto']

WS_STREAM_URL = 'wss://stream.data.alpaca.markets/v1beta1/crypto' if IS_CRYPTO else 'wss://stream.data.alpaca.markets/v2/iex'
AUTH_MSG = {'action': 'auth', 'key': f'{os.getenv("APCA_API_KEY_ID")}', 'secret': f'{os.getenv("APCA_API_SECRET_KEY")}'}
SUBSCRIPTION_MSG = {"action": "subscribe", "bars": [TARGET_SYMBOL]}

MAX_AMOUNT = 40

price_queue = get_last_prices(length=PARAMETERS['step'])


def handle_bar(queue: list, bar: dict):
    close_price = bar["vw"]
    queue.append(close_price)
    queue.pop(0)


# # # WEBSOCKETS # # #

async def handle_msg(websocket) -> dict:
    async for message in websocket:
        response = json.loads(message)
        return response


async def start_stream(stream: str):
    async with websockets.connect(stream) as websocket:
        # Connect
        connection_response: dict = await handle_msg(websocket)
        if connection_response[0]['T'] == 'success':
            print('CONNECTED')
        else:
            print('CONNECTION FAILED')
            return

        # Authenticate
        await websocket.send(json.dumps(AUTH_MSG))
        auth_response: dict = await handle_msg(websocket)
        if auth_response[0]['T'] == 'success':
            print('AUTHENTICATED')
        else:
            print('NOT AUTHENTICATED')
            return

        # Subscribe
        await websocket.send(json.dumps(SUBSCRIPTION_MSG))
        sub_response = await handle_msg(websocket)
        if sub_response[0]['T'] == 'subscription':
            print('SUBSCRIBED')
        else:
            print('NOT SUBSCRIBED')

        while True:
            message = await websocket.recv()
            bar = json.loads(message)[0]
            if bar['T'] == 'b':
                handle_bar(queue=price_queue, bar=bar)
                print(f'Added newest price ({bar["vw"]}) to queue')
                action, side = predict(price_queue)

                current_holding = portfolio.get_position(TARGET_SYMBOL)
                new_holding = float(current_holding * action)

                if current_holding > MAX_AMOUNT:
                    quantity = round(abs(current_holding - new_holding), SYMBOLS[TARGET_SYMBOL]['precision'])
                    portfolio.trade_crypto(symbol=TARGET_SYMBOL, quantity=quantity, side=side)
                else:
                    print(f'Max Amount of {TARGET_SYMBOL} reached.')

                print('=== END OF RUN ===')


asyncio.run(start_stream(WS_STREAM_URL))
