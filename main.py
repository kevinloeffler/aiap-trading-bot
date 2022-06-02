import math

from predict import predict
from symbols import symbols
import portfolio
import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()

TARGET_SYMBOL = 'ETHUSD'
IS_CRYPTO = symbols[TARGET_SYMBOL]['is_crypto']

WS_STREAM_URL = 'wss://stream.data.alpaca.markets/v1beta1/crypto' if IS_CRYPTO else 'wss://stream.data.alpaca.markets/v2/iex'
AUTH_MSG = {'action': 'auth', 'key': f'{os.getenv("APCA_API_KEY_ID")}', 'secret': f'{os.getenv("APCA_API_SECRET_KEY")}'}
SUBSCRIPTION_MSG = {"action": "subscribe", "bars": [TARGET_SYMBOL]}

price_queue = [
    1805.1,
    1807.2446169925,
    1806.7532990382,
    1806.4325507253,
    1806.6,
    1805.5604610715,
    1806.0883016299,
    1809.4040871087,
    1808.5101581658,
    1808.4057126708,
    1806.5215899897,
    1806.6165108715,
    1806.6884955752,
    1802.7893315471,
    1800.8010288066,
    1802.3564090299,
    1802.5734511645,
    1803.5487379863,
    1805.7866076116,
    1805.2,
    1804.1031205643,
    1804,
    1804.8978393119,
    1806.148575274,
    1802.7431957274,
    1803.4644831861,
    1803.8421445135,
    1802.6638549075,
    1802.4495995504,
    1805.8297718833,
]


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
                new_holding = current_holding * action
                quantity = np.round(abs(current_holding - new_holding), symbols[TARGET_SYMBOL]['precision'])
                print('Quantity:', quantity)
                portfolio.trade_crypto(symbol=TARGET_SYMBOL, quantity=quantity, side=side)

                print('=== END OF RUN ===')


asyncio.run(start_stream(WS_STREAM_URL))
