from predict import predict
import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
load_dotenv()


WS_STREAM_URL = 'wss://stream.data.alpaca.markets/v2/iex'
AUTH_MSG = {'action': 'auth', 'key': f'{os.getenv("APCA_API_KEY_ID")}', 'secret': f'{os.getenv("APCA_API_SECRET_KEY")}'}
SUBSCRIPTION_MSG = {"action": "subscribe", "bars": ["AAPL"]}


price_queue = [0] * 60

test_price_queue = [175.29, 175.12, 175.01, 174.94, 174.90, 174.96, 175.05, 175.29, 175.30, 175.31, 175.33, 175.36,
                    175.33, 175.44, 175.40, 175.49, 175.53, 175.52, 175.50, 175.46, 175.36, 175.31, 175.30, 175.39,
                    175.44, 175.40, 175.43, 175.50, 175.50, 175.49, 175.50, 175.51, 175.51, 175.50, 175.50, 175.49,
                    175.48, 175.49, 175.51, 175.52, 175.58, 175.54, 175.59, 175.61, 175.62, 175.50, 175.55, 175.59,
                    175.56, 175.49, 175.51, 175.56, 175.62, 175.71, 175.64, 175.62, 175.63, 175.68, 175.65, 175.66]

test_bar: dict = {'T': 'b', 'S': 'SPY', 'o': 388.985, 'h': 389.13, 'l': 388.975, 'c': 160.0, 'v': 49378, 't': '2021-02-22T19:15:00Z'}


def handle_bar(queue: list, bar: dict = None):
    if bar is None:
        bar = test_bar
    close_price = bar["c"]
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
                prediction = predict(price_queue)
                print('Prediction:', prediction)


asyncio.run(start_stream(WS_STREAM_URL))

'''
loop = asyncio.get_event_loop()
loop.run_until_complete(start_stream(WS_STREAM_URL))
loop.run_forever()
loop.run_until_complete(handle_msg(WS_STREAM_URL))
'''
