import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
load_dotenv()

WS_STREAM_URL = 'wss://stream.data.alpaca.markets/v2/iex'
AUTH_MSG = {'action': 'auth', 'key': f'{os.getenv("APCA_API_KEY_ID")}', 'secret': f'{os.getenv("APCA_API_SECRET_KEY")}'}
SUBSCRIPTION_MSG = {"action": "subscribe", "bars": ["AAPL"]}


async def handle_msg(websocket) -> dict:
    async for message in websocket:
        response = json.loads(message)
        return response


async def consume(stream: str):
    async with websockets.connect(stream) as websocket:
        await handle_msg(websocket)


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
        response = await handle_msg(websocket)
        print(response)


loop = asyncio.get_event_loop()
loop.run_until_complete(start_stream(WS_STREAM_URL))
loop.run_forever()
