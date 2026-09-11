import asyncio
import json
import websockets


async def test():

    uri = "ws://127.0.0.1:8000/stream"

    print("Connecting to WebSocket...")

    async with websockets.connect(uri) as websocket:

        print("✅ Connected")

        # ==========================================
        # START
        # ==========================================

        await websocket.send(
            json.dumps({
                "event": "start"
            })
        )

        response = await websocket.recv()

        print(
            "Server:",
            response
        )

        # ==========================================
        # MEDIA
        # ==========================================

        await websocket.send(
            json.dumps({
                "event": "media"
            })
        )

        response = await websocket.recv()

        print(
            "Server:",
            response
        )

        # ==========================================
        # STOP
        # ==========================================

        await websocket.send(
            json.dumps({
                "event": "stop"
            })
        )

        response = await websocket.recv()

        print(
            "Server:",
            response
        )


asyncio.run(test())