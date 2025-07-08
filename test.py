import asyncio
import websockets
import json
import base64
import mimetypes
import os

async def test_agent():
    uri = "wss://medrag.ilogicaisolutions.com/ws/agent"
    print("Connecting to WebSocket server at", uri)

    async with websockets.connect(uri) as websocket:
        while True:
            query = input("\nEnter your query (or type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting...")
                break

            send_image = input("Do you want to send an image? (y/n): ").strip().lower() == "y"
            base64_data_uri = ""

            if send_image:
                image_path = input("Enter image path (e.g., test.jpg): ").strip()
                if not os.path.exists(image_path):
                    print(f"❌ Image not found: {image_path}")
                    continue

                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

                mime_type, _ = mimetypes.guess_type(image_path)
                base64_data_uri = f"data:{mime_type};base64,{encoded_string}"

            # Send message
            message = {
                "query": query,
                "image": send_image,
                "image_path": base64_data_uri
            }
            await websocket.send(json.dumps(message))

            try:
                print("Streaming response:")
                while True:
                    response = await websocket.recv()
                    print("→", json.loads(response))
            except websockets.exceptions.ConnectionClosed:
                print("🔌 Connection closed by server.")
                break

# Run the test
asyncio.run(test_agent())
