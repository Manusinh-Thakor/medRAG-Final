import json
import anyio
import base64
from PIL import Image
from io import BytesIO
import uuid
import os
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent.agent_runner import call_agent

app = FastAPI()

# Enable CORS for browser/client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Limit to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            input_data = json.loads(message)

            print(f"[Received input] {input_data}")

            query = input_data.get("query", "")
            image_flag = input_data.get("image", False)
            image_base64 = input_data.get("image_path", "")

            # Step 1: Decode image if present
            image_file_path = ""
            if image_flag and image_base64:
                try:
                    header, encoded = image_base64.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    image = Image.open(BytesIO(image_data))
                    image_file_path = f"/tmp/{uuid.uuid4()}.jpg"
                    image.save(image_file_path)
                    print(f"[Image saved to] {image_file_path}")
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "data": f"Invalid image: {str(e)}"}))
                    continue

            # Step 2: Create formatted input
            final_input_str = f"Query: {query}\nImage: {image_flag}\nImage Path: {image_file_path}"

            # Step 3: Real-time streaming from sync generator
            data = []
            tool_use = False
            send_lock = threading.Lock()

            async def async_send_step(step):
                await websocket.send_text(json.dumps(step))

            def send_func(step):
                nonlocal tool_use
                if step["type"] == "tool":
                    tool_use = True
                if not any(step["type"] == old["type"] and step["data"] == old["data"] for old in data):
                    data.append(step)
                    with send_lock:
                        anyio.from_thread.run(async_send_step, step)

            def stream_steps():
                for step in call_agent(final_input_str):
                    send_func(step)

            await anyio.to_thread.run_sync(stream_steps)

            # If no tool was used, send final AI message
            if not tool_use:
                for step in data:
                    if step["type"] == "ai":
                        await websocket.send_text(json.dumps(step))
                        break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
