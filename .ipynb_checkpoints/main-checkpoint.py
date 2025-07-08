import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agent_runner import call_agent

app = FastAPI()

# Optional: Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
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

            # Extract fields
            query = input_data.get("query", "")
            image_flag = input_data.get("image", False)
            image_path = input_data.get("image_path", "")

            # Format input
            final_input_str = f"Query: {query}\nImage: {image_flag}\nImage Path: {image_path}"

            data = []
            skip = False
            tool_use = False

            async for step in call_agent(final_input_str):  # if call_agent is a generator
                if any(step["type"] == old["type"] and step["data"] == old["data"] for old in data):
                    continue

                if step["type"] == "tool":
                    tool_use = True
                    await websocket.send_text(json.dumps(step))
                data.append(step)

            if not tool_use:
                for step in data:
                    if step["type"] == "ai":
                        await websocket.send_text(json.dumps(step))
                        break

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run("websocket_server:app", host="0.0.0.0", port=8000, reload=True)
