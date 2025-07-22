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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def parse_step_response(response):
    """Parse the response containing next_tool information"""
    if response is None:
        return None
        
    # Initialize default values
    step_type = "ai"
    data = ""
    next_tool = "no_tool"
    text_next_tool = None
    
    # If response is a dictionary
    if isinstance(response, dict):
        step_type = response.get("type", step_type)
        data = response.get("data", data)
        next_tool = response.get("next_tool", next_tool)
        
        # Check if data contains a next_tool instruction
        if isinstance(data, str) and '\nnext_tool: ' in data:
            parts = data.split('\nnext_tool: ')
            data = parts[0].strip()
            text_next_tool = parts[1].strip()
    
    # If response is a string
    elif isinstance(response, str):
        # Parse type if present
        if '\ntype: ' in response:
            parts = response.split('\ntype: ')
            data_part = parts[0].strip()
            remaining = parts[1].split('\n')
            step_type = remaining[0].strip()
            response = '\n'.join([data_part] + remaining[1:])
        
        # Parse next_tool if present
        if '\nnext_tool: ' in response:
            parts = response.split('\nnext_tool: ')
            data = parts[0].strip()
            text_next_tool = parts[1].strip()
        else:
            data = response
    
    # Prioritize next_tool from text if both exist
    if text_next_tool is not None:
        next_tool = text_next_tool
    
    # Try to parse data as JSON if it's a string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    
    return {
        "type": step_type,
        "data": data,
        "next_tool": next_tool
    }

@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            input_data = json.loads(message)

            print(f"[Received input] {input_data.get('query', '')}")

            query = input_data.get("query", "")
            image_flag = input_data.get("image", False)
            image_base64 = input_data.get("image_path", "")

            # Decode and save image
            image_file_path = ""
            if image_flag and image_base64:
                try:
                    print("[Image decoding start]")
                    if "," in image_base64:
                        header, encoded = image_base64.split(",", 1)
                    else:
                        encoded = image_base64
                    image_data = base64.b64decode(encoded)
                    image = Image.open(BytesIO(image_data)).convert("RGB")  # ✅ fix here
                    image_file_path = f"/tmp/{uuid.uuid4()}.jpg"
                    image.save(image_file_path)
                    print(f"[Image saved to] {image_file_path}")
                except Exception as e:
                    print(f"[Image decoding failed] {str(e)}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": f"Invalid image: {str(e)}",
                        "next_tool": "no_tool"
                    }))
                    continue


            final_input_str = f"Query: {query}\nImage: {image_flag}\nImage Path: {image_file_path}"

            # Deduplication logic
            seen = set()
            send_lock = threading.Lock()

            async def async_send(step):
                # Ensure all three fields are present
                step.setdefault("type", "ai")
                step.setdefault("data", "")
                step.setdefault("next_tool", "no_tool")
                await websocket.send_text(json.dumps(step))

            def safe_send(step):
                # Skip if step is None
                if step is None:
                    return
                
                # Handle db_images type by converting image paths to actual image data
                if step["type"] == "db_images":
                    try:
                        if isinstance(step["data"], str):
                            image_paths = json.loads(step["data"])
                        else:
                            image_paths = step["data"]
                            
                        image_data_list = []
                        
                        for img_path in image_paths:
                            if os.path.exists(img_path):
                                base64_img = image_to_base64(img_path)
                                if base64_img:
                                    image_data_list.append({
                                        "path": img_path,
                                        "data": f"data:image/jpeg;base64,{base64_img}"
                                    })
                        
                        if image_data_list:
                            step["data"] = image_data_list
                    except Exception as e:
                        print(f"Error processing db_images: {e}")
                        step["data"] = []
                
                key = (step["type"], json.dumps(step["data"]), step["next_tool"])
                if key in seen:
                    return
                seen.add(key)
                with send_lock:
                    anyio.from_thread.run(async_send, step)

            def stream():
                for response in call_agent(final_input_str):
                    if response is None:
                        continue
                        
                    #print(f"[Received response] {response}")
                    
                    # Parse the response into step
                    step = parse_step_response(response)
                    if step:
                        print(f"[Sending step] {step["type"]}")
                        safe_send(step)

            await anyio.to_thread.run_sync(stream)

    except WebSocketDisconnect:
        print("WebSocket disconnected")