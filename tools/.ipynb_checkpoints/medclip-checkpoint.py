from langchain_core.tools import tool
import os
import yaml
import json

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@tool
def medclip_tool(phrase: str) -> str:

    """
    Retrieve top medical image URLs related to the given phrase from a local database using MedCLIP.
    Accepts a medical phrase as input and returns a list of relevant image URLs with captions.
   """
    # Dummy image URLs (mocked as if retrieved from local MedCLIP DB)
    dummy_images = [
        {"url": "http://localhost:8000/images/image1.jpg", "title": "X-ray showing right lung opacity"},
        {"url": "http://localhost:8000/images/image2.jpg", "title": "Chest X-ray with pleural effusion"},
        {"url": "http://localhost:8000/images/image3.jpg", "title": "Normal chest X-ray for comparison"},
        {"url": "http://localhost:8000/images/image4.jpg", "title": "Lung cancer visible in upper lobe"},
        {"url": "http://localhost:8000/images/image5.jpg", "title": "Infiltrates seen in lower lobe"}
    ]
    
    return json.dumps({"images": dummy_images})
