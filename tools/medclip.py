from langchain_core.tools import tool
import os
import yaml
import json
import boto3

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()

DUMMY_MODE = True 

@tool
def medclip_tool(phrase: str) -> str:
    """
    Retrieve top medical image URLs related to the given phrase from a local database using MedCLIP.
    Accepts a medical phrase as input and returns a list of relevant image URLs with captions.
    """

    if DUMMY_MODE:
        dummy_images = [
            "unique_subset/00028948_004.png",
            "unique_subset/00006462_007.png",
            "unique_subset/00011379_035.png",
            "unique_subset/00000193_019.png",
            "unique_subset/00028544_004.png"]
        return json.dumps({"images": dummy_images})
    
    try:
        endpoint_name = "medclip-ednpoint"
        inference_component_name = "Model-1752579465574-20250722-0927130"

        runtime = boto3.Session().client("sagemaker-runtime", region_name="ap-southeast-2")
        payload = {
            "label": phrase,
            "k": 5
        }

        payload_bytes = json.dumps(payload).encode("utf-8")

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            InferenceComponentName=inference_component_name,
            Body=payload_bytes
        )

        result = response["Body"].read().decode("utf-8")
        return result

    except Exception as e:
        return json.dumps({"error": str(e)})
