from langchain_core.tools import tool
from tavily import TavilyClient
import os
import yaml
import json

# Load config
def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()
os.environ["TAVILY_API_KEY"] = cfg["tools"]["websearch"]["tavily_api_key"]
tavily_client = TavilyClient()

@tool
def websearch_tool(phrase: str) -> str:
    """
    Retrieve top 5 images for a medical phrase using websearch.
    Returns a JSON string with list of {"url": ..., "title": ...}.
    """
    response = tavily_client.search(query=phrase, max_results=5,include_images=True, include_image_descriptions=True)
    results = response.get("results", [])

    image_list = []
    for r in results:
        url = r.get("url")
        title = r.get("title", "")
        if url:
            image_list.append({"url": url, "title": title})
        if len(image_list) == 5:
            break

    return json.dumps({"images": image_list})
