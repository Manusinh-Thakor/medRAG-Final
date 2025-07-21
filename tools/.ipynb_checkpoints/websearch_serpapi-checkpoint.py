from langchain.agents import tool
import requests
import yaml
import json
import os
# Load config
def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()
serpapi_key = cfg["tools"]["websearch"]["serpapi_api_key"]

@tool
def serpapi_search_tool(q: str,  gl: str = "us", num: int = 5, search_type: str = "image") -> list:
    
    """
    Perform a Google search (web, image) via SerpApi and return results.

    search_type='image': When image or images is specified in the user query, the function performs an image search.
    search_type='text': For all other case where images not specified.

    Parameters:
        q (str): Search query.
        gl (str): Country code for the search (e.g., 'us' for USA).
        num (int): Number of results to return.-
        search_type (str): 'text' for web search, 'image' for image search.

    Returns:
        list: A list of result dictionaries. 
              If search_type='text': each dict has 'title' and 'url'. 
              If search_type='image': each dict has 'image_url' ,'source_url' and 'title'.
    """

    location = "Austin, Texas, United States"
    hl = "en"

    if search_type not in {"text", "image"}:
        raise ValueError(f"Invalid search_type '{search_type}'. Use 'text', 'image'")

    params = {
        "engine": "google",
        "q": q,
        "location": location,
        "hl": hl,
        "gl": gl,
        "num": num,
        "api_key": serpapi_key
    }
    
    if search_type == "image":
        params["tbm"] = "isch"

    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"SerpApi request failed: {e}")

    data = response.json()
    if "error" in data:
        raise RuntimeError(f"SerpApi error: {data['error']}")

    results = []
    for item in data.get("organic_results", [])[:num]:
        title = item.get("title")
        link = item.get("link")
        if title and link:
            results.append({"image_url": "", "source_url": link}) 

    for item in data.get("images_results", [])[:num]:
        image_url = item.get("original") or item.get("thumbnail")
        source_url = item.get("link")
        title = item.get("title")
        if source_url:
            results.append({"image_url": image_url, "source_url": source_url, "title": title})

    return results

