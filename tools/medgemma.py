from langchain_core.tools import tool
import os
import yaml
import json
import base64
import requests

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()
dummy = True

@tool
def medgemma_tool(image_path: str, ) -> str:
    """
    Analyze a medical image and return reasoning answer with a final diagnosis.
    Accepts a local image path and outputs medical reasoning and conclusion.
    """
    
    if dummy:
        return """The chest X-ray shows a posteroanterior (PA) view of the thorax with proper inspiration and no significant rotation. The trachea appears central, and the cardiac silhouette is within  normal limits. The bony structures, including ribs, clavicles, and shoulders, appear intact.

On closer inspection of the lung fields, the left lung appears clear, but there is a distinct abnormality in the right lung. Specifically, in the mid-lung zone of the right lung field, there is a solitary, dense opacity. This lesion is relatively well defined but irregular in shape. There are no visible air bronchograms within the lesion, which reduces the likelihood of it being a simple pneumonia or consolidation. Furthermore, there is no evidence of cavitation or calcification, which argues against conditions such as tuberculosis or a benign granuloma.

The absence of features such as satellite nodules, calcified borders, or central cavitation makes infectious causes like TB less likely. Similarly, the lesion does not show characteristics of benign tumors like hamartoma, which often appear as well-circumscribed nodules with "popcorn" calcification.

Given the solitary nature of the lesion, its size, and its irregular borders, a neoplastic process must be considered. The most likely diagnosis based on these radiographic findings is a primary lung malignancy, such as bronchogenic carcinoma. This is especially concerning in the context of a middle-aged or older adult, particularly if there is a history of smoking.

In conclusion, the X-ray demonstrates a solitary pulmonary mass in the right mid-lung zone, with radiographic features most consistent with a primary bronchogenic carcinoma. Further imaging with contrast-enhanced CT and tissue biopsy would be necessary to confirm the diagnosis and guide management."""
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "prompt": "Analyze this medical image and provide step-by-step findings.",
            "image_base64": image_base64
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post("http://3.27.23.79:8000/predict", json=payload, headers=headers)

        if response.status_code == 200:
            return response.json().get("response", "No response field found.")
        else:
            return f"Error: Received status code {response.status_code} from MedGEMMA API."

    except Exception as e:
        return f"Error during image analysis: {str(e)}"
