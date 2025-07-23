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
        return """1.  **Technical Assessment:**
    *   Image Type: PA Chest X-ray
    *   Positioning: Standard
    *   Technical Quality: Good

2.  **Visual Findings:**
    *   Lungs: No obvious consolidation, nodules, or pleural effusions.
    *   Pleura: No visible pleural effusion or pneumothorax.
    *   Heart/Mediastinum: Heart size appears within normal limits. No mediastinal shift.
    *   Diaphragm/Bones: Diaphragms are visible. No obvious fractures or calcifications.

3.  **Abnormalities and Interpretation:**
    *   No focal consolidation, nodules, or pleural effusions are identified.
    *   The heart size appears within normal limits.
    *   The mediastinum is unremarkable.
    *   The diaphragms are visible.

4.  **Diagnostic Reasoning:**
    *   The absence of focal consolidation, nodules, or pleural effusions suggests that there are no obvious infectious or inflammatory processes affecting the lungs or pleura.
    *   The heart size appears within normal limits, and there is no mediastinal shift, which suggests that there is no significant cardiac enlargement or compression of the mediastinum.
    *   The diaphragms are visible, which confirms that they are not obscured by any pathology.

5.  **Final Diagnosis:**
    *   No acute abnormalities detected.

6.  **Confidence Level:**
    *   High

7.  **Supporting Evidence:**
    *   The image shows no obvious signs of pneumonia, pulmonary edema, or pleural effusion.
    *   The heart size appears within normal limits, and there is no mediastinal shift.
    *   The diaphragms are visible, which confirms that they are not obscured by any pathology.

8.  **Limitations:**
    *   The image may not be sensitive enough to detect subtle abnormalities.
    *   The image may be affected by factors such as patient positioning or technical artifacts.

9.  **Conclusion:**
    *   The chest X-ray shows no acute abnormalities.
    *   The heart size appears within normal limits.
    *   The mediastinum is unremarkable.
    *   The diaphragms are visible.
    *   No focal consolidation, nodules, or pleural effusions are identified.
    *   The absence of focal consolidation, nodules, or pleural effusions suggests that there"""
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "prompt": "Analyze this medical image and provide step-by-step findings.",
            "image_base64": image_base64
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post("http://3.24.74.65:8000/predict", json=payload, headers=headers)


        if response.status_code == 200:
            return response.json().get("response", "No response field found.")
        else:
            return f"Error: Received status code {response.status_code} from MedGEMMA API."

    except Exception as e:
        return f"Error during image analysis: {str(e)}"
