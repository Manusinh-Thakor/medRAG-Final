import base64
import os
import pandas as pd
from PIL import Image
from openai import OpenAI
import ollama
import io
import shutil

ollama_flag = True
client = OpenAI()

# === CONFIG ===
image_basepath = "../NIH_dataset/nih_16k_dataset/nih_balanced_images"
labels_file = "../NIH_dataset/nih_balanced_filtered.csv"
response_dir = "text_response"
new_images_folder = "resized_images"  # New folder for resized images
max_image_size_mb = 5

# Create the new images folder if it doesn't exist
os.makedirs(new_images_folder, exist_ok=True)

# === PROMPT ===
prompt = """
You are a clinical radiology assistant. Analyze the provided chest X-ray image and its ground-truth diagnosis label to generate a structured, evidence-based reasoning report (300-400 tokens). Proceed directly to the analysis without introductory text. Follow this exact format:*

1. Technical Assessment
Briefly note image type (e.g., PA chest X-ray), positioning adequacy, and technical artifacts (if any).

2. Visual Findings
Lungs: Opacities, nodules, consolidation, volume changes.

Pleura: Effusion, pneumothorax, thickening.

Heart/Mediastinum: Cardiomegaly, border obscuration, mediastinal shift.

Diaphragm/Bones: Elevation, fractures, calcifications.

3. Abnormalities & Interpretation
List key abnormal features and their clinical implications (e.g., "Right lower lobe consolidation suggests pneumonia").

4. Diagnostic Reasoning
Link findings directly to each label component (e.g., "Hyperinflated lungs and flattened diaphragms support emphysema").

5. Conclusion
Restate the diagnosis label and summarize radiographic evidence (e.g., "Final diagnosis: Pneumonia—supported by right mid-zone consolidation and air bronchograms").

Rules:

Be concise (strictly 300-400 tokens).

Never state readiness—begin analysis immediately.

Only use findings visible in the image; no speculation.

Cover every diagnosis in the label.

Now analyze the image and label provided."

"""

# === PREPARE ===
df = pd.read_csv(labels_file)
image_col, label_col = df.columns[0], df.columns[1]
image_list, labels = df[image_col].tolist(), df[label_col].tolist()

os.makedirs(response_dir, exist_ok=True)

def encode_image(image_path):
    # Resize image to 512x512 before encoding
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)
        
        # Save the resized image to the new folder with the same name
        new_image_path = os.path.join(new_images_folder, os.path.basename(image_path))
        img.save(new_image_path, format="JPEG")
        
        # Save to a temporary buffer for encoding
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

def is_image_valid(image_path):
    try:
        if os.path.getsize(image_path) > max_image_size_mb * 1024 * 1024:
            return False, "Too large"
        with Image.open(image_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)

def process_image(image_path, label):
    base64_image = encode_image(image_path)

    if ollama_flag:
        res = ollama.chat(
        model="amsaravi/medgemma-4b-it:q8",
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }
        ]
        )

        response = res['message']['content']
        return response

    else:  
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n{label}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content

num_to_process = None
total_images = len(image_list)
process_count = num_to_process if num_to_process is not None else total_images

for idx, (image, label) in enumerate(zip(image_list, labels), start=1):
    if idx > process_count:
        break

    image_name = os.path.splitext(os.path.basename(image))[0]
    image_path_full = os.path.join(image_basepath, image)
    response_file = os.path.join(response_dir, f"{image_name}.txt")

    print(f"[{idx}/{total_images}] Processing: {image}")

    if os.path.exists(response_file):
        print(f" → Skipped (already processed)")
        continue

    if not os.path.exists(image_path_full):
        print(f" → Skipped (not found)")
        continue

    valid, reason = is_image_valid(image_path_full)
    if not valid:
        print(f" → Skipped ({reason})")
        continue

    try:
        response_text = process_image(image_path_full, label)
        with open(response_file, "w", encoding="utf-8") as rf:
            rf.write(response_text)
    except Exception as e:
        print(f" → Error: {e}")

print(f"\n✅ Processed {min(process_count, total_images)} image(s). Responses saved to '{response_dir}'")
print(f"Resized images saved to '{new_images_folder}'")


