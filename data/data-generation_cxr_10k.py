import os
import csv
from datasets import load_from_disk
from PIL import Image
from ollama import chat

# === Configurable Sample Count ===
count = None  # Set to any integer for testing, or None to process all

# === Paths and Config ===
save_dir = "cxr_reasoning_dataset"
image_dir = os.path.join(save_dir, "images")
csv_path = os.path.join(save_dir, "reasoning_dataset.csv")
prompt_template_path = "prompt.txt"
os.makedirs(image_dir, exist_ok=True)

# === Load prompt template ===
with open(prompt_template_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# === Load dataset ===
dataset = load_from_disk("mimic_cxr_10k_local")['train']

# === Load already processed image filenames from CSV ===
processed = set()
if os.path.exists(csv_path):
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["image_filename"])

# === CSV Setup ===
header = ["image_filename", "findings", "impression", "reasoning"]
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# === Loop through samples ===
for idx, example in enumerate(dataset):
    if count is not None and idx >= count:
        break

    image_filename = f"cxr_{idx:05}.png"
    if image_filename in processed:
        print(f"[{idx}] ⏭️ Skipped (already processed): {image_filename}")
        continue

    findings = example["findings"]
    impression = example["impression"]
    image = example["image"]

    if not findings or findings.strip() == "":
        continue

    image_path = os.path.join(image_dir, image_filename)
    image.save(image_path)

    # Build prompt
    full_prompt = prompt_template.replace("{{FINDINGS}}", findings.strip())
    messages = [{"role": "user", "content": full_prompt}]

    try:
        res = chat(model="amsaravi/medgemma-4b-it:q8", messages=messages)
        reasoning = res['message']['content'].strip()
    except Exception as e:
        print(f"[{idx}] ❌ Error: {e}")
        reasoning = "[LLM ERROR]"

    # Save to CSV
    with open(csv_path, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([image_filename, findings, impression, reasoning])

    print(f"[{idx}] ✅ Saved: {image_filename}")
