import os
import json
import pickle
import pandas as pd
from typing import Any
from PIL import Image, UnidentifiedImageError

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ─────────────────────────────────────────────
# 🔹 CONFIGURATION
# ─────────────────────────────────────────────

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cache_file = "formatted_reasoning_dataset_cache.pkl"
val_cache_file = "validation_reasoning_data_cache.pkl"
response_csv_path = "cxr_reasoning_dataset/reasoning_dataset.csv"
image_folder = "cxr_reasoning_dataset/images"
model_id = "google/medgemma-4b-it"

# ─────────────────────────────────────────────
# 🔹 STAGE 1: DATASET FORMATTING
# ─────────────────────────────────────────────

print("🔹 Stage 1: Loading and formatting dataset")
df = pd.read_csv(response_csv_path)

def format_data(row):
    image_filename = row["image_filename"]
    image_path = os.path.join(image_folder, image_filename)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, UnicodeDecodeError) as e:
        raise RuntimeError(f"Corrupted image: {image_path} — {e}")

    assistant_response = row["reasoning"]
    if not assistant_response or not isinstance(assistant_response, str) or "LLM ERROR" in assistant_response:
        raise ValueError(f"Missing or invalid reasoning for image {image_filename}")

    return {
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Analyze this medical image and provide step-by-step findings."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_response}
                ]
            }
        ]
    }

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        output = pickle.load(f)
    print(f"✅ Loaded {len(output)} examples from cache.")
else:
    print("🔍 Checking image file availability before processing...")
    available_df = df[df["image_filename"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))].copy()
    print(f"✅ Found {len(available_df)} available images out of {len(df)}")

    output = []
    for i, row in available_df.iterrows():
        try:
            formatted = format_data(row)
            output.append(formatted)
        except Exception as e:
            print(f"[{i}] Skipping due to error: {e}")
    
    with open(cache_file, "wb") as f:
        pickle.dump(output, f)
    print(f"✅ Stage 1 complete: Converted and cached {len(output)} examples")

print(output[0])
print(f"✅ Stage 1 complete: Successfully formatted {len(output)} examples")

# ─────────────────────────────────────────────
# 🔹 STAGE 2: LOAD MODEL & PROCESSOR
# ─────────────────────────────────────────────

print("🔹 Stage 2: Loading model and processor")

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    ),
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "right"

print("✅ Stage 2 complete: Model and processor loaded")

# ─────────────────────────────────────────────
# 🔹 STAGE 3: DATA COLLATOR
# ───────────────────────────────────────────── 

print("🔹 Stage 3: Preparing data collator")

def collate_fn(examples: list[dict[str, Any]]):
    texts = []
    images = []

    for example in examples:
        # Ensure image is included once
        images.append([example["image"].convert("RGB")])
        
        # Generate chat-formatted prompt
        formatted_text = processor.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False
        ).strip()

        # 🛡️ Fix: Ensure exactly ONE <image> token
        # This trims repeated ones (sometimes HuggingFace bugs or bad templates create many)
        if formatted_text.count("<image>") > 1:
            first = formatted_text.find("<image>")
            formatted_text = formatted_text[:first] + "<image>" + formatted_text[first:].replace("<image>", "", 1)

        texts.append(formatted_text)

    # Prepare inputs
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    labels = batch["input_ids"].clone()

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
    pad_token_id = processor.tokenizer.pad_token_id

    labels[labels == pad_token_id] = -100
    if isinstance(image_token_id, int):
        labels[labels == image_token_id] = -100

    # Safety: remove very large token ids
    labels[labels >= processor.tokenizer.vocab_size] = -100
    labels[labels < 0] = -100

    batch["labels"] = labels
    return batch


print("✅ Stage 3 complete: Data collator ready")

# ─────────────────────────────────────────────
# 🔹 STAGE 4: LoRA CONFIGURATION
# ─────────────────────────────────────────────

print("🔹 Stage 4: Applying LoRA configuration")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

print("✅ Stage 4 complete: LoRA config applied")


# ─────────────────────────────────────────────
# 🔹 STAGE 5: DATASET SPLIT
# ─────────────────────────────────────────────

print("🔹 Stage 5: Splitting dataset into train and validation")
# Split: 90% train, 5% val, 5% test
n = len(output)
train_end = int(0.9 * n)
val_end = int(0.95 * n)

train = output[:train_end]
val = output[train_end:val_end]
test = output[val_end:]

if not os.path.exists(val_cache_file):
    with open(val_cache_file, "wb") as f:
        pickle.dump(val, f)
    print("✅ Saved validation set to", val_cache_file)

test_cache_file = "test_reasoning_data_cache.pkl"
if not os.path.exists(test_cache_file):
    with open(test_cache_file, "wb") as f:
        pickle.dump(test, f)
    print("✅ Saved test set to", test_cache_file)


print(f"✅ Stage 5 complete: Train size = {len(train)}, Validation size = {len(val)}")


# ─────────────────────────────────────────────
# 🔹 STAGE 6: TRAINING CONFIGURATION
# ─────────────────────────────────────────────

print("🔹 Stage 6: Configuring training settings")

args = SFTConfig(
    output_dir="medgemma-4b-it-sft-lora-cxr-10k-reasoning",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
    eval_steps=50,
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=True,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"],
)

print("✅ Stage 6 complete: Training configuration set")

# ─────────────────────────────────────────────
# 🔹 STAGE 7: TRAINING
# ─────────────────────────────────────────────

print("🔹 Stage 7: Initializing trainer and starting training")

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

trainer.train()

print("✅ Stage 7 complete: Training finished")
