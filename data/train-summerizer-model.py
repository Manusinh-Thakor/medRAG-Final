import os
import pickle
import pandas as pd
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ─────────────────────────────────────────────
# 🔹 CONFIGURATION
# ─────────────────────────────────────────────

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cache_file = "pre_summariser_dataset_cache.pkl"
val_cache_file = "pre_summariser_val_data_cache.pkl"
csv_path = "reasoning_dataset.csv"
model_id = "google/medgemma-4b-it"

# ─────────────────────────────────────────────
# 🔹 STAGE 1: FORMAT DATA
# ─────────────────────────────────────────────

print("🔹 Stage 1: Loading and formatting summariser dataset")
df = pd.read_csv(csv_path)

def format_data(row):
    reasoning = row.get("reasoning", "").strip()
    impression = row.get("impression", "").strip()

    if not reasoning or not impression:
        raise ValueError("Missing reasoning or impression.")

    if "[LLM ERROR]" in reasoning:
        raise ValueError("Invalid reasoning content.")

    prompt = (
        "<s>[INST] Summarise the following clinical reasoning into a concise radiology impression:\n\n"
        f"{reasoning}\n"
        "[/INST] "
        f"{impression}</s>"
    )

    return {"text": prompt}

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        output = pickle.load(f)
    print(f"✅ Loaded {len(output)} examples from cache.")
else:
    output = []
    for i, row in df.iterrows():
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
# 🔹 STAGE 2: LOAD MODEL AND TOKENIZER
# ─────────────────────────────────────────────

print("🔹 Stage 2: Loading model and tokenizer")

model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
)

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Stage 2 complete: Model and tokenizer loaded")

# ─────────────────────────────────────────────
# 🔹 STAGE 3: COLLATOR
# ─────────────────────────────────────────────

def collate_fn(examples: list[dict[str, Any]]):
    texts = [ex["text"] for ex in examples]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch

print("✅ Stage 3 complete: Collator ready")

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
# 🔹 STAGE 5: SPLIT DATASET
# ─────────────────────────────────────────────

print("🔹 Stage 5: Splitting dataset")

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

test_cache_file = "test_summariser_data_cache.pkl"
if not os.path.exists(test_cache_file):
    with open(test_cache_file, "wb") as f:
        pickle.dump(test, f)
    print("✅ Saved test set to", test_cache_file)

# ─────────────────────────────────────────────
# 🔹 STAGE 6: TRAINING CONFIGURATION
# ─────────────────────────────────────────────

print("🔹 Stage 6: Configuring training")

args = SFTConfig(
    output_dir="medgemma-4b-it-sft-lora-cxr-10k-summariser",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
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

print("✅ Stage 6 complete")

# ─────────────────────────────────────────────
# 🔹 STAGE 7: TRAINING
# ─────────────────────────────────────────────

print("🔹 Stage 7: Starting training")

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    peft_config=peft_config,
    processing_class=tokenizer, 
    data_collator=collate_fn,
)


trainer.train()

print("✅ Training complete")
