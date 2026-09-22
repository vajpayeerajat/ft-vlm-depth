import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from peft import LoraConfig, get_peft_model
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# Paths
ANNOTATIONS_FILE = "data/annotations.json"
IMAGE_DIR = "data/images/flickr/"
DEPTH_DIR = "data/depths/raw_npy/"
OUTPUT_DIR = "qwen2/qwen2_vl_spatialsense_lora"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------
# 1. Dataset Preparation
# -------------------------------------------------------------
class SpatialSenseDataset(Dataset):
  def __init__(self, json_path, image_dir, split="train"):
    self.image_dir = image_dir
    with open(json_path, "r") as f:
      raw_data = json.load(f)
    self.samples = []
    for entry in raw_data:
      if entry.get("split") != split:
        continue
      img_filename = entry.get("filename") or entry.get("url", "").split(
          "/"
      )[-1]
      depth_filename = img_filename.replace(".jpg", ".npy").replace(".png", ".npy")
      depth_path = os.path.join(DEPTH_DIR, depth_filename)
      # Load pre-cached depth information 
      depth_data = np.load(depth_path, allow_pickle=True)
      if "nyu" in img_filename.lower():
        img_path = os.path.join("data/images/nyu/", img_filename)
      else:
        img_path = os.path.join("data/images/flickr/", img_filename)
      if not os.path.exists(img_path):
        continue
      for anno in entry.get("annotations", []):
        sub = anno["subject"]["name"]
        obj = anno["object"]["name"]
        pred = anno["predicate"]
        label_text = "Yes" if anno["label"] is True else "No"
        self.samples.append({
            "image_path": img_path,
            "subject": sub,
            "object": obj,
            "predicate": pred,
            "answer": label_text,
            "depth": depth_data,
        })
    print(f"Loaded {len(self.samples)} valid samples for split: '{split}'")
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, idx):
    item = self.samples[idx]
    image = Image.open(item["image_path"]).convert("RGB")
    prompt = (
        "Examine the spatial arrangement of the image.\n"
        f"Question: Is the {item['subject']} {item['predicate']} the"
        f" {item['object']}?\n"
        "Answer with strictly 'Yes' or 'No'."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": item["answer"],
        },
    ]
    return {
        "image": image,
        "messages": messages,
        "answer": item["answer"],
    }


# -------------------------------------------------------------
# 2. Train/Val Split
# -------------------------------------------------------------
full_train_dataset = SpatialSenseDataset(
    ANNOTATIONS_FILE, IMAGE_DIR, split="train"
)

val_size = int(0.10 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size

train_dataset, eval_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

print(f"Training split size: {len(train_dataset)}")
print(f"Validation split size: {len(eval_dataset)}")

# -------------------------------------------------------------
# 3. Model & Processor Setup with LoRA
# -------------------------------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -------------------------------------------------------------
# 4. Collate Function (Robust Target Masking)
# -------------------------------------------------------------
def collate_fn(batch):
  images = [item["image"] for item in batch]
  texts = [
      processor.apply_chat_template(item["messages"], tokenize=False)
      for item in batch
  ]

  batch_inputs = processor(
      text=texts,
      images=images,
      padding=True,
      return_tensors="pt",
  )

  labels = torch.full_like(batch_inputs["input_ids"], -100)

  for i, item in enumerate(batch):
    ans_ids = processor.tokenizer(item["answer"], add_special_tokens=False)[
        "input_ids"
    ]
    seq = batch_inputs["input_ids"][i].tolist()
    k = len(ans_ids)

    for idx in range(len(seq) - k, -1, -1):
      if seq[idx : idx + k] == ans_ids:
        labels[i, idx : idx + k] = torch.tensor(ans_ids)
        break

  batch_inputs["labels"] = labels
  return batch_inputs

# -------------------------------------------------------------
# 5. Training Arguments and Execution
# -------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2.5e-4,
    num_train_epochs=2,
    logging_strategy="epoch",  # Logs training loss per epoch to match validation
    eval_strategy="epoch",  # Run evaluation at the end of each epoch
    save_strategy="epoch",  # Save checkpoint at the end of each epoch
    save_total_limit=2,
    load_best_model_at_end=True,  # Retains checkpoint with best eval loss
    metric_for_best_model="eval_loss",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="tensorboard",  # Changes 'none' to 'tensorboard'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

print("\n--- Starting Training ---")
trainer.train()

trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
print(f"\nAdapter weights saved to: {OUTPUT_DIR}/final_adapter")

# -------------------------------------------------------------
# 6. Extract Epoch Logs and Plot Train vs Val Loss
# -------------------------------------------------------------
log_history = trainer.state.log_history

train_epochs, train_losses = [], []
eval_epochs, eval_losses = [], []

for entry in log_history:
  if "loss" in entry and "epoch" in entry:
    train_epochs.append(round(entry["epoch"]))
    train_losses.append(entry["loss"])
  if "eval_loss" in entry and "epoch" in entry:
    eval_epochs.append(round(entry["epoch"]))
    eval_losses.append(entry["eval_loss"])

# Save loss records to CSV
df_train = pd.DataFrame({"epoch": train_epochs, "train_loss": train_losses})
df_eval = pd.DataFrame({"epoch": eval_epochs, "eval_loss": eval_losses})
df_logs = pd.merge(df_train, df_eval, on="epoch", how="outer").sort_values(
    "epoch"
)
df_logs.to_csv(os.path.join(OUTPUT_DIR, "epoch_loss_history.csv"), index=False)

# Plot Epoch-based Curves
plt.figure(figsize=(8, 5), dpi=150)
if train_epochs:
  plt.plot(
      train_epochs,
      train_losses,
      label="Train Loss",
      color="royalblue",
      lw=2,
      marker="o",
  )
if eval_epochs:
  plt.plot(
      eval_epochs,
      eval_losses,
      label="Validation Loss",
      color="crimson",
      lw=2,
      marker="s",
  )

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xticks(sorted(list(set(train_epochs + eval_epochs))))
plt.title(
    "SpatialSense Qwen2-VL LoRA: Epoch-wise Loss Curve", fontsize=13
)
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

curve_path = os.path.join(OUTPUT_DIR, "epoch_learning_curve.png")
plt.savefig(curve_path)
plt.close()

print(f"Loss curve saved to: {curve_path}")
print(f"Epoch log saved to: {OUTPUT_DIR}/epoch_loss_history.csv")
