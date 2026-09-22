import json
import os
import re
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Paths
ANNOTATIONS_FILE = "data/annotations.json"
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "qwen2/qwen2_vl_spatialsense_lora/final_adapter"
OUTPUT_REPORT = "qwen2/qwen2_vl_spatialsense_lora/test_evaluation_report.json"

# 1. Load Processor and Base Model with LoRA Adapter
print("Loading model and adapter...")
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 2. Extract Test Samples
with open(ANNOTATIONS_FILE, "r") as f:
    raw_data = json.load(f)

test_samples = []
for entry in raw_data:
    if entry.get("split") != "test":
        continue

    img_filename = entry.get("filename") or entry.get("url", "").split("/")[-1]
    if "nyu" in img_filename.lower():
        img_path = os.path.join("data/images/nyu/", img_filename)
    else:
        img_path = os.path.join("data/images/flickr/", img_filename)

    if not os.path.exists(img_path):
        continue

    for anno in entry.get("annotations", []):
        test_samples.append({
            "image_path": img_path,
            "subject": anno["subject"]["name"],
            "object": anno["object"]["name"],
            "predicate": anno["predicate"],
            "gt_label": "Yes" if anno["label"] is True else "No"
        })

print(f"Loaded {len(test_samples)} test samples.")

def parse_model_response(raw_text):
    """Cleans punctuation and extracts first binary token."""
    cleaned = re.sub(r"[^\w\s]", "", raw_text).strip().lower().split()
    if not cleaned:
        return "Unknown"
    first = cleaned[0]
    if first in ("yes", "true"):
        return "Yes"
    if first in ("no", "false"):
        return "No"
    return "Unknown"

# 3. Inference Evaluation Loop
results = []
y_true = []
y_pred = []

for sample in tqdm(test_samples, desc="Running Test Evaluation"):
    image = Image.open(sample["image_path"]).convert("RGB")
    
    prompt = (
        f"Examine the spatial arrangement of the image.\n"
        f"Question: Is the {sample['subject']} {sample['predicate']} the {sample['object']}?\n"
        f"Answer with strictly 'Yes' or 'No'."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=4, 
            do_sample=False  # Deterministic greedy decoding
        )

    # Decode only the generated response tokens
    gen_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
    pred_raw = processor.decode(gen_tokens, skip_special_tokens=True).strip()
    pred_label = parse_model_response(pred_raw)

    y_true.append(sample["gt_label"])
    y_pred.append(pred_label)

    results.append({
        "image": os.path.basename(sample["image_path"]),
        "subject": sample["subject"],
        "predicate": sample["predicate"],
        "object": sample["object"],
        "gt": sample["gt_label"],
        "pred": pred_label,
        "correct": (pred_label == sample["gt_label"])
    })

# 4. Metrics Computation
df = pd.DataFrame(results)

# Overall Accuracy
overall_acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", labels=["Yes", "No"], zero_division=0
)

print("\n" + "=" * 50)
print(f"OVERALL EVALUATION RESULTS (Test Split)")
print("=" * 50)
print(f"Total Samples Tested: {len(df)}")
print(f"Accuracy:  {overall_acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"Macro F1:  {f1 * 100:.2f}%\n")

# Breakdown by Predicate
print("--- ACCURACY BY PREDICATE ---")
predicate_summary = df.groupby("predicate")["correct"].agg(["count", "mean"]).rename(
    columns={"count": "Samples", "mean": "Accuracy"}
)
predicate_summary["Accuracy"] = (predicate_summary["Accuracy"] * 100).round(2)
print(predicate_summary.sort_values(by="Samples", ascending=False).to_string())

# Depth-specific predicates
depth_preds = df[df["predicate"].isin(["behind", "in front of"])]
if not depth_preds.empty:
    depth_acc = (depth_preds["correct"].mean()) * 100
    print("\n" + "-" * 50)
    print(f"Depth-Critical Accuracy ('behind' / 'in front of'): {depth_acc:.2f}%")
    print("-" * 50)

# Save Report
summary_data = {
    "overall_accuracy": overall_acc,
    "macro_f1": f1,
    "predicate_breakdown": predicate_summary.to_dict(orient="index"),
    "detailed_results": results
}

with open(OUTPUT_REPORT, "w") as f:
    json.dump(summary_data, f, indent=2)

print(f"\nDetailed evaluation report saved to: {OUTPUT_REPORT}")