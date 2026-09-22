SpatialSense is well-suited for this benchmark because it was curated specifically to eliminate 2D spatial language biases and force models to reason about actual 3D physical arrangements.

---

### Understanding the SpatialSense Annotation Format

SpatialSense annotations are structured as relational tuples between a subject and an object bounding box:

```json
{
  "image_id": "234567.jpg",
  "url": "http://...",
  "predicate": "behind",          // Target relationship (e.g., 'behind', 'in front of', 'closer than')
  "label": 1,                      // 1 = True (predicate holds), 0 = False (does not hold)
  "subject": {
    "name": "chair",
    "bbox": [ymin, xmin, ymax, xmax]  // 0 to 1 normalized coordinates or absolute pixels
  },
  "object": {
    "name": "table",
    "bbox": [ymin, xmin, ymax, xmax]
  }
}

```

The key predicates that depend directly on depth reasoning are:

* `in front of` vs. `behind`
* `closer than` vs. `further than`
* `under` vs. `on` (partially assisted by depth support planes)

---

### Step 1: The A/B Experimental Setup

To prove that injecting depth improves the model, compare performance across three parallel conditions using identical prompts:

* **Condition A (RGB Baseline):** Provide only the original image crop/context.
* **Condition B (RGB + Depth Map as Multi-Image):** Provide the RGB image and the predicted depth map side-by-side or as two image inputs into the VLM (e.g., Qwen2-VL, InternVL).
* **Condition C (RGB + Explicit Depth Extraction):** Use your depth model to sample median depth inside the subject bbox ($d_{sub}$) and object bbox ($d_{obj}$), then inject those distances directly into the prompt text.

---

### Step 2: Formulating the Evaluation Prompt

Use a strict binary-choice or multiple-choice prompt template to enable zero-shot accuracy parsing without open-ended generation noise:

```text
Given the image, consider the two items:
- Subject: {subject_name} at coordinates {subject_bbox}
- Object: {object_name} at coordinates {object_bbox}

Question: Is the statement "{subject_name} is {predicate} the {object_name}" correct?
Answer only with "Yes" or "No".

```

For **Condition C (Textual Depth Prompting)**, augment the system context:

```text
Additional spatial measurement from depth sensor:
- Estimated distance to {subject_name}: {d_sub:.2f} meters
- Estimated distance to {object_name}: {d_obj:.2f} meters

Question: Is the statement "{subject_name} is {predicate} the {object_name}" correct?
Answer only with "Yes" or "No".

```

---

### Step 3: Python Benchmark Evaluation Loop

This pipeline loads a sample from SpatialSense, extracts regional depth, and benchmarks the VLM across baseline and depth-augmented modes:

```python
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 1. Load your VLM (e.g., Qwen2-VL)
model_name = "Qwen/Qwen2-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)


def get_bbox_median_depth(depth_map, bbox, img_w, img_h):
    """bbox: [ymin, xmin, ymax, xmax] in normalized (0-1) coordinates."""
    ymin, xmin, ymax, xmax = bbox
    y1, y2 = int(ymin * img_h), int(ymax * img_h)
    x1, x2 = int(xmin * img_w), int(xmax * img_w)

    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return float(np.nanmedian(depth_map))
    return float(np.nanmedian(crop))


def run_vlm_query(image_list, text_prompt):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in image_list]
            + [{"type": "text", "text": text_prompt}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text], images=image_list, padding=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


# 2. Evaluation Step for a Single SpatialSense Item
def evaluate_sample(image_path, depth_map, sample):
    raw_img = Image.open(image_path).convert("RGB")
    w, h = raw_img.size

    sub_name = sample["subject"]["name"]
    obj_name = sample["object"]["name"]
    predicate = sample["predicate"]
    gt_label = "Yes" if sample["label"] == 1 else "No"

    # Sample depths inside the two target bounding boxes
    d_sub = get_bbox_median_depth(depth_map, sample["subject"]["bbox"], w, h)
    d_obj = get_bbox_median_depth(depth_map, sample["object"]["bbox"], w, h)

    # Condition 1: Baseline (RGB Only)
    prompt_baseline = (
        f"Is the {sub_name} {predicate} the {obj_name}? Answer with 'Yes' or 'No'."
    )
    ans_baseline = run_vlm_query([raw_img], prompt_baseline)

    # Condition 2: Depth-Informed Text Prompt
    prompt_depth = (
        f"Spatial depth sensor readings:\n"
        f"- {sub_name} distance: {d_sub:.2f} m\n"
        f"- {obj_name} distance: {d_obj:.2f} m\n\n"
        f"Based on visual and depth data, is the {sub_name} {predicate} the {obj_name}? "
        f"Answer with 'Yes' or 'No'."
    )
    ans_depth_augmented = run_vlm_query([raw_img], prompt_depth)

    return {
        "predicate": predicate,
        "gt": gt_label,
        "pred_baseline": "Yes" if "yes" in ans_baseline.lower() else "No",
        "pred_depth": "Yes" if "yes" in ans_depth_augmented.lower() else "No",
    }

```

---

### Metrics to Track

When tabulating your results across the SpatialSense test split, isolate:

* **Depth-Critical Accuracy:** Accuracy strictly on `behind` and `in front of` relationships.
* **Non-Depth Baseline Accuracy:** Accuracy on predicates like `left of`, `right of`, or `above` (as a control group to verify the depth injection didn't disrupt 2D coordinate understanding).
* **Flip Rate:** The percentage of examples where the baseline answered incorrectly, but providing the depth estimate flipped the answer to correct.SpatialSense is well-suited for this benchmark because it was curated specifically to eliminate 2D spatial language biases and force models to reason about actual 3D physical arrangements.

---

### Understanding the SpatialSense Annotation Format

SpatialSense annotations are structured as relational tuples between a subject and an object bounding box:

```json
{
  "image_id": "234567.jpg",
  "url": "http://...",
  "predicate": "behind",          // Target relationship (e.g., 'behind', 'in front of', 'closer than')
  "label": 1,                      // 1 = True (predicate holds), 0 = False (does not hold)
  "subject": {
    "name": "chair",
    "bbox": [ymin, xmin, ymax, xmax]  // 0 to 1 normalized coordinates or absolute pixels
  },
  "object": {
    "name": "table",
    "bbox": [ymin, xmin, ymax, xmax]
  }
}

```

The key predicates that depend directly on depth reasoning are:

* `in front of` vs. `behind`
* `closer than` vs. `further than`
* `under` vs. `on` (partially assisted by depth support planes)

---

### Step 1: The A/B Experimental Setup

To prove that injecting depth improves the model, compare performance across three parallel conditions using identical prompts:

* **Condition A (RGB Baseline):** Provide only the original image crop/context.
* **Condition B (RGB + Depth Map as Multi-Image):** Provide the RGB image and the predicted depth map side-by-side or as two image inputs into the VLM (e.g., Qwen2-VL, InternVL).
* **Condition C (RGB + Explicit Depth Extraction):** Use your depth model to sample median depth inside the subject bbox ($d_{sub}$) and object bbox ($d_{obj}$), then inject those distances directly into the prompt text.

---

### Step 2: Formulating the Evaluation Prompt

Use a strict binary-choice or multiple-choice prompt template to enable zero-shot accuracy parsing without open-ended generation noise:

```text
Given the image, consider the two items:
- Subject: {subject_name} at coordinates {subject_bbox}
- Object: {object_name} at coordinates {object_bbox}

Question: Is the statement "{subject_name} is {predicate} the {object_name}" correct?
Answer only with "Yes" or "No".

```

For **Condition C (Textual Depth Prompting)**, augment the system context:

```text
Additional spatial measurement from depth sensor:
- Estimated distance to {subject_name}: {d_sub:.2f} meters
- Estimated distance to {object_name}: {d_obj:.2f} meters

Question: Is the statement "{subject_name} is {predicate} the {object_name}" correct?
Answer only with "Yes" or "No".

```

---

### Step 3: Python Benchmark Evaluation Loop

This pipeline loads a sample from SpatialSense, extracts regional depth, and benchmarks the VLM across baseline and depth-augmented modes:

```python
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 1. Load your VLM (e.g., Qwen2-VL)
model_name = "Qwen/Qwen2-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)


def get_bbox_median_depth(depth_map, bbox, img_w, img_h):
    """bbox: [ymin, xmin, ymax, xmax] in normalized (0-1) coordinates."""
    ymin, xmin, ymax, xmax = bbox
    y1, y2 = int(ymin * img_h), int(ymax * img_h)
    x1, x2 = int(xmin * img_w), int(xmax * img_w)

    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return float(np.nanmedian(depth_map))
    return float(np.nanmedian(crop))


def run_vlm_query(image_list, text_prompt):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in image_list]
            + [{"type": "text", "text": text_prompt}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text], images=image_list, padding=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


# 2. Evaluation Step for a Single SpatialSense Item
def evaluate_sample(image_path, depth_map, sample):
    raw_img = Image.open(image_path).convert("RGB")
    w, h = raw_img.size

    sub_name = sample["subject"]["name"]
    obj_name = sample["object"]["name"]
    predicate = sample["predicate"]
    gt_label = "Yes" if sample["label"] == 1 else "No"

    # Sample depths inside the two target bounding boxes
    d_sub = get_bbox_median_depth(depth_map, sample["subject"]["bbox"], w, h)
    d_obj = get_bbox_median_depth(depth_map, sample["object"]["bbox"], w, h)

    # Condition 1: Baseline (RGB Only)
    prompt_baseline = (
        f"Is the {sub_name} {predicate} the {obj_name}? Answer with 'Yes' or 'No'."
    )
    ans_baseline = run_vlm_query([raw_img], prompt_baseline)

    # Condition 2: Depth-Informed Text Prompt
    prompt_depth = (
        f"Spatial depth sensor readings:\n"
        f"- {sub_name} distance: {d_sub:.2f} m\n"
        f"- {obj_name} distance: {d_obj:.2f} m\n\n"
        f"Based on visual and depth data, is the {sub_name} {predicate} the {obj_name}? "
        f"Answer with 'Yes' or 'No'."
    )
    ans_depth_augmented = run_vlm_query([raw_img], prompt_depth)

    return {
        "predicate": predicate,
        "gt": gt_label,
        "pred_baseline": "Yes" if "yes" in ans_baseline.lower() else "No",
        "pred_depth": "Yes" if "yes" in ans_depth_augmented.lower() else "No",
    }

```

---

### Metrics to Track

When tabulating your results across the SpatialSense test split, isolate:

* **Depth-Critical Accuracy:** Accuracy strictly on `behind` and `in front of` relationships.
* **Non-Depth Baseline Accuracy:** Accuracy on predicates like `left of`, `right of`, or `above` (as a control group to verify the depth injection didn't disrupt 2D coordinate understanding).
* **Flip Rate:** The percentage of examples where the baseline answered incorrectly, but providing the depth estimate flipped the answer to correct.