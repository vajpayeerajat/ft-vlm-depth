import gc
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
ANNOTATIONS_FILE = "data/annotations.json"
OUTPUT_DIR = "data/depths"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
BATCH_SIZE = 32

BIN_STEP_METRIC = 0.75  # Quantization step in meters
MIN_REGION_RATIO = 0.015  # 1.5% minimum area threshold for text labels

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_npy"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visuals"), exist_ok=True)


# -------------------------------------------------------------
# 1. Extract Unique Existing Images
# -------------------------------------------------------------
with open(ANNOTATIONS_FILE, "r") as f:
    raw_data = json.load(f)

unique_image_paths = []
seen_filenames = set()

for entry in raw_data:
    img_filename = entry.get("filename") or entry.get("url", "").split("/")[-1]
    if img_filename in seen_filenames:
        continue

    # Resolve image path based on dataset origin
    if "nyu" in img_filename.lower():
        img_path = os.path.join("data/images/nyu/", img_filename)
    else:
        img_path = os.path.join("data/images/flickr/", img_filename)

    if os.path.exists(img_path):
        unique_image_paths.append(img_path)
        seen_filenames.add(img_filename)

print(f"Found {len(unique_image_paths)} unique valid images to process.")


# -------------------------------------------------------------
# 2. Dataset for Batch Processing
# -------------------------------------------------------------
class ImagePathDataset(Dataset):

    def __init__(self, paths, size=(518, 518)):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Resize all images to identical dimensions so PyTorch can stack the batch
        img = Image.open(self.paths[idx]).convert("RGB")
        return img.resize(self.size, Image.Resampling.BILINEAR)


def annotate_and_save_depth(depth_map, save_path):
    """Draws depth map and overlays depth values on prominent regions."""
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    im = ax.imshow(depth_map, cmap="turbo")

    # Quantize depth to group similar regions
    quantized = np.round(depth_map / BIN_STEP_METRIC) * BIN_STEP_METRIC
    unique_bins = np.unique(quantized)
    min_pixels = depth_map.size * MIN_REGION_RATIO

    for val in unique_bins:
        mask = quantized == val
        labeled_mask, num_features = ndimage.label(mask)

        for region_id in range(1, num_features + 1):
            region = labeled_mask == region_id
            if np.sum(region) >= min_pixels:
                avg_val = depth_map[region].mean()
                cy, cx = ndimage.center_of_mass(region)

                ax.text(
                    cx,
                    cy,
                    f"{avg_val:.2f}m",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="black",
                        ec="none",
                        alpha=0.6,
                    ),
                )

    plt.colorbar(im, ax=ax, label="Depth (meters)", fraction=0.046, pad=0.04)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------
# 3. Inference & Depth Saving Loop
# -------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(task="depth-estimation", model=MODEL_NAME, device=device)

dataset = ImagePathDataset(unique_image_paths, size=(518, 518))

for idx, out in enumerate(
    tqdm(
        pipe(dataset, batch_size=BATCH_SIZE),
        total=len(dataset),
        desc="Generating Depth Maps",
    )
):
    source_path = unique_image_paths[idx]
    base_name = os.path.splitext(os.path.basename(source_path))[0]

    # Convert predicted depth to standard 2D float32 numpy array
    depth_map = np.asarray(out["predicted_depth"]).astype(np.float32)

    # 1. Save raw metric values (meters) for downstream VLM training/probing
    npy_save_path = os.path.join(OUTPUT_DIR, "raw_npy", f"{base_name}.npy")
    np.save(npy_save_path, depth_map)

    # 2. Save annotated visualization
    # visual_save_path = os.path.join(
        # OUTPUT_DIR, "visuals", f"{base_name}_annotated.png"
    # )
    # annotate_and_save_depth(depth_map, visual_save_path)

# Free GPU VRAM
del pipe
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nProcessing complete! All outputs saved to './{OUTPUT_DIR}/'")