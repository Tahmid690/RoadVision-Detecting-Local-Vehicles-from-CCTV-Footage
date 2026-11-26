import json
from pathlib import Path

cells = []

def md(src):  cells.append(("markdown", src))
def code(src): cells.append(("code", src))

md("""# RoadVision — Initial EDA & Setup

**Task:** vehicle detection, 13 classes, metric = **mAP@0.5**.

This notebook does **understanding only** (no training yet):
1. Auto-discover dataset paths under `/kaggle/input` (robust to exact mount path)
2. Verify GPU
3. Inspect `train.csv` annotation schema
4. Class distribution + boxes-per-image
5. Visualize sample images with bounding boxes (sanity-check the coordinate convention)
6. Inspect submission format

> Schemas verified from the uploaded CSVs:
> - `train.csv` = one row per box: `image_id, class_id, x_center, y_center, width, height` (normalized `[0,1]`).
> - submission = **packed** `image_id, PredictionString`, where PredictionString = groups of 6 `class conf xc yc w h`.
> - `image_id` ends in `.txt` (it's the annotation filename) — handled when locating image files below.""")

code("""import os
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)""")

code("""# --- Auto-discover dataset layout (don't hard-code the exact mount path) ---
INPUT_ROOT = Path("/kaggle/input")

def rglob_list(pat):
    return list(INPUT_ROOT.rglob(pat))

# Locate key CSVs anywhere under /kaggle/input
train_csv_hits  = rglob_list("train.csv")
sample_sub_hits = rglob_list("sample_submission.csv")
gt_hits         = rglob_list("ground_truth_sample.csv")

print("train.csv               :", [str(p) for p in train_csv_hits])
print("sample_submission.csv   :", [str(p) for p in sample_sub_hits])
print("ground_truth_sample.csv :", [str(p) for p in gt_hits])

# First candidate for each (edit if multiple appear)
TRAIN_CSV  = train_csv_hits[0]  if train_csv_hits  else None
SAMPLE_SUB = sample_sub_hits[0] if sample_sub_hits else None

# Locate image directories (folders literally named 'images')
image_dirs = [p for p in INPUT_ROOT.rglob("images") if p.is_dir()]
TRAIN_IMAGES = next((p for p in image_dirs if "train" in str(p).lower()), None)
TEST_IMAGES  = next((p for p in image_dirs if "test"  in str(p).lower()), None)

print("\\nResolved paths:")
print("  TRAIN_CSV    :", TRAIN_CSV)
print("  SAMPLE_SUB   :", SAMPLE_SUB)
print("  TRAIN_IMAGES :", TRAIN_IMAGES)
print("  TEST_IMAGES  :", TEST_IMAGES)

def count_images(d):
    if not d:
        return 0
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    return sum(len(list(d.rglob(e))) for e in exts)  # rglob: also counts nested folders

print("\\n# train images:", count_images(TRAIN_IMAGES))
print("# test  images:", count_images(TEST_IMAGES))""")

code("""# --- Verify GPU is attached: Kaggle > Settings > Accelerator > GPU ---
import subprocess
out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
print(out or "nvidia-smi not found (GPU not attached?)")

try:
    import torch
    print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed in this image yet (we'll install ultralytics in the training notebook)")""")

code("""# --- Inspect annotation schema ---
train = pd.read_csv(TRAIN_CSV)
print("shape:", train.shape)
print("columns:", list(train.columns))
print("\\ndtypes:")
print(train.dtypes)
print("\\nfirst rows:")
display(train.head(10))
print("\\nnumeric ranges (reveals normalized [0,1] vs pixel coords):")
display(train.describe())""")

md("""## Reading the schema

YOLO-format annotations are usually **one row per bounding box**:
`image_id, class_id, x_center, y_center, width, height` — coordinates normalized to `[0, 1]`.

Check the `describe()` output above:
- max of x/y/w/h **≈ 1.0** → normalized (YOLO convention)
- max in the **hundreds/thousands** → pixel coordinates

Now edit the CONFIG cell below to match the **actual** column names you just saw.""")

code("""# Column mapping — verified against the uploaded train.csv (already correct).
COL_IMAGE = "image_id"     # holds the annotation filename (ends in ".txt")
COL_CLASS = "class_id"     # class label (ints 0..12)
COL_XC    = "x_center"     # normalized center x
COL_YC    = "y_center"     # normalized center y
COL_W     = "width"        # normalized width
COL_H     = "height"       # normalized height

COORDS_NORMALIZED = True   # verified: coords are normalized [0,1]
# NOTE: image_id values end in ".txt" (the annotation filename). The actual image
# shares the same stem with an image extension, so we strip the extension when
# locating image files (see resolve_image_path below).

# Fill from the competition Data page (13 classes). Leave empty to just show numeric IDs.
# Known class examples mentioned in the brief (verify exact IDs on the page):
#   Medium Truck, Heavy Truck, Bus, Motorcycle, CNG auto-rickshaw, Rickshaw, Easy-bike, Leguna, ...
CLASS_NAMES = {
    # 0: "Medium Truck",
    # 1: "Heavy Truck",
}

# Sanity check
missing = [c for c in [COL_IMAGE, COL_CLASS, COL_XC, COL_YC, COL_W, COL_H] if c not in train.columns]
print("MISSING COLUMNS:", missing if missing else "none")""")

code("""# --- Class distribution (look for imbalance) ---
counts = train[COL_CLASS].value_counts().sort_index()
print("unique class ids:", counts.index.tolist())
print("n classes:", counts.shape[0])
print("total boxes:", len(train))
print("\\nper-class counts:")
print(counts)

plt.figure(figsize=(10, 4))
labels = [CLASS_NAMES.get(c, str(c)) for c in counts.index]
plt.bar(labels, counts.values)
plt.xticks(rotation=60, ha="right")
plt.ylabel("# boxes")
plt.title("Class distribution")
plt.tight_layout()
plt.show()""")

code("""# --- How many objects per image? ---
per_img = train.groupby(COL_IMAGE).size()
print(per_img.describe())

plt.figure(figsize=(8, 4))
plt.hist(per_img.values, bins=range(0, int(per_img.max()) + 2))
plt.xlabel("boxes per image")
plt.ylabel("# images")
plt.title("Objects per image")
plt.tight_layout()
plt.show()

print("\\nImages with most objects:")
print(per_img.sort_values(ascending=False).head())""")

code("""# --- Sample image dimensions (needed to convert normalized <-> pixels) ---
sample_files = []
for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
    sample_files += [str(p) for p in TRAIN_IMAGES.rglob(e)]
sample_files = sample_files[:20]

dims = []
for f in sample_files:
    with Image.open(f) as im:
        dims.append(im.size)  # (w, h)

print("sample (w, h):", dims[:10])
if dims:
    ws, hs = zip(*dims)
    print(f"width  range: {min(ws)}-{max(ws)}")
    print(f"height range: {min(hs)}-{max(hs)}")""")

code("""# --- Visualize sample images with bounding boxes ---
# If boxes don't land on vehicles: COORDS_NORMALIZED or the box convention is wrong.
def resolve_image_path(image_id):
    # image_id ends in ".txt"; the real image shares the stem with an image extension.
    clean = Path(str(image_id)).stem            # strips the ".txt"
    hits = list(TRAIN_IMAGES.glob(clean + ".*"))
    if not hits:                                 # fallback if images sit in subfolders
        hits = list(TRAIN_IMAGES.rglob(clean + ".*"))
    return hits[0] if hits else None

n = min(6, train[COL_IMAGE].nunique())
sample_ids = train[COL_IMAGE].drop_duplicates().sample(n, random_state=0).tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, img_id in zip(axes.ravel(), sample_ids):
    path = resolve_image_path(img_id)
    if path is None:
        ax.set_title(f"NOT FOUND: {img_id}")
        ax.axis("off")
        continue
    img = Image.open(path).convert("RGB")
    W, H = img.size
    ax.imshow(img)
    rows = train[train[COL_IMAGE] == img_id]
    for _, r in rows.iterrows():
        xc, yc, w, h = r[COL_XC], r[COL_YC], r[COL_W], r[COL_H]
        if COORDS_NORMALIZED:
            xc, yc, w, h = xc * W, yc * H, w * W, h * H
        x0, y0 = xc - w / 2, yc - h / 2
        ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, linewidth=2, edgecolor="lime"))
        cls = r[COL_CLASS]
        ax.text(x0, max(y0 - 3, 0), CLASS_NAMES.get(cls, str(cls)),
                fontsize=8, color="black", backgroundcolor="lime")
    ax.set_title(f"{img_id}  ({len(rows)} boxes)")
    ax.axis("off")
plt.tight_layout()
plt.show()""")

code("""# --- Submission format ---
if SAMPLE_SUB:
    sub = pd.read_csv(SAMPLE_SUB)
    sub.columns = sub.columns.str.strip()
    print("shape:", sub.shape, "| columns:", list(sub.columns))
    display(sub.head())

    # PACKED format: one row per image. PredictionString = space-separated groups of 6:
    #   [class_id, confidence, x_center, y_center, width, height]
    s = str(sub["PredictionString"].iloc[0])
    toks = s.split()
    print("\\nrow 0: %d tokens -> %d detections (6 values each)" % (len(toks), len(toks) // 6))
    print("first detection [class, conf, xc, yc, w, h]:", toks[:6])
    print("\\nNOTE: confidence in the sample is on a 0-100 scale (e.g. 65.38), not 0-1.")
    print("      (mAP only uses confidence to RANK predictions, but we'll match the format anyway.)")
else:
    print("sample_submission.csv not found - check the path-discovery output above.")""")

md("""## Confirmed (from the uploaded CSVs)

- `train.csv`: one row per box, normalized YOLO coords. ✓
- Submission: packed `image_id, PredictionString` (groups of 6: `class conf xc yc w h`), confidence 0-100. ✓
- `image_id` ends in `.txt`; image files share the stem. ✓

**Run this to still verify visually:** boxes land on vehicles in the viz cell, and the printed train/test image counts are non-zero.

**Next notebook (training):**
1. Convert `train.csv` -> Ultralytics YOLO layout: one `labels/<stem>.txt` per image (lines = `class xc yc w h`) + `data.yaml` (13 class names) + 90/10 train/val split
2. Fine-tune a pretrained YOLO (transfer learning — only ~810 images)
3. Mosaic/geometric augmentation for the small dataset
4. Inference on test -> assemble `PredictionString` -> `submission.csv`""")

# Build notebook JSON
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for ctype, src in cells:
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": ctype, "metadata": {}, "source": lines}
    if ctype == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

out = str(Path(__file__).resolve().parents[1] / "dist" / "roadvision_01_eda.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

# Validate
with open(out) as f:
    reloaded = json.load(f)
print("OK -", len(reloaded["cells"]), "cells written to", out)
