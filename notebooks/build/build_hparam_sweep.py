import json
from pathlib import Path

cells = []
def code(s): cells.append(("code", s))

# --- CELL 1 (CORRECTED): reconstruct the EXACT fold-0 holdout, per-clip ---
# Bug this replaced: original Cell 1 globally sorted all stems and took the last 12%,
# which grabbed one camera's tail end entirely (not the real per-clip fold split) ->
# 100% overlap with that fold's TRAINING set -> meaningless inflated local mAP (0.848).
cell1 = '''# ==============================================================================
# CELL 1 (CORRECTED): Reconstruct the EXACT fold-0 holdout, per-clip
# ==============================================================================
import os, shutil
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import YOLO

INPUT_ROOT = Path("/kaggle/input/competitions/road-vision/RoadVision_DUET")
TRAIN_CSV = next(iter(INPUT_ROOT.rglob("train.csv")), None)
TRAIN_IMAGES = INPUT_ROOT / "train" / "images"

# EDIT: point at whichever fold's weights you're evaluating
WEIGHTS_PATH = "/kaggle/input/notebooks/tahjibhossainkhan/k-fold/runs/fold_0/weights/best.pt"
K_FOLDS = 5   # MUST match the K_FOLDS used when these weights were trained

SCRATCH_DIR = Path("/kaggle/working/val_scratchpad")
VAL_IMG_DIR = SCRATCH_DIR / "images" / "val"
VAL_LBL_DIR = SCRATCH_DIR / "labels" / "val"
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LBL_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_CSV)
df.columns = df.columns.str.strip()
df["image_id"] = df["image_id"].str.replace(".txt", "", regex=False)

# Rebuild the SAME per-clip split logic as the training notebook (group by clip prefix,
# sort chronologically within each clip, np.array_split into K_FOLDS, take chunk[fold_idx]
# from every clip) -- NOT a global sort, which would grab only one camera.
stems = sorted(df["image_id"].unique())
by_clip = {}
for s in stems:
    clip_id = s.split("_")[0]
    by_clip.setdefault(clip_id, []).append(s)

folds_val = {i: [] for i in range(K_FOLDS)}
for clip, clip_stems in by_clip.items():
    clip_stems.sort()
    chunks = np.array_split(clip_stems, K_FOLDS)
    for i in range(K_FOLDS):
        folds_val[i].extend(chunks[i])

FOLD_IDX = 0   # EDIT to match WEIGHTS_PATH's fold
val_stems = set(folds_val[FOLD_IDX])
print(f"true fold-{FOLD_IDX} val set: {len(val_stems)} images across {len(by_clip)} clips")

grouped = df[df["image_id"].isin(val_stems)].groupby("image_id")
for img_id, group in grouped:
    lbl_path = VAL_LBL_DIR / f"{img_id}.txt"
    with open(lbl_path, "w") as f:
        for _, row in group.iterrows():
            f.write(f"{int(row['class_id'])} {row['x_center']:.6f} {row['y_center']:.6f} "
                     f"{row['width']:.6f} {row['height']:.6f}\\n")
    src_img = TRAIN_IMAGES / f"{img_id}.jpg"
    if not src_img.exists():
        src_img = TRAIN_IMAGES / f"{img_id}.png"
    if src_img.exists():
        shutil.copyfile(src_img, VAL_IMG_DIR / src_img.name)

# NOTE: these class names are UNVERIFIED against the competition's Data page (cosmetic only --
# model.val() scores by class_id, not name -- but don't trust these for per-class debugging
# until confirmed).
yaml_content = f"""
path: {SCRATCH_DIR.as_posix()}
train: images/val
val: images/val

names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle
  5: pedestrian
  6: van
  7: three_wheeler
  8: pickup
  9: e_rickshaw
  10: auto_rickshaw
  11: tractor
  12: traffic_sign
"""
YAML_PATH = SCRATCH_DIR / "local_val.yaml"
with open(YAML_PATH, "w") as f:
    f.write(yaml_content)

print(f"Validation environment prepared. Total TRUE holdout images: {len(val_stems)}")
assert os.path.exists(WEIGHTS_PATH), "Weights missing! Verify your attached notebook dataset input."
'''
cells.append(("code", cell1))

# --- CELL 2 & 3: unchanged from the user's original upload ---
cell2 = '''# ==============================================================================
# CELL 2: HIGH-FIDELITY LOCAL EVALUATION ENGINE
# ==============================================================================
def evaluate_local_configuration(weights, yaml_path, imgsz=960, conf=0.001, iou=0.60, augment=False):
    """
    Computes precise validation metrics on the holdout partition
    entirely inside Kaggle kernel memory.
    """
    model = YOLO(weights)
    metrics = model.val(
        data=yaml_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=augment,
        split="val",
        batch=16,
        save_json=False,
        plots=False,
        verbose=False,
    )
    return metrics.box.map50
'''
cells.append(("code", cell2))

cell3 = '''# ==============================================================================
# CELL 3: PARAMETER MATRIX SWEEP (GRID SEARCH)
# ==============================================================================
print("Calculating Baseline metric (No TTA, Imgsz=960, Conf=0.001, IoU=0.60)...")
baseline_score = evaluate_local_configuration(WEIGHTS_PATH, YAML_PATH, imgsz=960, conf=0.001, iou=0.60, augment=False)
print(f"======> Baseline Local mAP@0.5: {baseline_score:.5f}\\n")

print("Testing High-Res TTA configuration locally...")
tta_score = evaluate_local_configuration(WEIGHTS_PATH, YAML_PATH, imgsz=1280, conf=0.001, iou=0.60, augment=True)
print(f"======> High-Res TTA Local mAP@0.5: {tta_score:.5f}")
print(f"Delta: {tta_score - baseline_score:.5f}\\n")

conf_grid = [0.001, 0.01, 0.05]
iou_grid = [0.50, 0.60, 0.70]

best_score = -1.0
best_params = {}

print("Beginning parameter matrix sweep to maximize local mAP...")
for c in conf_grid:
    for i in iou_grid:
        score = evaluate_local_configuration(WEIGHTS_PATH, YAML_PATH, imgsz=960, conf=c, iou=i, augment=False)
        print(f"Tested -> Conf: {c:<5} | IoU: {i:<5} | Resulting Local mAP@0.5: {score:.5f}")
        if score > best_score:
            best_score = score
            best_params = {"conf": c, "iou": i, "imgsz": 960, "augment": False}

print("\\n" + "=" * 50)
print("OPTIMIZATION SWEEP COMPLETE")
print("=" * 50)
print(f"Top Local Performing Metric: {best_score:.5f}")
print(f"Optimal Hardware Parameters: {best_params}")
print("If this score beats your baseline, update your Inference notebook parameters and submit!")
'''
cells.append(("code", cell3))

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True, "accelerator": "nvidiaTeslaT4"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for t, s in cells:
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c["execution_count"] = None; c["outputs"] = []
    nb["cells"].append(c)

out = str(Path(__file__).resolve().parents[1] / "dist" / "hyperparameter_optimization_fixed.ipynb")
json.dump(nb, open(out, "w"), indent=1)
json.load(open(out))
print("OK -", len(nb["cells"]), "cells ->", out)
