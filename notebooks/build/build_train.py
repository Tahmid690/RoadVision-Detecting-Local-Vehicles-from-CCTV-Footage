import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# RoadVision — Training + Inference (YOLO)

**From EDA:** 810 train / 327 test images, all 1280×720, 13 classes (0–12), 10,475 boxes, dual Tesla T4.

This notebook is end-to-end (one *Run All* → `submission.csv`):
1. Convert `train.csv` → Ultralytics YOLO layout (one `labels/<stem>.txt` per image) + `data.yaml`
2. Leak-resistant train/val split
3. Fine-tune a COCO-pretrained YOLO (transfer learning)
4. Inference on test → packed `PredictionString` → `submission.csv`

### The math, briefly
- **IoU** = area(overlap)/area(union). A prediction is a true positive iff IoU ≥ 0.5 with a same-class GT box.
- **mAP@0.5**: per class, sort predictions by confidence, build the precision–recall curve, AP = area under it; mAP = mean AP over the 13 classes. *Confidence only affects ranking → its absolute scale (0–1 vs 0–100) doesn't change the score.*
- **Transfer learning**: COCO-pretrained weights already encode edges→wheels→vehicle shapes; with only ~810 images we fine-tune rather than learn filters from scratch.
- **Mosaic aug**: stitches 4 images into one, multiplying object/context variety — critical for a small dataset.

> **Before *Save Version → Save & Run All*:** Settings → **Accelerator = GPU** and **Internet = On** (the pip install + COCO weights need it). `submission.csv` is written to `/kaggle/working/`, so it appears in the version's **Output** to submit.
>
> If the install / weight download fails, Internet is off — tell me and I'll send the offline setup (wheel + weights as a Kaggle dataset).""")

code("""# --- Install / verify Ultralytics ---
try:
    import ultralytics
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
    import ultralytics

ultralytics.checks()   # prints versions + detected GPU
from ultralytics import YOLO""")

code("""import os, re, glob, shutil
from pathlib import Path
import numpy as np
import pandas as pd""")

code("""# --- Auto-discover dataset paths (same robust logic as the EDA notebook) ---
INPUT_ROOT = Path("/kaggle/input")

TRAIN_CSV  = next(iter(INPUT_ROOT.rglob("train.csv")), None)
SAMPLE_SUB = next(iter(INPUT_ROOT.rglob("sample_submission.csv")), None)

image_dirs = [p for p in INPUT_ROOT.rglob("images") if p.is_dir()]
TRAIN_IMAGES = next((p for p in image_dirs if "train" in str(p).lower()), None)
TEST_IMAGES  = next((p for p in image_dirs if "test"  in str(p).lower()), None)

print("TRAIN_CSV   :", TRAIN_CSV)
print("TRAIN_IMAGES:", TRAIN_IMAGES)
print("TEST_IMAGES :", TEST_IMAGES)
print("SAMPLE_SUB  :", SAMPLE_SUB)
assert TRAIN_CSV and TRAIN_IMAGES and TEST_IMAGES, "Path discovery failed - check /kaggle/input"  """)

code("""# ============================ CONFIG ============================
# Column names (verified in EDA)
COL_IMAGE, COL_CLASS = "image_id", "class_id"
COL_XC, COL_YC, COL_W, COL_H = "x_center", "y_center", "width", "height"

NUM_CLASSES = 13
# Submission uses numeric class IDs, so names here are cosmetic (don't affect score).
# Replace with the real 13 names from the competition page if you want readable plots.
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# Split
SPLIT_MODE = "temporal_tail"   # "temporal_tail" (leak-safe) | "group_clip" | "random"
VAL_FRAC   = 0.12
SEED       = 42

# Training (start fast; bump IMGSZ->1280 / MODEL->yolo11m if time/quota allows)
MODEL  = "yolo11s.pt"
IMGSZ  = 960
EPOCHS = 60
BATCH  = 16
DEVICE = 0                     # single T4. Dual: [0,1] is ~2x but DDP can be flaky in notebooks.

# Inference (low conf threshold maximizes mAP recall on the PR curve)
PRED_CONF, PRED_IOU, MAX_DET = 0.001, 0.6, 300

# Output dirs
# DATASET in /kaggle/temp -> scratch, NOT saved into the committed Output (keeps it clean,
# and avoids saving ~713 symlinked images). runs + submission.csv stay in /kaggle/working
# so they appear in the version Output (submit the csv; reuse best.pt later if you want).
DATASET = Path("/kaggle/temp/roadvision_yolo")
PROJECT, RUN = "/kaggle/working/runs", "exp"
SUB_PATH = "/kaggle/working/submission.csv"
# ===============================================================
print("config set")""")

code("""# --- Index image files by stem (filename without extension) ---
def index_images(folder):
    idx = {}
    for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in folder.rglob(e):
            idx[p.stem] = p          # stem has NO extension; matches image_id minus ".txt"
    return idx

train_idx = index_images(TRAIN_IMAGES)
test_idx  = index_images(TEST_IMAGES)
print("indexed train images:", len(train_idx))
print("indexed test  images:", len(test_idx))

df = pd.read_csv(TRAIN_CSV)
print("annotation rows:", len(df), "| unique image_ids:", df[COL_IMAGE].nunique())

# Keep only annotated images whose image file actually exists
df["stem"] = df[COL_IMAGE].map(lambda s: Path(str(s)).stem)   # strips ".txt"
have = set(train_idx)
missing = sorted(set(df["stem"]) - have)
print("annotated stems with no image file:", len(missing), missing[:3])
df = df[df["stem"].isin(have)].reset_index(drop=True)""")

code("""# --- Build the train/val split ---
rng = np.random.default_rng(SEED)

def clip_and_frame(stem):
    m = re.match(r"(.+?)_(\\d+)$", stem)            # CCTV01^...120000 , 62
    return (m.group(1), int(m.group(2))) if m else (stem, -1)

stems = sorted(df["stem"].unique())
val_set = set()

if SPLIT_MODE == "temporal_tail":
    # Within each clip, hold out the LAST VAL_FRAC of frames (a contiguous tail).
    # -> train/val frames are temporally separated => almost no adjacent-frame leakage,
    #    and every camera is represented in both splits.
    by_clip = {}
    for s in stems:
        c, f = clip_and_frame(s)
        by_clip.setdefault(c, []).append((f, s))
    for c, items in by_clip.items():
        items.sort()                                # by frame number
        k = max(1, int(round(len(items) * VAL_FRAC)))
        val_set.update(s for _, s in items[-k:])
elif SPLIT_MODE == "group_clip":
    # Hold out whole clips (tests cross-camera generalization; higher variance with 4 clips).
    clips = sorted({clip_and_frame(s)[0] for s in stems})
    rng.shuffle(clips)
    k = max(1, int(round(len(clips) * VAL_FRAC)))
    val_clips = set(clips[:k])
    val_set = {s for s in stems if clip_and_frame(s)[0] in val_clips}
else:  # "random" (NOT recommended: adjacent frames leak across splits)
    perm = rng.permutation(stems)
    k = int(round(len(stems) * VAL_FRAC))
    val_set = set(perm[:k])

train_stems = [s for s in stems if s not in val_set]
val_stems   = [s for s in stems if s in val_set]
print(f"split={SPLIT_MODE}  train={len(train_stems)}  val={len(val_stems)}")

# Per-clip val coverage sanity check
from collections import Counter
print("val images per clip:", dict(Counter(clip_and_frame(s)[0] for s in val_stems)))""")

md("""## Why this split
Frames in each clip are ~1–2 apart, so neighbours are near-identical. A **random** split would put frame *N* in train and *N+1* in val → the model "sees" val during training and the local mAP is inflated. The **temporal tail** keeps train and val frames in separate time windows within every camera, giving a more honest estimate while still training on all 4 scenes.""")

code("""# --- Materialize the Ultralytics dataset (symlink images, write label .txt files) ---
if DATASET.exists():
    shutil.rmtree(DATASET)
for split in ("train", "val"):
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# Group annotations by image once (fast)
groups = {k: g for k, g in df.groupby("stem")}

def write_split(stem_list, split):
    n_lbl = 0
    for stem in stem_list:
        src = train_idx[stem]
        dst_img = DATASET / "images" / split / (stem + src.suffix)
        try:
            os.symlink(src.resolve(), dst_img)      # symlink = no copy, instant
        except (OSError, FileExistsError):
            if not dst_img.exists():
                shutil.copy(src, dst_img)           # fallback if symlinks disabled
        g = groups[stem]
        lines = [f"{int(r[COL_CLASS])} {r[COL_XC]:.6f} {r[COL_YC]:.6f} {r[COL_W]:.6f} {r[COL_H]:.6f}"
                 for _, r in g.iterrows()]
        (DATASET / "labels" / split / (stem + ".txt")).write_text("\\n".join(lines))
        n_lbl += len(lines)
    return n_lbl

n_tr = write_split(train_stems, "train")
n_va = write_split(val_stems, "val")
print(f"wrote labels -> train boxes: {n_tr}  val boxes: {n_va}")

# data.yaml
yaml_text = (
    f"path: {DATASET}\\n"
    f"train: images/train\\n"
    f"val: images/val\\n"
    f"nc: {NUM_CLASSES}\\n"
    f"names: {CLASS_NAMES}\\n"
)
(DATASET / "data.yaml").write_text(yaml_text)
print("\\ndata.yaml:\\n" + yaml_text)

# Verify one example label matches the CSV
ex = train_stems[0]
print("example label", ex + ".txt", "->")
print((DATASET / "labels" / "train" / (ex + ".txt")).read_text()[:200])""")

md("""## What Ultralytics does during training
YOLO11 is **anchor-free**: the head predicts, per grid cell, an objectness/class score and box offsets. The loss is **box (CIoU) + classification (BCE) + DFL** (distribution focal loss for sub-pixel box edges). It runs mosaic/affine/HSV augmentation on the fly and disables mosaic for the final epochs (`close_mosaic`) so the model fine-tunes on clean images before stopping.""")

code("""# --- Train ---
model = YOLO(MODEL)   # downloads COCO-pretrained weights (needs internet on)
results = model.train(
    data=str(DATASET / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    project=PROJECT,
    name=RUN,
    seed=SEED,
    patience=20,          # early stop if val mAP plateaus
    mosaic=1.0,
    close_mosaic=10,      # turn off mosaic for the last 10 epochs
    cos_lr=True,
    verbose=True,
)
print("\\nbest weights:", Path(results.save_dir) / "weights" / "best.pt")""")

code("""# --- Validate (local mAP estimate) ---
metrics = model.val()
print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
# Per-class AP@0.5 (spot weak classes)
try:
    for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  class {int(i):2d} ({CLASS_NAMES[int(i)]}): AP50={ap:.3f}")
except Exception as e:
    print("per-class AP unavailable:", e)""")

md("""## Inference → submission
For each test image, output every detection (low `conf` → better mAP recall) as
`class conf x_center y_center width height`, space-separated, grouped into one `PredictionString` per image.
`image_id` is the stem + `.txt` (matching the sample). Confidence is scaled to 0–100 to mirror the sample (the metric ranks by it, so the scale itself is harmless).""")

code("""# --- Predict on test set and build submission.csv ---
best_pt = Path(results.save_dir) / "weights" / "best.pt"
pred_model = YOLO(str(best_pt))

test_paths = sorted(set(str(p) for p in test_idx.values()))
print("test images to predict:", len(test_paths))

preds = pred_model.predict(
    source=test_paths, imgsz=IMGSZ, conf=PRED_CONF, iou=PRED_IOU,
    max_det=MAX_DET, stream=True, verbose=False, device=DEVICE,
)

rows, n_with = [], 0
for r in preds:
    stem = Path(r.path).stem
    b = r.boxes
    parts = []
    if b is not None and len(b) > 0:
        cls   = b.cls.cpu().numpy().astype(int)
        conf  = b.conf.cpu().numpy()
        xywhn = b.xywhn.cpu().numpy()            # normalized xc, yc, w, h
        for i in conf.argsort()[::-1]:           # confidence-descending
            xc, yc, w, h = xywhn[i]
            parts += [str(int(cls[i])), f"{conf[i] * 100:.2f}",
                      f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
        n_with += 1
    rows.append({"image_id": stem, "PredictionString": " ".join(parts)})

sub = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
sub.to_csv(SUB_PATH, index=False)
print(f"saved {SUB_PATH}  | rows: {len(sub)}  | images with >=1 detection: {n_with}")""")

code("""# --- Sanity-check the submission against the sample format ---
sub = pd.read_csv(SUB_PATH)
print("rows:", len(sub), "(expected", len(test_idx), ")")
assert len(sub) == len(test_idx), "Row count mismatch - submission is incomplete!"  
print("columns:", list(sub.columns))
toks = str(sub["PredictionString"].iloc[0]).split()
print("row 0:", len(toks), "tokens ->", len(toks) // 6, "detections; first 6:", toks[:6])
display(sub.head())

if SAMPLE_SUB:
    samp = pd.read_csv(SAMPLE_SUB)
    samp.columns = samp.columns.str.strip()
    print("\\nsample columns:", list(samp.columns), "| our columns match:", list(sub.columns) == list(samp.columns))

assert os.path.exists(SUB_PATH), "submission.csv missing!"
print("\\nOK -> submission at", SUB_PATH, "(", round(os.path.getsize(SUB_PATH) / 1e6, 2), "MB )")
print("After Save & Run All this file appears in the version Output -> submit it.")""")

md("""## If time/quota remains — ways to push mAP
- **Higher resolution**: `IMGSZ=1280` (these are 1280×720; helps the many small/distant vehicles most).
- **Bigger model**: `yolo11m.pt` (more capacity; slower).
- **TTA at inference**: `pred_model.predict(..., augment=True)`.
- **Longer training**: raise `EPOCHS`; `patience` will stop early if it plateaus.
- **Pseudo-labeling**: predict on the competition's unannotated images, keep high-confidence boxes, add to training (semi-supervised).
- **Class imbalance**: from the EDA bar chart, weak/rare classes hurt mAP most — inspect their per-class AP50 above.

**Submit:** *Save Version → Save & Run All*, then attach `submission.csv` from the version's Output (or submit it directly from the competition page).""")

# ---- build notebook json ----
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
    cell = {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True)}
    if ctype == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

out = str(Path(__file__).resolve().parents[1] / "dist" / "roadvision_02_train.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
with open(out) as f:
    json.load(f)
print("OK -", len(nb["cells"]), "cells ->", out)
