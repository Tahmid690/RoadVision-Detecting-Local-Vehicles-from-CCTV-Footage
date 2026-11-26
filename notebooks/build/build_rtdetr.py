import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# RoadVision — RT-DETR (diverse 2nd submission)

A deliberately *different* model from your YOLO k-fold, so your two final submissions aren't correlated:
**RT-DETR** is a transformer detector (global self-attention, NMS-free) — it makes different mistakes
than CNN/YOLO (e.g. handles overlapping vehicles via scene context rather than NMS).

**Honest caveat:** RT-DETR is data-hungry, and 810 images is small. It may *not* beat your 0.638 YOLO
ensemble — this is the high-variance "wildcard" slot. That's exactly why the default config keeps a
clean validation holdout: you want to *know* if it's viable before submitting it.

**Pipeline:** single full-fit model (no k-fold), so it's structurally different from your other notebook too.

**Checkpoint selection (the fix vs the naive full-fit):** training with `val = train` would pick `best.pt`
by memorized score (overfit checkpoint, no ensemble to wash it out). So:
- `FULL_DATA=False` (default): train on the holdout-excluded set, validate on a clean leak-free holdout
  → honest mAP + per-class AP, and an honestly-selected `best.pt`.
- `FULL_DATA=True`: train on all 810 but use `last.pt` (avoids the leaked-`best.pt` problem).

Settings: GPU on, Internet on. Run in a 2nd Kaggle session alongside the k-fold (Kaggle allows 2
concurrent GPU sessions — just confirm your weekly GPU hours cover both runs before the deadline).""")

code("""import os, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import ultralytics
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
    import ultralytics
from ultralytics import RTDETR
ultralytics.checks()""")

code("""# --- Path auto-discovery ---
INPUT_ROOT = Path("/kaggle/input")
TRAIN_CSV = next(iter(INPUT_ROOT.rglob("train.csv")), None)
image_dirs = [p for p in INPUT_ROOT.rglob("images") if p.is_dir()]
TRAIN_IMAGES = next((p for p in image_dirs if "train" in str(p).lower()), None)
TEST_IMAGES  = next((p for p in image_dirs if "test"  in str(p).lower()), None)
print("TRAIN_CSV   :", TRAIN_CSV)
print("TRAIN_IMAGES:", TRAIN_IMAGES)
print("TEST_IMAGES :", TEST_IMAGES)
assert TRAIN_CSV and TRAIN_IMAGES and TEST_IMAGES, "Path discovery failed"  """)

code("""# ============================ CONFIG ============================
COL_IMAGE, COL_CLASS = "image_id", "class_id"
COL_XC, COL_YC, COL_W, COL_H = "x_center", "y_center", "width", "height"
NUM_CLASSES = 13

MODEL = "rtdetr-l.pt"      # transformer detector ('l' = large, COCO-pretrained)
IMGSZ = 1280               # if CUDA OOM -> drop to 1024
BATCH = 4                  # RT-DETR is VRAM-heavy; if OOM -> 2
DEVICE = 0
EPOCHS = 70                # transformers converge slower than YOLO; 70-100 is reasonable for one model
MIXUP = 0.15               # transformers tolerate stronger aug

# Data strategy
FULL_DATA = False          # False = honest clean holdout (recommended for this unproven-on-small-data model)
                           # True  = train on all 810 (uses last.pt, flying blind)
HOLDOUT_FOLDS = 5          # clean holdout = fold 0 of a per-clip split (~164 imgs)

# Inference (RT-DETR is NMS-free; no TTA -> ultralytics augment=True is YOLO-specific, can error on RT-DETR)
PRED_CONF, PRED_IOU = 0.001, 0.60

BASE_TEMP = Path("/kaggle/temp/roadvision_rtdetr")
RUNS_DIR = Path("/kaggle/working/runs_rtdetr")
SUB_PATH = "/kaggle/working/submission_rtdetr.csv"
# ===============================================================
print(f"{MODEL} @ {IMGSZ}px, {EPOCHS} epochs, batch={BATCH}, FULL_DATA={FULL_DATA}")""")

code("""# --- Load annotations, index images, build the (clean) holdout ---
df = pd.read_csv(TRAIN_CSV)
df["stem"] = df[COL_IMAGE].str.replace(".txt", "", regex=False)
train_idx = {p.stem: p for p in TRAIN_IMAGES.rglob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
df = df[df["stem"].isin(train_idx)].reset_index(drop=True)
stems = sorted(df["stem"].unique())
groups = {k: g for k, g in df.groupby("stem")}

by_clip = {}
for s in stems:
    by_clip.setdefault(s.split("_")[0], []).append(s)
holdout = []
for c, cs in by_clip.items():
    cs.sort()
    holdout.extend(np.array_split(cs, HOLDOUT_FOLDS)[0])
holdout = set(holdout)

if FULL_DATA:
    train_stems, val_stems = stems, sorted(holdout)     # val leaked; we'll use last.pt, ignore val metrics
else:
    train_stems = [s for s in stems if s not in holdout] # clean: holdout excluded from training
    val_stems = sorted(holdout)

print(f"train: {len(train_stems)} | val: {len(val_stems)} | FULL_DATA={FULL_DATA}")

DATASET = BASE_TEMP / "ds"
if DATASET.exists():
    shutil.rmtree(DATASET)
for sp in ("train", "val"):
    (DATASET / "images" / sp).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / sp).mkdir(parents=True, exist_ok=True)

def populate(stem_list, sp):
    for stem in stem_list:
        src = train_idx[stem]
        dst = DATASET / "images" / sp / src.name
        try:
            os.symlink(src.resolve(), dst)
        except (OSError, FileExistsError):
            if not dst.exists():
                shutil.copy(src, dst)
        lines = [f"{int(r[COL_CLASS])} {r[COL_XC]:.6f} {r[COL_YC]:.6f} {r[COL_W]:.6f} {r[COL_H]:.6f}"
                 for _, r in groups[stem].iterrows()]
        (DATASET / "labels" / sp / f"{stem}.txt").write_text("\\n".join(lines))

populate(train_stems, "train")
populate(val_stems, "val")
yaml_text = (f"path: {DATASET}\\ntrain: images/train\\nval: images/val\\n"
             f"nc: {NUM_CLASSES}\\nnames: {[str(i) for i in range(NUM_CLASSES)]}\\n")
DATA_YAML = DATASET / "data.yaml"
DATA_YAML.write_text(yaml_text)
print("dataset compiled at", DATASET)""")

code("""# --- Train RT-DETR ---
t0 = time.time()
model = RTDETR(MODEL)
model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
            project=str(RUNS_DIR), name="rtdetr", seed=0,
            mosaic=1.0, close_mosaic=10, mixup=MIXUP, cos_lr=True, verbose=True)
print(f"\\ntraining done in {(time.time()-t0)/60:.1f} min")

# Clean checkpoint if we held out honestly; last.pt if we trained on everything (avoids leaked best.pt)
WEIGHTS = RUNS_DIR / "rtdetr" / "weights" / ("last.pt" if FULL_DATA else "best.pt")
print("using weights:", WEIGHTS)
assert WEIGHTS.exists(), "weights not found"  """)

code("""# --- Honest validation (only meaningful when FULL_DATA=False; holdout was excluded from training) ---
if not FULL_DATA:
    m = RTDETR(str(WEIGHTS))
    metrics = m.val(data=str(DATA_YAML), imgsz=IMGSZ, conf=PRED_CONF, iou=PRED_IOU, split="val", verbose=False)
    print(f"RT-DETR honest mAP@0.5 = {metrics.box.map50:.4f} | mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print("(compare to your YOLO 0.638 LB -- if this is far lower, RT-DETR isn't viable on this data)\\n")
    print(f"{'class_id':>8}  {'AP@0.5':>7}  note")
    try:
        ap_by_class = {int(i): float(a) for i, a in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    except Exception as e:
        ap_by_class = {}
        print("  per-class unavailable:", e)
    for c in range(NUM_CLASSES):
        ap = ap_by_class.get(c, float("nan"))
        note = "WEAK" if (c in ap_by_class and ap < 0.30) else ""
        print(f"{c:>8}  {ap:>7.3f}  {note}")
else:
    print("FULL_DATA=True -> no honest local validation (trained on everything). Flying blind.")""")

code("""# --- Inference (single model) -> submission ---
infer_model = RTDETR(str(WEIGHTS))
test_idx = {p.stem: p for p in TEST_IMAGES.rglob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
test_stems = sorted(test_idx.keys())
print(f"test images: {len(test_stems)}")

rows = []
for stem in test_stems:
    p = str(test_idx[stem])
    b = infer_model.predict(source=p, imgsz=IMGSZ, conf=PRED_CONF, iou=PRED_IOU,
                            max_det=300, verbose=False)[0].boxes
    parts = []
    if b is not None and len(b) > 0:
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        xywhn = b.xywhn.cpu().numpy()
        for i in conf.argsort()[::-1]:                      # confidence-descending
            xc, yc, w, h = xywhn[i]
            parts += [str(int(cls[i])), f"{conf[i]*100:.2f}",
                      f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
    rows.append({"image_id": stem, "PredictionString": " ".join(parts)})   # image_id = stem, NO extension

sub = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
sub.to_csv(SUB_PATH, index=False)
n_with = int((sub["PredictionString"].str.len() > 0).sum())
print(f"\\nSAVED {SUB_PATH} | rows: {len(sub)} (expect {len(test_stems)}) | with detections: {n_with}")
print(sub.head())
assert len(sub) == len(test_stems), "Row count mismatch - submission incomplete!"  """)

md("""## How to use this
- **Submit `submission_rtdetr.csv` as your 2nd (diverse) slot** — keep your YOLO 0.638 file as slot 1.
- **Decision rule from the honest mAP above (FULL_DATA=False):**
  - If RT-DETR's mAP@0.5 is close to / above your YOLO CV → it's viable; submitting it as the diverse slot is sound.
  - If it's far below → RT-DETR didn't have enough data; don't waste a slot on it, and consider instead
    folding its predictions into a YOLO+RT-DETR **WBF ensemble** (diversity can still help even a weaker model).
- **Bigger potential win (optional next):** an inference-only notebook that WBF-merges your 5 YOLO fold
  weights *with* this RT-DETR model. Cross-architecture WBF (CNN + transformer) often beats either alone,
  because their errors are uncorrelated. Tell me when both runs are done and I'll build it.""")

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

out = str(Path(__file__).resolve().parents[1] / "dist" / "roadvision_06_rtdetr.ipynb")
json.dump(nb, open(out, "w"), indent=1)
json.load(open(out))
print("OK -", len(nb["cells"]), "cells ->", out)
