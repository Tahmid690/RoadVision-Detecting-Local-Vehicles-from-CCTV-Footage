import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# RoadVision — Inference-only (reuse K-Fold weights → fixed submission)

No retraining. This loads the 5 `best.pt` files your committed K-Fold run already produced and writes a correct submission.

**Setup (one manual step):** right panel → **Add Input** → **Your Work / Notebook Output** → select the K-Fold notebook version that finished. The competition data should already be attached (it's the competition notebook).

**Settings:** Accelerator = **GPU**, Internet = **On** (only to `pip install` the packages; the weights load locally).

Fixes carried over: builds a row for **every** test image (sample_submission has only 10), `image_id` = **stem with no extension**, and **Weighted Box Fusion** to merge the 5 models.""")

code("""# Install + imports
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import YOLO
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "ensemble-boxes"], check=True)
    from ensemble_boxes import weighted_boxes_fusion

import re
from pathlib import Path
import numpy as np
import pandas as pd""")

code("""# --- Auto-discover test images + the fold weights (no hard-coded paths) ---
INPUT_ROOT = Path("/kaggle/input")

# Test images: an 'images' dir with 'test' in its path
TEST_IMAGES = next((p for p in INPUT_ROOT.rglob("images")
                    if p.is_dir() and "test" in str(p).lower()), None)

# Fold weights: every best.pt anywhere under /kaggle/input (your mounted notebook output)
weight_paths = [p for p in INPUT_ROOT.rglob("best.pt")]

def fold_key(p):
    m = re.search(r"fold[_-]?(\\d+)", str(p))
    return int(m.group(1)) if m else 999
weight_paths = sorted(weight_paths, key=fold_key)

print("TEST_IMAGES:", TEST_IMAGES)
print(f"found {len(weight_paths)} weight file(s):")
for p in weight_paths:
    print("  ", p)

assert TEST_IMAGES and TEST_IMAGES.exists(), "Test images not found - is the competition data attached?"
assert weight_paths, "No best.pt found - did you Add Input the finished K-Fold notebook output?"

IMGSZ = 960          # must match what the models were trained at
PRED_IOU = 0.60
WBF_IOU, SKIP_THR = 0.6, 0.0001""")

code("""# --- WBF ensemble inference -> submission.csv ---
test_idx = {p.stem: p for p in TEST_IMAGES.rglob("*.*")
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
test_stems = sorted(test_idx.keys())
print(f"test images to predict: {len(test_stems)}  (expect 327)")

models = [YOLO(str(p)) for p in weight_paths]
print(f"loaded {len(models)} models")

rows = []
for stem in test_stems:
    p = str(test_idx[stem])
    boxes_l, scores_l, labels_l = [], [], []
    for m in models:
        b = m.predict(source=p, imgsz=IMGSZ, conf=0.001, iou=PRED_IOU,
                      max_det=300, verbose=False)[0].boxes
        if b is not None and len(b) > 0:
            boxes_l.append(b.xyxyn.cpu().numpy().tolist())     # normalized [x1,y1,x2,y2]
            scores_l.append(b.conf.cpu().numpy().tolist())
            labels_l.append(b.cls.cpu().numpy().astype(int).tolist())

    if not boxes_l:
        rows.append({"image_id": stem, "PredictionString": ""})
        continue

    fb, fs, fl = weighted_boxes_fusion(boxes_l, scores_l, labels_l,
                                       iou_thr=WBF_IOU, skip_box_thr=SKIP_THR)
    parts = []
    for i in np.argsort(fs)[::-1]:                             # confidence-descending
        x1, y1, x2, y2 = fb[i]
        parts += [str(int(fl[i])), f"{fs[i] * 100:.2f}",
                  f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}",
                  f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"]
    rows.append({"image_id": stem, "PredictionString": " ".join(parts)})

sub = pd.DataFrame(rows)                # image_id = stem, NO extension
sub.to_csv("/kaggle/working/submission.csv", index=False)

n_with = int((sub["PredictionString"].str.len() > 0).sum())
print(f"\\nsaved /kaggle/working/submission.csv | rows: {len(sub)} (expect 327) | with detections: {n_with}")
print(sub.head())
assert len(sub) == len(test_stems), "Row count mismatch - submission incomplete!"  """)

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

out = str(Path(__file__).resolve().parents[1] / "dist" / "roadvision_infer_from_weights.ipynb")
json.dump(nb, open(out, "w"), indent=1)
json.load(open(out))
print("OK -", len(nb["cells"]), "cells ->", out)
