import json
from pathlib import Path

cells = []
def md(s):   cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md("""# RoadVision — 5-Fold yolo11m @1280 WBF Ensemble

Scales up the proven recipe: **1280 / yolo11m, 3-fold WBF scored 0.638** (the best so far).
Going 3 → 5 folds means each model trains on **80%** of the data (vs 67%) and the ensemble has
**5 members** (vs 3) → more data per model + more averaging → should edge past 0.638 with less variance.

**Note on k-fold (important):** each fold's model validates on the one fold it was *excluded* from,
so validation is clean by construction — `best.pt` is honestly selected (no leak), and you get a free
**per-class AP** breakdown. Keeping TTA on (part of the 0.638 recipe; it only hurt earlier on the
960-trained model due to train/inference resolution mismatch).

**Honest framing:** the per-fold mAP will read LOW (~0.40–0.55) because the per-clip temporal holdout
is *structurally harder* than Kaggle's test set — that's why your LB (0.638) is higher than local CV.
Use the fold scores for *relative* comparison + the per-class table, not as an absolute predictor.

Settings: GPU on, Internet on.""")

code("""import os, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import YOLO
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "ensemble-boxes"], check=True)
    from ensemble_boxes import weighted_boxes_fusion""")

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

# Proven-best family (0.638 = 1280/yolo11m, 3-fold). Scaling folds 3 -> 5.
MODEL = "yolo11m.pt"
IMGSZ = 1280
FOLDS = 5
BATCH = 8                  # 1280px + yolo11m. If CUDA OOM -> set 4.
DEVICE = 0

# --- BUDGET SIZING (read before a headless Save & Run All) ---
# Save & Run All can't be adjusted mid-run, so EPOCHS must fit up front.
# Your 0.638 run was 3 folds x 50 epochs = 150 "fold-epochs" and fit your budget.
# 5 folds x 30 epochs = 150 fold-epochs  -> ~same total compute as that run.
# If your 0.638 run was already near the session cap, drop to EPOCHS=25 or FOLDS=4.
EPOCHS = 30
# patience >= EPOCHS disables early stopping -> deterministic, predictable wall-time per fold.
PATIENCE = EPOCHS + 5
MIXUP = 0.1

# Inference (exact 0.638 recipe)
USE_TTA = True
PRED_CONF, PRED_IOU = 0.001, 0.60
WBF_IOU, SKIP_THR = 0.60, 0.0001

BASE_TEMP = Path("/kaggle/temp/roadvision_kfold5")
RUNS_DIR = Path("/kaggle/working/runs_kfold5")
SUB_PATH = "/kaggle/working/submission_kfold5.csv"
# ===============================================================
print(f"{FOLDS}-fold {MODEL} @ {IMGSZ}px, {EPOCHS} epochs/fold, batch={BATCH}, TTA={USE_TTA}")
print(f"total fold-epochs: {FOLDS*EPOCHS} (your 0.638 run was 150)")""")

code("""# --- Load annotations, index images, build per-clip K-fold val sets ---
df = pd.read_csv(TRAIN_CSV)
df["stem"] = df[COL_IMAGE].str.replace(".txt", "", regex=False)
train_idx = {p.stem: p for p in TRAIN_IMAGES.rglob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
df = df[df["stem"].isin(train_idx)].reset_index(drop=True)
stems = sorted(df["stem"].unique())
groups = {k: g for k, g in df.groupby("stem")}

by_clip = {}
for s in stems:
    by_clip.setdefault(s.split("_")[0], []).append(s)

folds_val = {i: [] for i in range(FOLDS)}
for clip, cs in by_clip.items():
    cs.sort()                                   # chronological (zero-padded frame numbers)
    for i, chunk in enumerate(np.array_split(cs, FOLDS)):
        folds_val[i].extend(chunk)

print(f"total images: {len(stems)} across {len(by_clip)} clips")
for i in range(FOLDS):
    print(f"  fold {i} val: {len(folds_val[i])}")

def build_fold_dir(fold_idx):
    \"\"\"train = all folds EXCEPT fold_idx ; val = fold_idx (clean -> honest best.pt).\"\"\"
    root = BASE_TEMP / f"fold_{fold_idx}"
    if root.exists():
        shutil.rmtree(root)
    for sp in ("train", "val"):
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
    val_set = set(folds_val[fold_idx])
    for stem in stems:
        sp = "val" if stem in val_set else "train"
        src = train_idx[stem]
        dst = root / "images" / sp / src.name
        try:
            os.symlink(src.resolve(), dst)
        except (OSError, FileExistsError):
            if not dst.exists():
                shutil.copy(src, dst)
        lines = [f"{int(r[COL_CLASS])} {r[COL_XC]:.6f} {r[COL_YC]:.6f} {r[COL_W]:.6f} {r[COL_H]:.6f}"
                 for _, r in groups[stem].iterrows()]
        (root / "labels" / sp / f"{stem}.txt").write_text("\\n".join(lines))
    yaml_text = (f"path: {root}\\ntrain: images/train\\nval: images/val\\n"
                 f"nc: {NUM_CLASSES}\\nnames: {[str(i) for i in range(NUM_CLASSES)]}\\n")
    (root / "data.yaml").write_text(yaml_text)
    return str(root / "data.yaml")""")

code("""# --- Train the K folds (clean val per fold) + collect CV metrics ---
fold_weights, fold_map50, per_class = [], [], {c: [] for c in range(NUM_CLASSES)}
t_start = time.time()

for fold in range(FOLDS):
    print(f"\\n{'='*46}\\nFOLD {fold}/{FOLDS-1}\\n{'='*46}")
    yaml_path = build_fold_dir(fold)
    t0 = time.time()
    model = YOLO(MODEL)
    model.train(data=yaml_path, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                project=str(RUNS_DIR), name=f"fold_{fold}", seed=fold, patience=PATIENCE,
                mosaic=1.0, close_mosaic=10, mixup=MIXUP, cos_lr=True, verbose=False)
    w = RUNS_DIR / f"fold_{fold}" / "weights" / "best.pt"
    fold_weights.append(w)

    metrics = model.val(data=yaml_path, imgsz=IMGSZ, conf=PRED_CONF, iou=PRED_IOU, split="val", verbose=False)
    fold_map50.append(float(metrics.box.map50))
    try:
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            per_class[int(idx)].append(float(ap))
    except Exception as e:
        print("  per-class unavailable:", e)

    dt = time.time() - t0
    print(f"fold {fold}: mAP@0.5={metrics.box.map50:.4f} | {dt/60:.1f} min")
    if fold == 0:
        est = dt * FOLDS
        print(f"\\n>>> BUDGET: ~{est/3600:.1f}h projected for all {FOLDS} folds (+ ~20-40 min TTA inference).")
        if est/3600 > 8:
            print(">>> If running interactively and this is too long: interrupt, lower EPOCHS or FOLDS, rerun.")

print(f"\\nAll folds done in {(time.time()-t_start)/60:.1f} min")
print(f"CV mAP@0.5 per fold: {[round(x,4) for x in fold_map50]}")
print(f"CV mean: {np.mean(fold_map50):.4f} +/- {np.std(fold_map50):.4f}  (reads LOW vs LB - hard split, expected)")""")

code("""# --- Per-class AP@0.5 across folds (the weak-class blind spot) ---
print(f"{'class_id':>8}  {'mean AP@0.5':>11}  {'folds':>5}  note")
weak = []
for c in range(NUM_CLASSES):
    vals = per_class[c]
    ap = float(np.mean(vals)) if vals else float("nan")
    note = "WEAK" if (vals and ap < 0.30) else ""
    if note:
        weak.append(c)
    print(f"{c:>8}  {ap:>11.3f}  {len(vals):>5}  {note}")
print(f"\\nWeak classes (mean AP<0.30): {weak if weak else 'none'}")
print("Highest-leverage next step: check if these are rare-in-train (oversample) vs genuinely hard.")
print("Confirm real class NAMES on the competition Data page before deep per-class work.")""")

code("""# --- Inference: WBF over the 5 folds (+ TTA) -> submission ---
test_idx = {p.stem: p for p in TEST_IMAGES.rglob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
test_stems = sorted(test_idx.keys())
print(f"test images: {len(test_stems)}")

models = [YOLO(str(w)) for w in fold_weights]
print(f"loaded {len(models)} fold models | TTA={USE_TTA}")

rows = []
for stem in test_stems:
    p = str(test_idx[stem])
    boxes_l, scores_l, labels_l = [], [], []
    for m in models:
        b = m.predict(source=p, imgsz=IMGSZ, conf=PRED_CONF, iou=PRED_IOU,
                      max_det=300, augment=USE_TTA, verbose=False)[0].boxes
        if b is not None and len(b) > 0:
            boxes_l.append(b.xyxyn.cpu().numpy().tolist())
            scores_l.append(b.conf.cpu().numpy().tolist())
            labels_l.append(b.cls.cpu().numpy().astype(int).tolist())
    if not boxes_l:
        rows.append({"image_id": stem, "PredictionString": ""})
        continue
    fb, fs, fl = weighted_boxes_fusion(boxes_l, scores_l, labels_l,
                                       iou_thr=WBF_IOU, skip_box_thr=SKIP_THR)
    parts = []
    for i in np.argsort(fs)[::-1]:
        x1, y1, x2, y2 = fb[i]
        parts += [str(int(fl[i])), f"{fs[i]*100:.2f}",
                  f"{(x1+x2)/2:.6f}", f"{(y1+y2)/2:.6f}", f"{x2-x1:.6f}", f"{y2-y1:.6f}"]
    rows.append({"image_id": stem, "PredictionString": " ".join(parts)})

sub = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)   # image_id = stem, NO extension
sub.to_csv(SUB_PATH, index=False)
n_with = int((sub["PredictionString"].str.len() > 0).sum())
print(f"\\nSAVED {SUB_PATH} | rows: {len(sub)} (expect {len(test_stems)}) | with detections: {n_with}")
print(sub.head())
assert len(sub) == len(test_stems), "Row count mismatch - submission incomplete!"  """)

md("""## After this run
- Submit `submission_kfold5.csv`. Compare to 0.638 — 5-fold should match or beat it.
- The **per-class table** is your next lever: oversample weak classes that are merely rare in training.
- If you want even more diversity without much extra time, an inference-only notebook can WBF these 5
  weights together with your earlier 3-fold weights (8 models, all 1280/yolo11m).""")

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

out = str(Path(__file__).resolve().parents[1] / "dist" / "roadvision_05_kfold5_yolo11m.ipynb")
json.dump(nb, open(out, "w"), indent=1)
json.load(open(out))
print("OK -", len(nb["cells"]), "cells ->", out)
