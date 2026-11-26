import json
from pathlib import Path

cells = []
def code(s): cells.append(("code", s))
def md(s):   cells.append(("markdown", s))

md("""# RoadVision — High-Res Retrain (1280px, yolo11m, 3-fold WBF+TTA)

Upgrades over the previous 0.567 run:
- `IMGSZ` 960 → **1280** (full resolution — most boxes are small/distant vehicles)
- `yolov8s` → **yolo11m** (more capacity)
- `K_FOLDS` 5 → **3** (keeps one commit inside Kaggle's session cap at the higher cost/epoch)
- `mixup=0.1` augmentation
- Inference: `conf=PRED_CONF` (fixes a hardcoded `conf=0.01` from before) + **TTA** (`augment=True`) + WBF
- A scan cell that checks for unannotated images on disk, settling that question with evidence

Settings: Accelerator = **GPU**, Internet = **On**.""")

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

code("""# --- Path auto-discovery (robust to exact mount path) ---
INPUT_ROOT = Path("/kaggle/input")
TRAIN_CSV = next(iter(INPUT_ROOT.rglob("train.csv")), None)

image_dirs = [p for p in INPUT_ROOT.rglob("images") if p.is_dir()]
TRAIN_IMAGES = next((p for p in image_dirs if "train" in str(p).lower()), None)
TEST_IMAGES  = next((p for p in image_dirs if "test"  in str(p).lower()), None)

print("TRAIN_CSV   :", TRAIN_CSV)
print("TRAIN_IMAGES:", TRAIN_IMAGES)
print("TEST_IMAGES :", TEST_IMAGES)
assert TRAIN_CSV and TRAIN_IMAGES and TEST_IMAGES, "Path discovery failed - check /kaggle/input"  """)

code("""# --- Settle the unannotated-images question with evidence, not guesswork ---
df_check = pd.read_csv(TRAIN_CSV)
annotated_stems = set(df_check["image_id"].str.replace(".txt", "", regex=False))

all_train_imgs = {p.stem for p in TRAIN_IMAGES.rglob("*.*")
                   if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
unannotated_in_train_dir = all_train_imgs - annotated_stems

# Also check for any sibling dirs that look like extra/unlabeled pools
candidate_extra = [p for p in INPUT_ROOT.rglob("*")
                    if p.is_dir() and p.name.lower() not in ("train", "test", "images")
                    and any(k in p.name.lower() for k in ["unlabel", "unannot", "extra", "pool", "pseudo"])]

print(f"images in TRAIN_IMAGES folder      : {len(all_train_imgs)}")
print(f"annotated (appear in train.csv)    : {len(annotated_stems & all_train_imgs)}")
print(f"present but NOT in train.csv       : {len(unannotated_in_train_dir)}")
print(f"other suspicious folders found     : {[str(p) for p in candidate_extra] or 'none'}")
print()
if unannotated_in_train_dir or candidate_extra:
    print(">> Unannotated images DO appear to exist - pseudo-labeling is viable as a follow-up notebook.")
else:
    print(">> No unannotated images found. Skipping pseudo-labeling; this notebook is the full plan.")""")

code("""# ============================ CONFIG ============================
COL_IMAGE, COL_CLASS = "image_id", "class_id"
COL_XC, COL_YC, COL_W, COL_H = "x_center", "y_center", "width", "height"
NUM_CLASSES = 13

K_FOLDS = 3                 # down from 5: 1280px + yolo11m costs ~3-4x compute/epoch vs the 960/yolov8s run
EPOCHS_PER_FOLD = 50
MODEL_ARCH = "yolo11m.pt"
IMGSZ = 1280
BATCH = 8                   # reduced from 16: 1280px needs more VRAM per image on a single T4
DEVICE = 0
SEED = 42

MIXUP = 0.1                 # safe for detection (whole-image blend). copy_paste needs seg masks - skipped.

PRED_CONF, PRED_IOU = 0.001, 0.60   # was hardcoded 0.01 before - this restores full recall for mAP
WBF_IOU, SKIP_THR = 0.6, 0.0001
USE_TTA = True               # augment=True at inference - free accuracy, no retrain cost

BASE_TEMP_DIR = Path("/kaggle/temp/roadvision_kfold_hires")
RUNS_DIR = Path("/kaggle/working/runs_hires")
SUB_PATH = "/kaggle/working/submission_hires.csv"
# ===============================================================
print(f"config set: {K_FOLDS} folds x {EPOCHS_PER_FOLD} epochs @ {IMGSZ}px, {MODEL_ARCH}, batch={BATCH}")
print("NOTE: if fold 0 takes longer than ~90 min, consider lowering K_FOLDS/EPOCHS_PER_FOLD for the")
print("      remaining folds (Kaggle GPU sessions are capped, historically 9-12h).")""")

code("""# --- Temporal-chunk K-fold split (same leak-aware logic as before) ---
df = pd.read_csv(TRAIN_CSV)
df["stem"] = df[COL_IMAGE].str.replace(".txt", "", regex=False)

train_idx = {p.stem: p for p in TRAIN_IMAGES.rglob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
df = df[df["stem"].isin(train_idx)].reset_index(drop=True)
stems = sorted(df["stem"].unique())

by_clip = {}
for s in stems:
    clip_id = s.split("_")[0]
    by_clip.setdefault(clip_id, []).append(s)

folds_val_stems = {i: [] for i in range(K_FOLDS)}
for clip, clip_stems in by_clip.items():
    clip_stems.sort()                                  # chronological (zero-padded frame numbers)
    chunks = np.array_split(clip_stems, K_FOLDS)
    for i in range(K_FOLDS):
        folds_val_stems[i].extend(chunks[i])

print(f"total unique images: {len(stems)}")
for i in range(K_FOLDS):
    print(f"  fold {i} val images: {len(folds_val_stems[i])}")

groups = {k: g for k, g in df.groupby("stem")}

def compile_fold(fold_idx):
    fold_dir = BASE_TEMP_DIR / f"fold_{fold_idx}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    for split in ("train", "val"):
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    val_set = set(folds_val_stems[fold_idx])
    for stem in stems:
        split = "val" if stem in val_set else "train"
        src = train_idx[stem]
        dst_img = fold_dir / "images" / split / src.name
        try:
            os.symlink(src.resolve(), dst_img)
        except (OSError, FileExistsError):
            if not dst_img.exists():
                shutil.copy(src, dst_img)
        g = groups[stem]
        lines = [f"{int(r[COL_CLASS])} {r[COL_XC]:.6f} {r[COL_YC]:.6f} {r[COL_W]:.6f} {r[COL_H]:.6f}"
                 for _, r in g.iterrows()]
        (fold_dir / "labels" / split / f"{stem}.txt").write_text("\\n".join(lines))

    yaml_text = (f"path: {fold_dir}\\ntrain: images/train\\nval: images/val\\n"
                 f"nc: {NUM_CLASSES}\\nnames: {[str(i) for i in range(NUM_CLASSES)]}\\n")
    (fold_dir / "data.yaml").write_text(yaml_text)
    return str(fold_dir / "data.yaml")

fold_yamls = [compile_fold(i) for i in range(K_FOLDS)]
print("all fold directories compiled.")""")

code("""# --- Sequential training loop (with per-fold timing for session-budget visibility) ---
print("Starting K-Fold training...")
fold_times = []
for fold in range(K_FOLDS):
    print(f"\\n{'='*40}\\nFOLD {fold}\\n{'='*40}")
    t0 = time.time()
    model = YOLO(MODEL_ARCH)
    model.train(
        data=fold_yamls[fold],
        epochs=EPOCHS_PER_FOLD,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=f"fold_{fold}",
        seed=SEED,
        patience=15,
        mosaic=1.0,
        close_mosaic=10,
        mixup=MIXUP,
        cos_lr=True,
        verbose=False,
    )
    dt = time.time() - t0
    fold_times.append(dt)
    print(f"fold {fold} done in {dt/60:.1f} min  | weights: {RUNS_DIR}/fold_{fold}/weights/best.pt")
    if fold == 0 and dt > 90 * 60:
        print("  ^^ WARNING: fold 0 took >90min - remaining folds may push close to the session limit.")

print(f"\\nall folds done. total: {sum(fold_times)/60:.1f} min")""")

code("""# --- Inference: WBF ensemble + TTA -> submission ---
test_idx = {p.stem: p for p in TEST_IMAGES.rglob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
test_stems = sorted(test_idx.keys())
print(f"test images: {len(test_stems)}")

models = [YOLO(f"{RUNS_DIR}/fold_{i}/weights/best.pt") for i in range(K_FOLDS)]
print(f"loaded {len(models)} fold models")

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

sub = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
sub.to_csv(SUB_PATH, index=False)

n_with = int((sub["PredictionString"].str.len() > 0).sum())
print(f"\\nsaved {SUB_PATH} | rows: {len(sub)} (expect {len(test_stems)}) | with detections: {n_with}")
print(sub.head())
assert len(sub) == len(test_stems), "Row count mismatch!"  """)

md("""## After this run

- Submit `submission_hires.csv`.
- If it beats 0.567: the resolution/model bump worked — consider this the new baseline.
- If you want to push further afterward: combine **these 3 new fold weights + the previous 5 fold weights** (8 models total, different res/arch) in one WBF ensemble — architectural diversity often helps WBF more than same-arch folds. I can build that inference-only notebook next if useful.""")

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

out = str(Path(__file__).resolve().parents[1] / "dist" / "roadvision_03_hires_kfold.ipynb")
json.dump(nb, open(out, "w"), indent=1)
json.load(open(out))
print("OK -", len(nb["cells"]), "cells ->", out)
