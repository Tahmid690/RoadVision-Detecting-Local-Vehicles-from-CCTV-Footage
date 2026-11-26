# RoadVision — Notebook Scoreboard

One row per notebook, all configs + scores in one table. The actual table lives in
**[`SCOREBOARD.csv`](SCOREBOARD.csv)** (open in Excel/Sheets, or any editor with CSV table
rendering, for a readable view — Markdown tables don't render in plain text editors). This file
holds the narrative notes that don't fit in CSV cells. `experiments/LOG.md` remains the source of
truth for full narrative detail (why a run was done, what was learned, open threads).

**Update both `SCOREBOARD.csv` and this file's notes whenever a new notebook is built or a PENDING
score comes back** — keep them in sync with `LOG.md`.

## Notes

- **#2 and #4 are "change one inference variable" experiments** — #2 changed resolution+TTA
  together (regression), #4 changed iou alone (also worse). Both are negative results, kept for
  the record per this repo's "log every run, win or lose" rule (see `AGENTS.md` §9).
- **#3, #5, and #6's "Local val" column is fold-level mAP50, not an ensemble score.** For #3 and
  #5, the WBF-fused public score came in *higher* than any single fold's own number (ensembling
  helps). **#6 broke that pattern on public**: its local fold avg (0.5063) is higher than #5's
  (0.408), yet #6 scored *lower* on public LB (0.60328 vs 0.63809). **Treat local fold averages as
  diagnostic only, never as the basis for picking a config to submit.**
- **Public LB rank ≠ private LB rank — this is the single most important finding in this table.**
  #5 beats #6 on public (0.63809 vs 0.60328), but #6 beats #5 on private (0.64542 vs 0.6252) — and
  0.64542 is the **best private score in the entire project**. If final standing is decided by
  private score, picking "the public-best notebook" as your final submission would have been the
  wrong call here. Same pattern (private > public, and reordering rank) shows up for #2, #4, and
  #7 vs each other too — public score alone is not a safe proxy for this competition's final rank.
- **#7 (RT-DETR) is not a clean result.** Training was manually interrupted at epoch 49/70 (taking
  too long, not a bug), and the submission came from a separate, undocumented inference notebook —
  not `build_rtdetr.py`'s own inference cell. 0.47912/0.50688 conflates "RT-DETR may not suit this
  dataset" with "this model was only 70%-trained and inferred via an unverified path." A fresh
  full-length run through the documented pipeline would be needed for a real verdict.
- **#8 (`merged.ipynb`, "Weighted 5 & 6" on the tracking sheet) is actually EQUAL-weighted**, not
  performance-weighted — no `weights=` is passed to `weighted_boxes_fusion`. It also has **no
  `build_*.py` generator script** in this repo (same gap as `k-fold-fixed.ipynb`) — it was added as
  a standalone `.ipynb`. Scores between #5 and #6 on both public and private; the equal-weight
  merge of a stronger model (#6, private 0.64542) with a weaker one (#5, private 0.6252) landed
  *between* them (0.64323) rather than beating the stronger one — diluting toward the average
  rather than improving on the best component, the same risk flagged when the original
  cross-resolution ensemble notebook was shelved earlier in this project.
- Append new rows here for any future notebook, and keep this file in sync with `LOG.md`.
