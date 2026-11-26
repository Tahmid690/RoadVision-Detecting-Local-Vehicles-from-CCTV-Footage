"""
Run this on any notebook in notebooks/dist/ before uploading to Kaggle.
Catches the bug classes that have actually bitten this project:
  - invalid JSON (a bad str_replace edit, etc.)
  - submission built from sample_submission.csv (the 10-row truncation bug)
  - image_id with a stray ".txt" / image extension in the FINAL submission write
  - missing row-count assert before saving a submission
  - hardcoded conf=0.01 sneaking in instead of a PRED_CONF variable

Usage: python scripts/validate_notebook.py notebooks/dist/your_notebook.ipynb
"""
import json
import re
import sys


def all_source(nb):
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )


def check(path):
    problems = []

    try:
        nb = json.load(open(path))
    except Exception as e:
        print(f"FAIL  invalid JSON: {e}")
        return False

    src = all_source(nb)
    writes_submission = "to_csv" in src and ("submission" in src.lower())

    if writes_submission:
        if "sample_df" in src or re.search(r"pd\.read_csv\(\s*SAMPLE_SUB", src):
            problems.append(
                "references sample_df / reads SAMPLE_SUB for row construction "
                "-> likely the 10-row truncation bug (sample_submission.csv is a FORMAT EXAMPLE, "
                "not the real test list)"
            )
        if "assert len(" not in src and "assert len(sub" not in src:
            problems.append(
                "no row-count assert found before/after saving the submission "
                "-> add `assert len(sub) == len(test_stems)` (or similar)"
            )
        if re.search(r'image_id["\']?\s*[:=]\s*.*\.txt', src):
            problems.append(
                'possible ".txt" left in a final image_id value -> submission image_id '
                "must have NO extension at all"
            )
        if "conf=0.01" in src and "PRED_CONF" not in src:
            problems.append(
                "hardcoded conf=0.01 with no PRED_CONF variable in scope -> probably should be "
                "the low-confidence value used elsewhere for mAP recall"
            )

    if problems:
        print(f"WARN  {path}")
        for p in problems:
            print(f"   - {p}")
        return False

    print(f"OK    {path}  ({len(nb['cells'])} cells)")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ok = True
    for p in sys.argv[1:]:
        ok = check(p) and ok
    sys.exit(0 if ok else 1)
