"""Gather the figures and tables shown in the README into the repository
(runs/ and datasets/ are git-ignored, so the images must be copied out).

    python scripts/collect_assets.py --run site11hh_v2 --control site_final

Copies into assets/training/:
    results.png, confusion_matrix_normalized.png, BoxPR_curve.png, BoxF1_curve.png,
    val_batch1_pred.jpg      (Ultralytics plots of the released run)
    <run>_curves.png, <control>_curves.png   (scripts/training_report.py)
and into docs/training/ the per-epoch tables (<run>_epochs.md, ...) and the
evaluation tables written by eval_safety.py / external_test.py.
Missing files are listed, not fatal.
"""

import argparse
import shutil
from pathlib import Path

PLOTS = ["results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
         "val_batch1_pred.jpg", "val_batch0_pred.jpg", "labels.jpg"]


def find_run(name: str) -> Path | None:
    hits = sorted(Path("runs").rglob(f"{name}/results.csv"), key=lambda f: f.stat().st_mtime)
    return hits[-1].parent if hits else None


def copy(src: Path, dst: Path, missing: list):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src} -> {dst}")
    else:
        missing.append(str(src))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="site11hh_v2", help="released training run")
    p.add_argument("--control", nargs="*", default=["site_final"], help="other runs to document")
    a = p.parse_args()
    missing: list[str] = []
    assets, docs = Path("assets/training"), Path("docs/training")

    run_dir = find_run(a.run)
    if run_dir is None:
        raise SystemExit(f"no runs/**/{a.run}/results.csv")
    for f in PLOTS:
        copy(run_dir / f, assets / f, missing)
    for name in [a.run] + a.control:
        copy(Path("results") / f"{name}_curves.png", assets / f"{name}_curves.png", missing)
        copy(Path("results") / f"{name}_epochs.md", docs / f"{name}_epochs.md", missing)
        copy(Path("results") / f"{name}_metrics.json", docs / f"{name}_metrics.json", missing)
    for f in ("eval_test.md", "eval_val.md"):
        copy(Path("results") / f, docs / f, missing)
    for d in ("ext_before", "ext_v2", "ext_final"):
        copy(Path("results") / d / "external_eval.md", docs / f"{d}_external_eval.md", missing)
    if missing:
        print("\nnot found (run the script that makes them, or ignore):\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
