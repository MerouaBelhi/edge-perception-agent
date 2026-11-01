"""Per-epoch training report from Ultralytics' results.csv: a markdown table,
loss / metric curves (PNG) and, optionally, TensorBoard event files.

    python scripts/training_report.py --run site11_960
    python scripts/training_report.py --run site11_960 --tensorboard
    tensorboard --logdir runs/tensorboard --port 6006

--run is a training name (searched under runs/) or a folder / results.csv path.
Outputs: results/<run>_epochs.md, results/<run>_curves.png, runs/tensorboard/<run>/
Several runs can be given to compare them in TensorBoard (one curve per run).
"""

import argparse
import csv
from pathlib import Path

COLS = [("epoch", "Epoch"),
        ("train/box_loss", "train box"), ("train/cls_loss", "train cls"), ("train/dfl_loss", "train dfl"),
        ("val/box_loss", "val box"), ("val/cls_loss", "val cls"), ("val/dfl_loss", "val dfl"),
        ("metrics/precision(B)", "P"), ("metrics/recall(B)", "R"),
        ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]


def find_csv(run: str) -> Path:
    p = Path(run)
    if p.is_file():
        return p
    if (p / "results.csv").is_file():
        return p / "results.csv"
    hits = sorted(Path("runs").rglob(f"{run}/results.csv"), key=lambda f: f.stat().st_mtime)
    if not hits:
        raise SystemExit(f"no results.csv for '{run}' under runs/")
    return hits[-1]


def read(path: Path) -> list[dict]:
    with open(path) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return [{k: float(v) for k, v in r.items() if v not in ("", None)} for r in rows]


def table(rows: list[dict], name: str) -> str:
    cols = [(k, h) for k, h in COLS if k in rows[0]]
    best = max(range(len(rows)), key=lambda i: rows[i].get("metrics/mAP50-95(B)", 0))
    out = [f"**{name}**: {len(rows)} epochs, best epoch {int(rows[best]['epoch'])} "
           f"(mAP50 {rows[best].get('metrics/mAP50(B)', 0):.3f}, "
           f"mAP50-95 {rows[best].get('metrics/mAP50-95(B)', 0):.3f}, validation split)", "",
           "| " + " | ".join(h for _, h in cols) + " |",
           "|" + "---:|" * len(cols)]
    for i, r in enumerate(rows):
        cells = [str(int(r[k])) if k == "epoch" else f"{r[k]:.3f}" for k, _ in cols]
        if i == best:
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def plot(rows: list[dict], name: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, loss in zip(axes[:3], ("box_loss", "cls_loss", "dfl_loss")):
        for split, style in (("train", "-"), ("val", "--")):
            k = f"{split}/{loss}"
            if k in rows[0]:
                ax.plot(ep, [r[k] for r in rows], style, label=split)
        ax.set_title(loss)
        ax.set_xlabel("epoch")
        ax.legend()
    for k, lab in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
                   ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")):
        if k in rows[0]:
            axes[3].plot(ep, [r[k] for r in rows], label=lab)
    axes[3].set_title("validation metrics")
    axes[3].set_xlabel("epoch")
    axes[3].legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def tensorboard(rows: list[dict], name: str, logdir: Path):
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(str(logdir / name))
    for r in rows:
        step = int(r["epoch"])
        for k, v in r.items():
            if k != "epoch":
                w.add_scalar(k.replace("(B)", ""), v, step)
    w.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True, help="training name(s), folder(s) or results.csv")
    p.add_argument("--tensorboard", action="store_true", help="also write TensorBoard event files")
    p.add_argument("--logdir", default="runs/tensorboard")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for run in a.run:
        csv_path = find_csv(run)
        name = csv_path.parent.name
        rows = read(csv_path)
        md = table(rows, name)
        (Path(a.out) / f"{name}_epochs.md").write_text(md + "\n")
        plot(rows, name, Path(a.out) / f"{name}_curves.png")
        print(md, f"\n-> {a.out}/{name}_epochs.md, {a.out}/{name}_curves.png")
        if a.tensorboard:
            tensorboard(rows, name, Path(a.logdir))
            print(f"-> TensorBoard: {a.logdir}/{name}")


if __name__ == "__main__":
    main()
