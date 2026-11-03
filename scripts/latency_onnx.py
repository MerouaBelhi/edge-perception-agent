"""CPU latency of one or more ONNX models on real images, with the hand-written
pipeline (pre-processing, inference, post-processing timed separately).
Used to compare the released model in FP32 and INT8 at its own input size.

    python scripts/latency_onnx.py models/site11hh_v2.onnx models/site11hh_v2_int8.onnx \\
        --images datasets/site11/images/test --runs 100

Each model runs in a fresh session on the same images, after 10 warm-up runs.
Writes results/latency_onnx.md.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--images", required=True)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--out", default="results/latency_onnx.md")
    a = p.parse_args()

    files = sorted(f for f in Path(a.images).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs = [im for im in (cv2.imread(str(f)) for f in files[:50]) if im is not None]
    if not imgs:
        raise SystemExit(f"no images in {a.images}")

    rows = ["| Model | Input | Size (MB) | Pre (ms) | Inference (ms) | Post (ms) | Total mean (ms) | p95 (ms) | FPS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in a.models:
        det = OnnxYoloDetector(m, conf_thres=0.3)
        for i in range(10):
            det(imgs[i % len(imgs)])
        t = {"pre_ms": [], "infer_ms": [], "post_ms": []}
        for i in range(a.runs):
            det(imgs[i % len(imgs)])
            for k in t:
                t[k].append(det.last_timing[k])
        tot = np.sum([t[k] for k in t], axis=0)
        rows.append(f"| `{Path(m).name}` | {det.input_size} | {Path(m).stat().st_size / 1e6:.1f} | "
                    f"{np.mean(t['pre_ms']):.1f} | {np.mean(t['infer_ms']):.1f} | {np.mean(t['post_ms']):.1f} | "
                    f"{tot.mean():.1f} | {np.percentile(tot, 95):.1f} | {1000 / tot.mean():.1f} |")
    text = "\n".join(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
