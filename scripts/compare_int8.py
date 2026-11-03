"""Accuracy check after quantization: detections of the FP32 and INT8 models
on the same images, side by side.

    python scripts/compare_int8.py --images datasets/site11/images/test

Writes results/int8_check.md (table for the README).
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder of test images")
    p.add_argument("--fp32", default="models/yolov8n.onnx")
    p.add_argument("--int8", default="models/yolov8n_int8.onnx")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", default="results/int8_check.md")
    a = p.parse_args()

    fp32 = OnnxYoloDetector(a.fp32, conf_thres=a.conf)
    int8 = OnnxYoloDetector(a.int8, conf_thres=a.conf)
    rows = ["| Image | Object | FP32 score | INT8 score | Box IoU FP32 vs INT8 |",
            "|---|---|---:|---:|---:|"]
    matched = drops = 0
    total_fp32 = total_int8 = 0
    for f in sorted(Path(a.images).iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(f))
        d32, d8 = fp32(img), int8(img)
        total_fp32 += len(d32)
        total_int8 += len(d8)
        used = set()
        for d in sorted(d32, key=lambda x: -x["score"]):
            cands = [(iou(d["box"], e["box"]), j) for j, e in enumerate(d8)
                     if j not in used and e["cls_id"] == d["cls_id"]]
            best = max(cands, default=(0.0, None))
            if best[1] is not None and best[0] >= 0.5:
                used.add(best[1])
                e = d8[best[1]]
                matched += 1
                drops = max(drops, d["score"] - e["score"])
                rows.append(f"| {f.name} | {d['cls_name']} | {d['score']:.2f} | {e['score']:.2f} "
                            f"| {best[0]:.2f} |")
            else:
                rows.append(f"| {f.name} | {d['cls_name']} | {d['score']:.2f} | missed | – |")
        for j, e in enumerate(d8):
            if j not in used:
                rows.append(f"| {f.name} | {e['cls_name']} | not detected | {e['score']:.2f} | – |")
    summary = (f"\n{matched}/{total_fp32} FP32 detections found again by INT8 "
               f"(INT8 total: {total_int8}); largest confidence drop: {drops:.2f}.")
    text = "\n".join(rows) + "\n" + summary
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
