"""Show the score of every vest / helmet / headwear box in a few images, to see
whether wrong decisions come from low-confidence detections (and so could be
removed by a score threshold) or from confident ones (a data problem).

    python scripts/ppe_scores.py path/to/img1.jpg path/to/img2.jpg --out results/ppe_scores

Writes the images with each box and its score, and prints the scores.
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402

COLORS = {"vest": (0, 200, 255), "no_vest": (0, 0, 255), "helmet": (0, 200, 0),
          "no_helmet": (0, 0, 200), "other_headwear": (255, 0, 255)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="models/site11hh_v2.onnx")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="results/ppe_scores")
    a = p.parse_args()

    det = OnnxYoloDetector(a.model, conf_thres=a.conf)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.images:
        img = cv2.imread(f)
        if img is None:
            print(f"cannot read {f}")
            continue
        scale = max(1.0, 1200 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=scale, fy=scale)
        print(f"\n{Path(f).name}")
        for d in sorted(det(img), key=lambda d: d["box"][0]):
            if d["cls_name"] not in COLORS:
                continue
            x1, y1, x2, y2 = map(int, d["box"])
            print(f"  {d['cls_name']:15s} score {d['score']:.2f}  x={x1}")
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[d["cls_name"]], 2)
            cv2.putText(img, f"{d['cls_name']} {d['score']:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[d["cls_name"]], 1, cv2.LINE_AA)
        cv2.imwrite(str(out / Path(f).name), img)
    print(f"\nimages -> {out}/")


if __name__ == "__main__":
    main()
