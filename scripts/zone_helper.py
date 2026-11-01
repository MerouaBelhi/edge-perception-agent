"""Help choose zone coordinates for a new video.

    python scripts/zone_helper.py --source assets/site.mp4 --t 2.0
    python scripts/zone_helper.py --source assets/site.mp4 --zones configs/site_zones.json

Saves one frame with a labelled 10 % grid (results/zone_helper.jpg). Read the
corners of your zone on the grid, e.g. (0.10, 0.55), and write them in the zones
JSON with "normalized": true. With --zones, the zones are drawn on the frame too,
so you can check them before running the pipeline.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.zones import load_zones  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--t", type=float, default=0.0, help="time of the frame, in seconds")
    p.add_argument("--zones", default=None, help="zones JSON to draw on top")
    p.add_argument("--out", default="results/zone_helper.jpg")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(a.t * fps), max(n - 1, 0)))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame at that time")
    H, W = frame.shape[:2]
    print(f"video: {W}x{H}, {fps:.1f} fps, {n / fps:.1f} s")

    if a.zones:
        overlay = frame.copy()
        for z in load_zones(a.zones, W, H):
            pts = z.polygon.astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 255))
            cv2.polylines(frame, [pts], True, (0, 200, 255), 3)
            cv2.putText(frame, z.name, tuple(pts[0] + [6, 28]), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 255), 2, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    thick = max(1, W // 800)
    for i in range(1, 10):
        x, y = int(W * i / 10), int(H * i / 10)
        cv2.line(frame, (x, 0), (x, H), (255, 255, 255), thick)
        cv2.line(frame, (0, y), (W, y), (255, 255, 255), thick)
        for (px, py, txt) in ((x + 4, 24, f"{i / 10:.1f}"), (4, y - 6, f"{i / 10:.1f}")):
            cv2.putText(frame, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, txt, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                        cv2.LINE_AA)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, frame)
    print(f"-> {a.out}  (x along the top, y down the left side, both from 0 to 1)")


if __name__ == "__main__":
    main()
