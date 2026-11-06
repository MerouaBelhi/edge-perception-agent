"""Find which images a README panel was made from, so it can be rebuilt after the
rules change (each tile is matched to the most similar image of a folder).

    python scripts/find_panel_sources.py assets/results_panel.jpg results/images_v2 --cols 3 --rows 2

Prints the file names in panel order (left to right, top to bottom).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}
S = 32


def sig(img):
    g = cv2.cvtColor(cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def centre_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("panel")
    p.add_argument("folder")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--rows", type=int, default=2)
    a = p.parse_args()

    panel = cv2.imread(a.panel)
    H, W = panel.shape[:2]
    th, tw = H // a.rows, W // a.cols
    m = 6                                            # skip the gaps between tiles
    tiles = [panel[r * th + m:(r + 1) * th - m, c * tw + m:(c + 1) * tw - m]
             for r in range(a.rows) for c in range(a.cols)]
    tsig = [sig(t) for t in tiles]

    cands = []
    for f in sorted(Path(a.folder).rglob("*")):
        if f.suffix.lower() in EXTS:
            img = cv2.imread(str(f))
            if img is not None:
                cands.append((f, sig(img), sig(centre_square(img))))
    if not cands:
        raise SystemExit(f"no images in {a.folder}")
    for i, ts in enumerate(tsig):
        best = max(cands, key=lambda c: max(float(ts @ c[1]), float(ts @ c[2])))
        score = max(float(tsig[i] @ best[1]), float(tsig[i] @ best[2])) / (S * S)
        print(f"{i + 1}  {best[0].name}  (similarity {score:.2f})")


if __name__ == "__main__":
    main()
