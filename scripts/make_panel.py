"""Tile annotated images into one figure for the README.

    python scripts/make_panel.py results/images_v3/a.jpg results/images_v3/b.jpg ... \\
        --cols 3 --out assets/results_panel.jpg

Each image is cropped to the cell's aspect ratio around its centre (no
stretching) and resized to the same cell size.
"""

import argparse

import cv2
import numpy as np


def fit(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y0:y0 + h, x0:x0 + w]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--cell", type=int, nargs=2, default=(420, 420), metavar=("W", "H"))
    p.add_argument("--gap", type=int, default=6)
    p.add_argument("--out", default="assets/results_panel.jpg")
    p.add_argument("--numbers", action="store_true",
                   help="write the image number in each cell (contact sheet for choosing images)")
    a = p.parse_args()

    w, h = a.cell
    cells = []
    for f in a.images:
        img = cv2.imread(f)
        if img is None:
            raise SystemExit(f"cannot read {f}")
        cell = fit(img, w, h)
        if a.numbers:
            n = str(len(cells) + 1)
            cv2.rectangle(cell, (0, 0), (60, 44), (0, 0, 0), -1)
            cv2.putText(cell, n, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            print(f"{n:>3}  {f}")
        cells.append(cell)
    rows = []
    for i in range(0, len(cells), a.cols):
        row = cells[i:i + a.cols]
        row += [np.full((h, w, 3), 255, np.uint8)] * (a.cols - len(row))
        gap = np.full((h, a.gap, 3), 255, np.uint8)
        rows.append(np.hstack(sum(([c, gap] for c in row), [])[:-1]))
    gap = np.full((a.gap, rows[0].shape[1], 3), 255, np.uint8)
    panel = np.vstack(sum(([r, gap] for r in rows), [])[:-1])
    cv2.imwrite(a.out, panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(cells)} images -> {a.out} ({panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    main()
