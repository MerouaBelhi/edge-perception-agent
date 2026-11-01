"""Build the PPE training set from Ultralytics' Construction-PPE dataset.

    python scripts/prepare_ppe_dataset.py --src datasets/construction-ppe --dst datasets/ppe3
    python scripts/prepare_ppe_dataset.py --src datasets/construction-ppe --dst datasets/ppe4 --with-vest

Source: https://docs.ultralytics.com/datasets/detect/construction-ppe/
(download: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip,
license AGPL-3.0). The source has 11 classes; this task only needs three:

    0 person     (source 6 "Person")
    1 helmet     (source 0 "helmet")
    2 no_helmet  (source 7 "no_helmet": a head without a hard hat)
    3 vest       (source 2 "vest", only with --with-vest). There is no "no_vest"
                 class: a missing vest is inferred from a visible torso with no
                 vest box on it (see src/epa/safety.py)

Fewer classes means a smaller head and faster convergence on ~1.1k images.
Images are symlinked where possible (Linux/macOS), copied otherwise (Windows
without Developer Mode does not allow symlinks).
"""

import argparse
import os
import shutil
from pathlib import Path

REMAP = {6: 0, 0: 1, 7: 2}
NAMES = ["person", "helmet", "no_helmet"]
VEST = {2: 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="datasets/construction-ppe")
    p.add_argument("--dst", default="datasets/ppe3")
    p.add_argument("--with-vest", action="store_true", help="add class 3 vest")
    a = p.parse_args()
    remap = {**REMAP, **VEST} if a.with_vest else REMAP
    names = NAMES + ["vest"] if a.with_vest else NAMES
    src, dst = Path(a.src).resolve(), Path(a.dst).resolve()

    for split in ("train", "val", "test"):
        img_dir = src / "images" / split
        if not img_dir.exists():
            continue
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        kept = boxes = 0
        for img in sorted(img_dir.iterdir()):
            lbl = src / "labels" / split / (img.stem + ".txt")
            lines = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if parts and int(parts[0]) in remap:
                        lines.append(" ".join([str(remap[int(parts[0])])] + parts[1:5]))
            link = dst / "images" / split / img.name
            if not link.exists():
                try:
                    os.symlink(img, link)
                except OSError:
                    shutil.copy2(img, link)
            (dst / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines))
            kept += 1
            boxes += len(lines)
        print(f"{split}: {kept} images, {boxes} boxes")

    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    print(f"wrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
