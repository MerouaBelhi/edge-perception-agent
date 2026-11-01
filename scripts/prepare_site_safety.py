"""Build the 11-class site-safety dataset: PPE, "headwear that is not a hard hat",
machines, vehicles and safety cones.

    pip install fiftyone            # only for the Open Images part
    python scripts/prepare_site_safety.py --css datasets/css --dst datasets/site8
    python scripts/prepare_site_safety.py --css datasets/css --dst datasets/site8 --no-openimages

Classes of the output dataset:

    0 person          1 helmet (hard hat)   2 no_helmet (bare head)
    3 vest            4 no_vest             5 mask
    6 no_mask         7 other_headwear (cap, hat, sun hat, bicycle / football helmet...)
    8 machinery       9 vehicle             10 safety_cone

Sources
1. Construction Site Safety (Roboflow, CC BY 4.0), exported as "YOLOv8": classes
   Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
   Safety Vest, machinery, vehicle. Classes are mapped BY NAME (several versions
   of this dataset exist with different class orders).
2. Open Images V7 (Google; annotations CC BY 4.0, images CC BY 2.0), downloaded
   with FiftyOne: images that contain non-work headwear. Their headwear boxes
   become `other_headwear`, their people `person`, and their cars, trucks, buses
   and vans `vehicle` (left unlabelled, they would teach the model that vehicles
   are background). Open Images has no cone class. Without these, the model has
   never seen a sun hat or a bicycle helmet and calls them hard hats (found in
   the headwear stress test, see README).
   Images that also contain the generic Open Images class "Helmet" are skipped:
   that class mixes hard hats with other helmets, and a hard hat left unlabelled
   would teach the model that hard hats are background.
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml

NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask", "other_headwear",
         "machinery", "vehicle", "safety_cone"]

# Construction Site Safety names (normalised: lower case, no spaces / dashes / underscores)
CSS_MAP = {"person": 0, "hardhat": 1, "nohardhat": 2, "safetyvest": 3, "nosafetyvest": 4,
           "mask": 5, "nomask": 6, "machinery": 8, "vehicle": 9, "safetycone": 10}

OI_HEADWEAR = ["Bicycle helmet", "Cowboy hat", "Football helmet", "Hat", "Sun hat", "Swim cap"]
OI_PERSON = {"Person", "Man", "Woman", "Boy", "Girl"}
OI_VEHICLE = {"Car", "Truck", "Bus", "Van", "Taxi", "Ambulance", "Land vehicle", "Vehicle"}
OI_SKIP = {"Helmet"}                      # mixes hard hats with other helmets
OI_SPLITS = {"train": "train", "val": "validation", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------- Construction Site Safety
def css_class_map(names) -> dict[int, int]:
    """Source class id -> output class id, by name. Unknown names are dropped."""
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return {i: CSS_MAP[norm(n)] for i, n in enumerate(names) if norm(n) in CSS_MAP}


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Roboflow export: <root>/{train,valid,test}/{images,labels}. Also accepts 'val'."""
    out = {}
    for name, split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        d = root / name
        if (d / "images").is_dir() and split not in out:
            out[split] = d
    return out


# class order of the published dataset (roboflow-universe-projects/construction-site-safety,
# Kaggle snehilsanyal/construction-site-safety-image-dataset-roboflow), used if data.yaml is missing
CSS_DEFAULT_NAMES = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person",
                     "Safety Cone", "Safety Vest", "machinery", "vehicle"]


def find_css_root(root: Path) -> Path:
    """The folder holding train/ (and valid/, test/): root itself or one level below
    (the Kaggle zip adds a folder such as css-data/)."""
    for cand in [root] + sorted(d for d in root.iterdir() if d.is_dir()):
        if (cand / "train" / "images").is_dir():
            return cand
    raise SystemExit(f"no train/images folder found in {root} or its sub-folders")


def convert_css(root: Path, dst: Path) -> dict:
    root = find_css_root(root)
    yml = next((f for f in (root / "data.yaml", root.parent / "data.yaml") if f.exists()), None)
    names = yaml.safe_load(yml.read_text())["names"] if yml else CSS_DEFAULT_NAMES
    if yml is None:
        print(f"no data.yaml in {root}: using the published class order {CSS_DEFAULT_NAMES}")
    cmap = css_class_map(names)
    missing = set(CSS_MAP) - {norm(n) for n in (names.values() if isinstance(names, dict) else names)}
    if missing:
        print(f"warning: the dataset has no class for {sorted(missing)}")
    splits = find_split_dirs(root)
    if not splits:
        raise SystemExit(f"no train/valid/test folders with images/ found in {root}")
    stats = {}
    for split, d in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for img in sorted((d / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lines = []
            lbl = d / "labels" / (img.stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if len(p) == 5 and int(p[0]) in cmap:      # 5 values: boxes, not polygons
                        lines.append(" ".join([str(cmap[int(p[0])])] + p[1:]))
            name = "css_" + img.name
            link_or_copy(img, dst / "images" / split / name)
            (dst / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
        stats[split] = (n_img, n_box)
        print(f"css {split}: {n_img} images, {n_box} boxes")
    return stats


# ------------------------------------------------------------------------- Open Images
def oi_yolo_lines(detections) -> list[str] | None:
    """FiftyOne Open Images detections -> YOLO lines, or None if the image is skipped.
    `detections`: objects with .label, .bounding_box [x, y, w, h] (relative, top-left)
    and optional .IsGroupOf."""
    if any(d.label in OI_SKIP for d in detections):
        return None
    lines, kept = [], {0: [], 9: []}
    for d in detections:
        if getattr(d, "IsGroupOf", False) or getattr(d, "IsDepiction", False):
            continue                                 # a box around a crowd, or a drawing
        if d.label in OI_HEADWEAR:
            cls = 7
        elif d.label in OI_PERSON:
            cls = 0
        elif d.label in OI_VEHICLE:
            cls = 9
        else:
            continue
        x, y, w, h = d.bounding_box
        box = (x, y, x + w, y + h)
        if cls in kept:                    # "Person" + "Man", "Car" + "Vehicle" on one object
            if any(_iou(box, q) > 0.8 for q in kept[cls]):
                continue
            kept[cls].append(box)
        lines.append(f"{cls} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
    if not any(l.startswith("7 ") for l in lines):
        return None
    return lines


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def add_openimages(dst: Path, n_train: int, n_eval: int, seed: int = 0) -> dict:
    try:
        import fiftyone.zoo as foz
    except ImportError:
        raise SystemExit("Open Images needs FiftyOne: pip install fiftyone "
                         "(or run with --no-openimages)")
    stats = {}
    for split, oi_split in OI_SPLITS.items():
        n = n_train if split == "train" else n_eval
        # all labels of the loaded images are kept (only_matching=False), so people are
        # labelled too, not just the headwear
        ds = foz.load_zoo_dataset("open-images-v7", split=oi_split, label_types=["detections"],
                                  classes=OI_HEADWEAR, max_samples=n, shuffle=True, seed=seed,
                                  dataset_name=f"epa-headwear-{oi_split}")
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        kept = skipped = boxes = 0
        for s in ds:
            dets = s.ground_truth.detections if s.ground_truth else []
            lines = oi_yolo_lines(dets)
            if lines is None:
                skipped += 1
                continue
            src = Path(s.filepath)
            name = "oi_" + src.name
            link_or_copy(src, dst / "images" / split / name)
            (dst / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            kept += 1
            boxes += len(lines)
        stats[split] = (kept, boxes)
        print(f"open images {split}: {kept} images ({skipped} skipped), {boxes} boxes")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--css", required=True, help="Construction Site Safety, Roboflow YOLOv8 export")
    p.add_argument("--dst", default="datasets/site8")
    p.add_argument("--no-openimages", action="store_true")
    p.add_argument("--oi-train", type=int, default=1500, help="Open Images headwear images (train)")
    p.add_argument("--oi-eval", type=int, default=200, help="... per val / test split")
    a = p.parse_args()
    dst = Path(a.dst).resolve()
    convert_css(Path(a.css), dst)
    if not a.no_openimages:
        add_openimages(dst, a.oi_train, a.oi_eval)
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print(f"wrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
