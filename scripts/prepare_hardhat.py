"""Add the Hard Hat Workers dataset (Voxel51/hard-hat-detection, CC0, ~5k images,
labels helmet / head / person) to the site11 training set, keeping a HELD-OUT
part that is never trained on, for a fair before / after external test.

    pip install fiftyone huggingface_hub
    python scripts/prepare_hardhat.py --site datasets/site11 --dst datasets/site11hh \\
        --model models/site11_960.onnx
    # before:  python scripts/external_test.py --model models/site11_960.onnx --holdout results/hardhat_holdout.txt
    # train:   python scripts/train_ppe.py --data datasets/site11hh/data.yaml --weights models/site11_960.pt ...
    # after:   python scripts/external_test.py --model models/<new>.onnx --holdout results/hardhat_holdout.txt

Why pseudo-labels: this dataset labels only helmet / head (and few persons).
Its images also show vests, machines, many unlabelled people... Added as they
are, they would teach the model that these are background. So the classes the
dataset does not label are filled in by the current model (confident
predictions only), while helmet / no_helmet come from the dataset's own labels,
which replace whatever the model predicted on heads (the point of the exercise:
the model called coloured Chinese hard hats "other headwear").
"""

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402
from epa.safety import iou  # noqa: E402

NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask", "other_headwear",
         "machinery", "vehicle", "safety_cone"]
GT_MAP = {"helmet": 1, "head": 2, "person": 0}
HEAD_IDS = {1, 2, 7}                      # never pseudo-labelled: the dataset's labels decide
PSEUDO_CONF = 0.5
PERSON_CONF = 0.5                         # --person-conf: a missed person is learned as background,
                                          # so a lower threshold for people is safer (v1 used 0.5)


def merge(gt: list[tuple[int, list]], pred: list[tuple[int, float, list]],
          person_conf: float = PERSON_CONF) -> list[tuple[int, list]]:
    """gt: [(cls, xyxy)], pred: [(cls, score, xyxy)] -> final [(cls, xyxy)]."""
    out = list(gt)
    gt_persons = [b for c, b in gt if c == 0]
    for c, s, b in pred:
        if c in HEAD_IDS or s < (person_conf if c == 0 else PSEUDO_CONF):
            continue
        if c == 0 and any(iou(b, g) > 0.6 for g in gt_persons):
            continue                                   # person already labelled
        out.append((c, b))
    return out


def yolo_lines(boxes, W, H) -> list[str]:
    lines = []
    for c, (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def gt_boxes(sample, W, H):
    import fiftyone.core.labels as fol
    for _, val in sample.iter_fields():
        if isinstance(val, fol.Detections):
            out = []
            for d in val.detections:
                c = GT_MAP.get(d.label.lower())
                if c is not None:
                    x, y, w, h = d.bounding_box
                    out.append((c, [x * W, y * H, (x + w) * W, (y + h) * H]))
            return out
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="datasets/site11")
    p.add_argument("--dst", default="datasets/site11hh")
    p.add_argument("--model", default="models/site11_960.onnx", help="for pseudo-labels")
    p.add_argument("--holdout", type=int, default=500, help="images kept for the external test")
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--holdout-file", default="results/hardhat_holdout.txt")
    p.add_argument("--person-conf", type=float, default=PERSON_CONF,
                   help="pseudo-label threshold for people (lower = fewer people learned as background)")
    a = p.parse_args()

    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh
    if fo.dataset_exists("epa-hardhat"):
        ds = fo.load_dataset("epa-hardhat")
    else:
        ds = fouh.load_from_hub("Voxel51/hard-hat-detection", name="epa-hardhat", persistent=True)
    ids = sorted(ds.values("id"))
    random.Random(0).shuffle(ids)
    hold, val, train = ids[:a.holdout], ids[a.holdout:a.holdout + a.val], ids[a.holdout + a.val:]
    Path(a.holdout_file).parent.mkdir(parents=True, exist_ok=True)
    Path(a.holdout_file).write_text("\n".join(hold) + "\n")
    print(f"{len(ids)} images: {len(train)} train, {len(val)} val, {len(hold)} held out "
          f"-> {a.holdout_file}")

    det = OnnxYoloDetector(a.model, conf_thres=min(0.25, a.person_conf))
    if list(det.class_names) != NAMES:
        raise SystemExit(f"{a.model} classes {det.class_names} != {NAMES}")
    dst = Path(a.dst).resolve()
    stats = {}
    for split, sel in (("train", train), ("val", val)):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = [0] * len(NAMES)
        for s in ds.select(sel):
            img = cv2.imread(s.filepath)
            if img is None:
                continue
            H, W = img.shape[:2]
            pred = [(NAMES.index(d["cls_name"]), d["score"], d["box"]) for d in det(img)]
            boxes = merge(gt_boxes(s, W, H), pred, a.person_conf)
            name = "hh_" + Path(s.filepath).name
            link = dst / "images" / split / name
            if not link.exists():
                link.symlink_to(Path(s.filepath).resolve())
            lines = yolo_lines(boxes, W, H)
            (dst / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            for l in lines:
                counts[int(l.split()[0])] += 1
        stats[split] = counts
        print(split, {n: c for n, c in zip(NAMES, counts) if c})

    site = Path(a.site).resolve()
    (dst / "data.yaml").write_text(
        f"path: {dst}\n"
        f"train: [{site}/images/train, {dst}/images/train]\n"
        f"val: [{site}/images/val, {dst}/images/val]\n"
        f"test: {site}/images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print(f"wrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
