"""Test on an EXTERNAL dataset the model never saw: Hard Hat Workers
(Voxel51/hard-hat-detection on Hugging Face, ~5k construction-site images, CC0,
labels: helmet / head / person).

    pip install fiftyone
    python scripts/external_test.py --model models/site11_960.onnx --n 300

For every labelled head (helmet = worker wears a hard hat, head = bare head)
the system's decision is taken from the detected person whose head region
contains that head: helmet / no_helmet / other_headwear / unknown, or
"person not detected". Annotated images go to --out for visual review.
No training image comes from this dataset, so this measures generalisation
to new sites, cameras and people.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402
from epa.safety import SafetyConfig, frame_status, head_region  # noqa: E402
from detect_images import draw  # noqa: E402

COLS = ("helmet", "no_helmet", "other_headwear", "unknown", "person not detected")


def gt_heads(sample, W, H):
    """[(label, box)] for helmet / head boxes, from the first Detections field."""
    import fiftyone.core.labels as fol
    for name, val in sample.iter_fields():
        if isinstance(val, fol.Detections):
            out = []
            for d in val.detections:
                lab = d.label.lower()
                if lab in ("helmet", "head"):
                    x, y, w, h = d.bounding_box
                    out.append(("helmet" if lab == "helmet" else "no_helmet",
                                [x * W, y * H, (x + w) * W, (y + h) * H]))
            return out
    return []


def decision_for(head_box, persons, cfg):
    cx, cy = (head_box[0] + head_box[2]) / 2, (head_box[1] + head_box[3]) / 2
    best, best_d = None, float("inf")
    for p in persons:
        r = head_region(p["box"], cfg)
        if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
            d = np.hypot(cx - (r[0] + r[2]) / 2, cy - (r[1] + r[3]) / 2)
            if d < best_d:
                best, best_d = p, d
    return best["ppe"] if best else "person not detected"


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="models/site11_960.onnx")
    a.add_argument("--n", type=int, default=300, help="number of images")
    a.add_argument("--conf", type=float, default=0.3)
    a.add_argument("--out", default="results/external")
    a.add_argument("--save", type=int, default=60, help="annotated images to save")
    a.add_argument("--holdout", default=None,
                   help="sample-id list from prepare_hardhat.py (never trained on)")
    a = a.parse_args()

    if a.holdout:
        # the exact images prepare_hardhat.py kept out of training: fair before / after test
        import fiftyone as fo
        ds = fo.load_dataset("epa-hardhat").select(Path(a.holdout).read_text().split())
    else:
        import fiftyone.utils.huggingface as fouh
        ds = fouh.load_from_hub("Voxel51/hard-hat-detection", max_samples=a.n,
                                name="epa-hardhat-external", overwrite=True)
    det = OnnxYoloDetector(a.model, conf_thres=a.conf)
    base = SafetyConfig(check_vest="vest" in det.class_names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conf = {g: dict.fromkeys(COLS, 0) for g in ("helmet", "no_helmet")}
    saved = 0
    for s in ds:
        img = cv2.imread(s.filepath)
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = det(img)
        cfg = replace(base, input_scale=det.input_size / max(H, W))
        persons = frame_status(dets, cfg)
        for lab, box in gt_heads(s, W, H):
            conf[lab][decision_for(box, persons, cfg)] += 1
        if saved < a.save:
            ppe = [d for d in dets if d["cls_name"] != "person"]
            cv2.imwrite(str(out / Path(s.filepath).name), draw(img, persons, ppe, False, dets, None))
            saved += 1

    h, n = conf["helmet"], conf["no_helmet"]
    nh, nn = sum(h.values()), sum(n.values())
    bad = lambda r: r["no_helmet"] + r["other_headwear"]  # noqa: E731
    lines = [f"**External test: Hard Hat Workers, {len(ds)} images, {nh} hard hats, {nn} bare heads "
             f"(never seen in training)**", "",
             "| Bare heads caught ↑ | Bare heads called 'helmet' ↓ | False alarms on hard hats ↓ | "
             "Hard hats recognised ↑ |", "|---:|---:|---:|---:|",
             f"| {bad(n) / max(nn, 1):.0%} | {n['helmet'] / max(nn, 1):.0%} | {bad(h) / max(nh, 1):.0%} | "
             f"{h['helmet'] / max(nh, 1):.0%} |", "",
             "| Label | " + " | ".join(COLS) + " |", "|---|" + "---:|" * len(COLS)]
    for g in ("helmet", "no_helmet"):
        lines.append(f"| {g} | " + " | ".join(str(conf[g][c]) for c in COLS) + " |")
    text = "\n".join(lines)
    (out / "external_eval.md").write_text(text + "\n")
    print(text, f"\n\nannotated images -> {out}/")


if __name__ == "__main__":
    main()
