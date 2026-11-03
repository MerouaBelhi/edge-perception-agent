"""PPE check on still images (one file or a whole folder).

    python scripts/detect_images.py --source datasets/ppe3/images/test --limit 20
    python scripts/detect_images.py --source my_photos/ --model models/ppe_yolov8n.onnx
    python scripts/detect_images.py --source photos/ --model models/site8_yolov8n.onnx --requires helmet vest mask

For each image: every person gets a status from the same rules as the video
pipeline (src/epa/safety.py), decided on this single frame:
    helmet     a helmet box sits on the person's head          (green)
    no_helmet  a bare head is detected, or no helmet is found on a
               clearly visible head                            (red)
    unknown    head too small, cut off or hidden               (grey)
With a model that has a "vest" class, each person also gets
    vest       a vest box on the torso
    no_vest    no vest on a clearly visible torso
    unknown    torso too small or hidden
Only the items in --requires (default: helmet vest) are checked and drawn, as on
a construction site; add mask for e.g. a paint room. The box is green if every
required item is worn, red if one is missing, grey if the system cannot tell.
Labels are short ('OK', 'no vest', 'not hard hat', '?') so crowds stay readable;
--show-ppe also draws the individual helmet / vest / mask boxes.
Annotated images go to --out, and a summary line per image is printed.
"""

import argparse
from dataclasses import replace
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402
from epa.hazards import HazardConfig, foot, frame_hazards  # noqa: E402
from epa.safety import SafetyConfig, frame_status, ppe_summary, short_label  # noqa: E402

MACHINE, CONE = (0, 215, 255), (0, 140, 255)
MACHINES = ("machinery", "vehicle")

COLORS = {"helmet": (60, 200, 60), "no_helmet": (40, 40, 230), "unknown": (160, 160, 160),
          "vest": (0, 165, 255), "no_vest": (40, 40, 230), "other_headwear": (200, 0, 200),
          "mask": (255, 200, 0), "no_mask": (40, 40, 230)}
VERDICT = {"ok": COLORS["helmet"], "bad": COLORS["no_helmet"], "unknown": COLORS["unknown"]}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw(img, persons, ppe, show_ppe=False, dets=(), hz=None, ok_text="OK"):
    """ok_text: label of a compliant worker ('hard hat' when only hard hats are checked)."""
    H, W = img.shape[:2]
    thick = max(1, W // 500)
    hz = hz or {"near": {}, "in_cone_area": set(), "hull": None}
    if hz["hull"] is not None:                                   # area marked by cones
        ov = img.copy()
        cv2.fillPoly(ov, [hz["hull"].astype(np.int32)], CONE)
        img[:] = cv2.addWeighted(ov, 0.2, img, 0.8, 0)
        cv2.polylines(img, [hz["hull"].astype(np.int32)], True, CONE, thick + 1)
    for d in dets:                                               # machines, vehicles, cones
        x1, y1, x2, y2 = map(int, d["box"])
        if d["cls_name"] in MACHINES:
            cv2.rectangle(img, (x1, y1), (x2, y2), MACHINE, thick + 1)
            cv2.putText(img, d["cls_name"], (x1 + 3, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, min(W / 1400, 0.8)), MACHINE, 1, cv2.LINE_AA)
        elif d["cls_name"] == "safety_cone":
            cv2.rectangle(img, (x1, y1), (x2, y2), CONE, 1)
    for pi, (mi, _) in hz["near"].items():                       # worker -> machine warning line
        pb, mb = dets[pi]["box"], dets[mi]["box"]
        fx, fy = foot(pb)
        cv2.line(img, (int(fx), int(fy)), (int(min(max(fx, mb[0]), mb[2])), int(mb[3])),
                 VERDICT["bad"], thick + 1)
    if show_ppe:
        for d in ppe:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[d["cls_name"]], 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # farthest first (feet highest in the image): labels of nearer people end on top
    for p in sorted(persons, key=lambda d: d["box"][3]):
        x1, y1, x2, y2 = map(int, p["box"])
        _, verdict = ppe_summary(p)
        label = short_label(p)
        label = ok_text if label == "OK" else label
        extra = [t for t, hit in (("near machine", p["track_id"] in hz["near"]),
                                  ("in cone area", p["track_id"] in hz["in_cone_area"])) if hit]
        if extra:
            verdict = "bad"
            label = " | ".join(([] if label == ok_text else [label]) + extra)
        col = VERDICT[verdict]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick + 1)
        # font sized to the person: the label must not be much wider than its box
        scale = max(0.35, min(W / 1400, 1.0))
        while scale > 0.35 and cv2.getTextSize(label, font, scale, 1)[0][0] > 1.3 * (x2 - x1):
            scale -= 0.05
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="image file or folder")
    p.add_argument("--model", default="models/site11hh_v2.onnx")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", default="results/images")
    p.add_argument("--limit", type=int, default=None, help="max number of images")
    p.add_argument("--requires", nargs="+", default=["helmet", "vest"],
                   choices=["helmet", "vest", "mask"], help="PPE items to check")
    p.add_argument("--show-ppe", action="store_true", help="also draw helmet / vest / mask boxes")
    p.add_argument("--no-hazards", action="store_true",
                   help="do not check machine proximity / coned-off areas (on by default when the "
                        "model has machinery / vehicle / safety_cone classes)")
    a = p.parse_args()

    src = Path(a.source)
    # a folder of sub-folders (one per test case, e.g. photos/bike_helmet/) is searched
    # recursively and summarised per sub-folder
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.suffix.lower() in EXTS)
    files = files[:a.limit] if a.limit else files
    if not files:
        raise SystemExit(f"No images found in {src}")

    det = OnnxYoloDetector(a.model, conf_thres=a.conf)
    missing = {"person", "helmet", "no_helmet"} - set(det.class_names)
    if missing:
        raise SystemExit(f"{a.model} is not the helmet model (missing classes {sorted(missing)})")
    for item, cls in (("vest", "vest"), ("mask", "mask")):
        if item in a.requires and cls not in det.class_names:
            print(f"warning: {a.model} has no '{cls}' class, {item} is not checked")
    cfg = SafetyConfig(check_vest="vest" in a.requires and "vest" in det.class_names,
                       check_mask="mask" in a.requires and "mask" in det.class_names)
    hazards = not a.no_hazards and bool(set(det.class_names) & {"machinery", "vehicle", "safety_cone"})
    hcfg = HazardConfig(proximity=bool(set(det.class_names) & set(MACHINES)),
                        cones="safety_cone" in det.class_names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hz_tot = {"near machine": 0, "in cone area": 0}

    totals = {"helmet": 0, "no_helmet": 0, "other_headwear": 0, "unknown": 0}
    vest_tot = {"vest": 0, "no_vest": 0, "unknown": 0}
    groups: dict[str, dict] = {}
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name}: cannot read, skipped")
            continue
        dets = det(img)
        persons = frame_status(dets, replace(cfg, input_scale=det.input_size / max(img.shape[:2])))
        ppe = [d for d in dets if d["cls_name"] in COLORS]
        group = f.parent.name if f.parent != src else "."
        (out / group).mkdir(parents=True, exist_ok=True)
        hz = frame_hazards(dets, img.shape, hcfg) if hazards else None
        if hz:                                  # only persons kept after duplicate removal
            kept = {p["track_id"] for p in persons}
            hz["near"] = {i: v for i, v in hz["near"].items() if i in kept}
            hz["in_cone_area"] = {i for i in hz["in_cone_area"] if i in kept}
            hz_tot["near machine"] += len(hz["near"])
            hz_tot["in cone area"] += len(hz["in_cone_area"])
        cv2.imwrite(str(out / group / (f.stem + ".jpg")),
                    draw(img, persons, ppe, a.show_ppe, dets, hz))
        counts = {k: sum(p["ppe"] == k for p in persons) for k in totals}
        g = groups.setdefault(group, {"images": 0, **{k: 0 for k in totals}})
        g["images"] += 1
        for k in totals:
            totals[k] += counts[k]
            g[k] += counts[k]
        line = (f"{group}/{f.name}: {len(persons)} people | helmet {counts['helmet']} | "
                f"no helmet {counts['no_helmet']} | not a hard hat {counts['other_headwear']} | "
                f"unknown {counts['unknown']}")
        if cfg.check_vest:
            vc = {k: sum(p["vest"] == k for p in persons) for k in vest_tot}
            for k in vest_tot:
                vest_tot[k] += vc[k]
            line += f" || vest {vc['vest']} | no vest {vc['no_vest']} | unknown {vc['unknown']}"
        if hz:
            n_m = sum(d["cls_name"] in MACHINES for d in dets)
            line += (f" || machines {n_m} | near machine {len(hz['near'])} | "
                     f"in cone area {len(hz['in_cone_area'])}")
        print(line)
    if len(groups) > 1:
        print("\n| Folder | Images | helmet | no helmet | not a hard hat | unknown |\n"
              "|---|---:|---:|---:|---:|---:|")
        for name, g in sorted(groups.items()):
            print(f"| {name} | {g['images']} | {g['helmet']} | {g['no_helmet']} | "
                  f"{g['other_headwear']} | {g['unknown']} |")
    print(f"\n{len(files)} images -> {out}/  | total: helmet {totals['helmet']}, "
          f"no helmet {totals['no_helmet']}, not a hard hat {totals['other_headwear']}, "
          f"unknown {totals['unknown']}"
          + (f" | vest {vest_tot['vest']}, no vest {vest_tot['no_vest']}, unknown {vest_tot['unknown']}"
             if cfg.check_vest else "")
          + (f" | near machine {hz_tot['near machine']}, in cone area {hz_tot['in cone area']}"
             if hazards else ""))


if __name__ == "__main__":
    main()
