"""Person-level evaluation of the PPE rules against the dataset labels.

The detector metrics (mAP) score boxes; this scores what the safety system
actually decides for each worker.

    # choose rule settings on the validation split ...
    python scripts/eval_safety.py --data datasets/ppe3 --split val --sweep
    # ... then report once on the test split
    python scripts/eval_safety.py --data datasets/ppe3 --split test --sweep
    # helmet + vest (+ non-work headwear, mask) datasets
    python scripts/eval_safety.py --data datasets/site8 --model models/site8_yolov8n.onnx --split val --sweep

Class names are read from <data>/data.yaml, so ppe3 (person, helmet, no_helmet),
ppe4 (+ vest) and site8 (+ no_vest, mask, no_mask, other_headwear) all work.

Head ground truth per labelled person: a labelled helmet / no_helmet /
other_headwear box whose centre lies inside the person box, in its upper 40 %
(nearest to the top centre wins). Persons with no head label are "unlabelled"
and not scored (the datasets often omit no_helmet labels); their predictions are
reported. A worker in other_headwear has no hard hat: the system is right if it
says no_helmet or other_headwear (the second one names the reason).

Vest ground truth: a labelled vest / no_vest box whose centre lies in the lower
85 % of the person box (wide on purpose: many persons are upper-body close-ups).
 - datasets with a no_vest class (site8): persons with neither are "unlabelled";
 - without one (ppe4, Construction-PPE): vests are labelled exhaustively (checked
   by eye on persons without a vest box), so no vest label means "no_vest".

Detections are computed once per image; every rule variant is evaluated on the
same detections, so differences come only from the rules.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402
from epa.safety import SafetyConfig, frame_status, iou  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_NAMES = ["person", "helmet", "no_helmet", "vest"]   # if data.yaml is missing
HEAD_GT = ("helmet", "no_helmet", "other_headwear")
BAD_HEAD = ("no_helmet", "other_headwear")                    # "no hard hat" decisions

VARIANTS = {
    "current rules": {},
    "fixed head region on close-ups (v1)": {"head_bottom_partial": 0.35},
    "no duplicate-box removal": {"dedupe_persons": False},
    "no inference from absence": {"infer_from_absence": False},
    "centred helmet (dx <= 0.35 w)": {"max_center_dx": 0.35},
    "centred helmet (dx <= 0.25 w)": {"max_center_dx": 0.25},
    "no side margin": {"head_margin_x": 0.0},
    "shorter head region (25 %)": {"head_bottom": 0.25},
    "centred 0.35 + shorter 25 %": {"max_center_dx": 0.35, "head_bottom": 0.25},
    "helmet >= 0.20 person width": {"min_ppe_width": 0.20},
    "helmet >= 0.25 person width": {"min_ppe_width": 0.25},
}
VEST_VARIANTS = {
    "vest rules (default)": {},
    "fixed torso band 15-65 % (v1)": {"full_body_ratio": 0.0},
    "fixed torso band 15-95 %": {"full_body_ratio": 0.0, "torso_bottom": 0.95},
    "vest >= 0.2 person width": {"min_vest_width": 0.2},
    "vest >= 0.4 person width": {"min_vest_width": 0.4},
}


def load_names(data: Path) -> list[str]:
    f = data / "data.yaml"
    if not f.exists():
        return DEFAULT_NAMES
    names = yaml.safe_load(f.read_text())["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def read_labels(path: Path, w: int, h: int, names=DEFAULT_NAMES):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:5])
                out.append((names[c], [(x - bw / 2) * w, (y - bh / 2) * h,
                                       (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def gt_status(labels):
    persons = [b for c, b in labels if c == "person"]
    heads = [(c, b) for c, b in labels if c in HEAD_GT]
    status = ["unlabelled"] * len(persons)
    used = set()
    for c, hb in heads:
        cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
        best, best_d = None, float("inf")
        for i, pb in enumerate(persons):
            if i in used:
                continue
            if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[1] + 0.4 * (pb[3] - pb[1]):
                d = np.hypot(cx - (pb[0] + pb[2]) / 2, cy - pb[1])
                if d < best_d:
                    best, best_d = i, d
        if best is not None:
            used.add(best)
            status[best] = c
    return persons, status


def gt_vest(labels, persons, explicit_no_vest: bool):
    def on_torso(pb, b):
        h = pb[3] - pb[1]
        return pb[0] <= (b[0] + b[2]) / 2 <= pb[2] and pb[1] + 0.15 * h <= (b[1] + b[3]) / 2 <= pb[3]
    vests = [b for c, b in labels if c == "vest"]
    no_vests = [b for c, b in labels if c == "no_vest"]
    out = []
    for pb in persons:
        if any(on_torso(pb, v) for v in vests):
            out.append("vest")
        elif not explicit_no_vest or any(on_torso(pb, v) for v in no_vests):
            out.append("no_vest")
        else:
            out.append("unlabelled")
    return out


def gt_mask(labels, persons):
    """mask / no_mask label whose centre lies in the upper 40 % of the person box
    (75 % for upper-body boxes, less than twice as tall as wide: close-ups), nearest
    wins, one per person; persons without one are "unlabelled"."""
    faces = [(c, b) for c, b in labels if c in ("mask", "no_mask")]
    status = ["unlabelled"] * len(persons)
    used = set()
    for c, fb in faces:
        cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        def band(pb):
            h = pb[3] - pb[1]
            return (0.4 if h >= 2 * (pb[2] - pb[0]) else 0.75) * h
        cands = [(np.hypot(cx - (pb[0] + pb[2]) / 2, cy - pb[1]), i) for i, pb in enumerate(persons)
                 if i not in used and pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[1] + band(pb)]
        if cands:
            i = min(cands)[1]
            used.add(i)
            status[i] = c
    return status


def match_preds(gps, preds):
    """Ground-truth person index -> predicted person (IoU >= 0.5, greedy)."""
    pairs = sorted(((iou(gp, pp["box"]), i, j) for i, gp in enumerate(gps)
                    for j, pp in enumerate(preds)), reverse=True)
    gi, pj, match = set(), set(), {}
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in gi and j not in pj:
            gi.add(i)
            pj.add(j)
            match[i] = preds[j]
    return match


def evaluate(cache, cfg):
    """Confusion counts {gt status: {decision: n}} for the head (helmet) decision."""
    cols = ("helmet", "no_helmet", "other_headwear", "unknown", "missed")
    conf = {g: dict.fromkeys(cols, 0) for g in HEAD_GT + ("unlabelled",)}
    for c in cache:
        preds = frame_status(c["dets"], replace(cfg, input_scale=c["scale"]))
        match = match_preds(c["persons"], preds)
        for i, g in enumerate(c["head"]):
            conf[g][match[i]["ppe"] if i in match else "missed"] += 1
    return conf


def evaluate_vest(cache, cfg):
    cols = ("vest", "no_vest", "unknown", "missed")
    conf = {g: dict.fromkeys(cols, 0) for g in ("vest", "no_vest", "unlabelled")}
    for c in cache:
        preds = frame_status(c["dets"], replace(cfg, input_scale=c["scale"], check_vest=True))
        match = match_preds(c["persons"], preds)
        for i, g in enumerate(c["vest"]):
            conf[g][match[i]["vest"] if i in match else "missed"] += 1
    return conf


def evaluate_mask(cache, cfg):
    cols = ("mask", "no_mask", "unknown", "missed")
    conf = {g: dict.fromkeys(cols, 0) for g in ("mask", "no_mask", "unlabelled")}
    for c in cache:
        preds = frame_status(c["dets"], replace(cfg, input_scale=c["scale"], check_mask=True))
        match = match_preds(c["persons"], preds)
        for i, g in enumerate(c["mask"]):
            conf[g][match[i]["mask"] if i in match else "missed"] += 1
    return conf


def pct(a, b):
    return f"{a / b:.0%}" if b else "n/a"


def head_row(name, c):
    h, n, o = c["helmet"], c["no_helmet"], c["other_headwear"]
    nh, nn, no = sum(h.values()), sum(n.values()), sum(o.values())
    row = (f"| {name} | {pct(sum(n[k] for k in BAD_HEAD), nn)} | {pct(n['helmet'], nn)} | "
           f"{pct(sum(h[k] for k in BAD_HEAD), nh)} | {pct(h['helmet'], nh)} |")
    if no:
        row += f" {pct(sum(o[k] for k in BAD_HEAD), no)} | {pct(o['helmet'], no)} |"
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/ppe3")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--model", default="models/ppe_yolov8n.onnx")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--sweep", action="store_true", help="compare all rule variants")
    p.add_argument("--out", default=None, help="markdown output (default results/eval_<split>.md)")
    a = p.parse_args()

    data = Path(a.data)
    names = load_names(data)
    explicit_no_vest = "no_vest" in names
    files = sorted(f for f in (data / "images" / a.split).iterdir() if f.suffix.lower() in EXTS)
    det = OnnxYoloDetector(a.model, conf_thres=a.conf)
    cache = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        H, W = img.shape[:2]
        labels = read_labels(data / "labels" / a.split / (f.stem + ".txt"), W, H, names)
        gps, gst = gt_status(labels)
        cache.append({"dets": det(img), "persons": gps, "head": gst,
                      "scale": det.input_size / max(H, W),
                      "vest": gt_vest(labels, gps, explicit_no_vest), "mask": gt_mask(labels, gps)})

    base = SafetyConfig()
    variants = VARIANTS if a.sweep else {"current rules": {}}
    confs = {name: evaluate(cache, replace(base, **over)) for name, over in variants.items()}
    c = confs["current rules"]
    n_other = sum(c["other_headwear"].values())
    out = [f"**{a.split} split: {len(cache)} images, {sum(c['helmet'].values())} labelled workers "
           f"with a hard hat, {sum(c['no_helmet'].values())} bare heads"
           + (f", {n_other} non-work headwear" if n_other else "") + "**", "",
           "| Rules | Bare heads caught ↑ | Bare heads called 'helmet' ↓ | "
           "False alarms on hard hats ↓ | Hard hats recognised ↑ |"
           + (" Non-work headwear caught ↑ | Non-work headwear called 'helmet' ↓ |" if n_other else ""),
           "|---|---:|---:|---:|---:|" + ("---:|---:|" if n_other else "")]
    out += [head_row(name, cf) for name, cf in confs.items()]
    out += ["", "Current rules, full breakdown (rows: label, columns: system decision; "
            "'other' = not a hard hat):", "",
            "| Label | helmet | no_helmet | other | unknown | person not detected |",
            "|---|---:|---:|---:|---:|---:|"]
    for g in HEAD_GT + ("unlabelled",):
        if g == "other_headwear" and not n_other:
            continue
        r = c[g]
        out.append(f"| {g} | {r['helmet']} | {r['no_helmet']} | {r['other_headwear']} | "
                   f"{r['unknown']} | {r['missed']} |")

    if "vest" in det.class_names:
        vvar = VEST_VARIANTS if a.sweep else {"vest rules (default)": {}}
        vconfs = {name: evaluate_vest(cache, replace(base, **over)) for name, over in vvar.items()}
        vc = vconfs["vest rules (default)"]
        out += ["", f"**Vests** ({a.split} split: {sum(vc['vest'].values())} labelled vests, "
                f"{sum(vc['no_vest'].values())} without)", "",
                "| Rules | Missing vests caught ↑ | Missing vests called 'vest' ↓ | "
                "False alarms on vests ↓ | Vests recognised ↑ |", "|---|---:|---:|---:|---:|"]
        for name, x in vconfs.items():
            v, n = x["vest"], x["no_vest"]
            nv, nn = sum(v.values()), sum(n.values())
            out.append(f"| {name} | {pct(n['no_vest'], nn)} | {pct(n['vest'], nn)} | "
                       f"{pct(v['no_vest'], nv)} | {pct(v['vest'], nv)} |")
        out += ["", "Vest, default rules (rows: label, columns: system decision):", "",
                "| Label | vest | no_vest | unknown | person not detected |", "|---|---:|---:|---:|---:|"]
        for g in ("vest", "no_vest") + (("unlabelled",) if explicit_no_vest else ()):
            r = vc[g]
            out.append(f"| {g} | {r['vest']} | {r['no_vest']} | {r['unknown']} | {r['missed']} |")

    if "mask" in det.class_names:
        mc = evaluate_mask(cache, base)
        m, n = mc["mask"], mc["no_mask"]
        nm, nn = sum(m.values()), sum(n.values())
        out += ["", f"**Masks** ({a.split} split: {nm} labelled masks, {nn} without; masks are "
                "never inferred, only detected)", "",
                "| Missing masks caught ↑ | Missing masks called 'mask' ↓ | False alarms on masks ↓ | "
                "Masks recognised ↑ |", "|---:|---:|---:|---:|",
                f"| {pct(n['no_mask'], nn)} | {pct(n['mask'], nn)} | {pct(m['no_mask'], nm)} | "
                f"{pct(m['mask'], nm)} |", "",
                "| Label | mask | no_mask | unknown | person not detected |", "|---|---:|---:|---:|---:|"]
        for g in ("mask", "no_mask", "unlabelled"):
            r = mc[g]
            out.append(f"| {g} | {r['mask']} | {r['no_mask']} | {r['unknown']} | {r['missed']} |")

    text = "\n".join(out)
    path = Path(a.out or f"results/eval_{a.split}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
