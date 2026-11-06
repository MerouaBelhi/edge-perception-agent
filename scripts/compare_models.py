"""Side-by-side before / after on the held-out external images, chosen
automatically: the images where the two models' decisions differ most, in BOTH
directions (fixed and broken), so the review is not cherry-picked.

    python scripts/compare_models.py --before models/site11_960.onnx --after models/site11hh.onnx \\
        --holdout results/hardhat_holdout.txt --out results/compare
    # README figure, full decision (hard hat + vest) drawn:
    python scripts/compare_models.py --before models/site11_960.onnx --after models/site11hh_v2.onnx \
        --out results/compare_vest --check-vest --panel 4

Per image, every labelled head (helmet / bare head) is scored correct or not for
each model (same rule as external_test.py). Images are ranked by
(fixed - broken) heads; the top --n "improved" and the top --n "worse" ones are
saved as before | after pairs, plus a summary line per image.
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
from epa.safety import SafetyConfig, frame_status  # noqa: E402
from detect_images import draw  # noqa: E402
from external_test import decision_for, gt_heads  # noqa: E402


def correct(label, decision):
    if label == "helmet":
        return decision == "helmet"
    return decision in ("no_helmet", "other_headwear")


def run(det, img, base):
    H, W = img.shape[:2]
    cfg = replace(base, input_scale=det.input_size / max(H, W))
    dets = det(img)
    return dets, frame_status(dets, cfg), cfg


def label_bar(img, text, ok):
    bar = np.full((32, img.shape[1], 3), (40, 140, 40) if ok else (40, 40, 40), np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--holdout", default="results/hardhat_holdout.txt")
    p.add_argument("--out", default="results/compare")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--check-vest", action="store_true",
                   help="draw the full decision (hard hat + vest) instead of the hard hat only; "
                        "the head counts in the title bars do not change")
    p.add_argument("--panel", type=int, default=0, metavar="K",
                   help="also stack the K most improved pairs into <out>/before_after.jpg")
    a = p.parse_args()

    import fiftyone as fo
    ds = fo.load_dataset("epa-hardhat").select(Path(a.holdout).read_text().split())
    dets = {k: OnnxYoloDetector(m, conf_thres=0.3) for k, m in (("before", a.before), ("after", a.after))}
    base = SafetyConfig()                       # scoring: hard hat decisions only
    show = replace(base, check_vest=a.check_vest)   # drawing
    rows = []
    for s in ds:
        img = cv2.imread(s.filepath)
        if img is None:
            continue
        H, W = img.shape[:2]
        heads = gt_heads(s, W, H)
        if not heads:
            continue
        res = {k: run(d, img, base) for k, d in dets.items()}
        ok = {k: [correct(lab, decision_for(box, r[1], r[2])) for lab, box in heads]
              for k, r in res.items()}
        fixed = sum(not b and a_ for b, a_ in zip(ok["before"], ok["after"]))
        broken = sum(b and not a_ for b, a_ in zip(ok["before"], ok["after"]))
        rows.append((fixed - broken, fixed, broken, sum(ok["before"]), sum(ok["after"]), len(heads),
                     s.filepath))

    out = Path(a.out)
    for sub in ("improved", "worse"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: -r[0])
    picks = [("improved", r) for r in rows[:a.n] if r[0] > 0] + \
            [("worse", r) for r in rows[::-1][:a.n] if r[0] < 0]
    pairs = []
    for rank, (sub, (delta, fixed, broken, nb, na, n, path)) in enumerate(picks):
        img = cv2.imread(path)
        panels = []
        for k, d in dets.items():
            dts, persons, _ = run(d, img, show)
            ppe = [x for x in dts if x["cls_name"] != "person"]
            # without --check-vest only hard hats are judged (this dataset labels heads,
            # not vests), so a compliant worker is labelled 'hard hat', not 'OK'
            pic = draw(img.copy(), persons, ppe, False, dts, None,
                       ok_text="OK" if a.check_vest else "hard hat")
            nk = nb if k == "before" else na
            panels.append(label_bar(pic, f"{k}: {nk}/{n} heads correct", nk == n))
        pair = np.hstack(panels)
        cv2.imwrite(str(out / sub / f"{rank:02d}_{Path(path).name}"), pair)
        if sub == "improved":
            pairs.append(pair)
    if a.panel and pairs:
        width = 832
        rows_img = [cv2.resize(x, (width, round(x.shape[0] * width / x.shape[1])))
                    for x in pairs[:a.panel]]
        cv2.imwrite(str(out / "before_after.jpg"), np.vstack(rows_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"panel of {len(rows_img)} pairs -> {out / 'before_after.jpg'}")

    tot = np.array([r[1:6] for r in rows]).sum(0)
    print(f"{len(rows)} images | heads fixed {tot[0]}, broken {tot[1]} | correct before "
          f"{tot[2]}/{tot[4]}, after {tot[3]}/{tot[4]}")
    print(f"images improved: {sum(r[0] > 0 for r in rows)}, worse: {sum(r[0] < 0 for r in rows)}, "
          f"same: {sum(r[0] == 0 for r in rows)}  -> {out}/improved, {out}/worse")


if __name__ == "__main__":
    main()
