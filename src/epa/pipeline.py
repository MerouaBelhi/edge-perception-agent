"""Video pipeline: detection -> multi-object tracking (ByteTrack) -> zone analytics.

Usage:
    python -m epa.pipeline --source video.mp4 --model models/yolov8n.onnx \
        --zones configs/zones.json --out results/run1

Outputs in --out:
    annotated.mp4   video with boxes, track IDs, zones and live counters
    events.jsonl    one JSON line per zone enter/exit event
    tracks.jsonl    one JSON line per frame: tracks + occupancy of every zone
    keyframes/      raw frames saved on every event and every --keyframe-every s
                    (used by the scene agent's vision-language model)
  with --safety (PPE model, zones with "requires": ["helmet"] and/or ["vest"]) and/or
  --hazards (model with machinery / vehicle / safety_cone):
    violations.jsonl  one line per violation (rule helmet_required / vest_required /
                      mask_required / machine_proximity / cone_area_entry)
    evidence/         cropped image of the worker for each violation
    summary.json    per-zone totals, per-class counts, unique tracks, latency stats
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from epa.hazards import HazardConfig, HazardMonitor, foot
from epa.safety import SafetyConfig, SafetyMonitor, ppe_summary
from epa.zones import Zone, anchor_point, load_zones

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
           (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0)]


def color_for(track_id: int):
    return PALETTE[track_id % len(PALETTE)]


PPE_COLORS = {"helmet": (60, 200, 60), "no_helmet": (40, 40, 230), "unknown": (160, 160, 160),
              "vest": (0, 165, 255), "no_vest": (40, 40, 230), "other_headwear": (200, 0, 200),
              "mask": (255, 200, 0), "no_mask": (40, 40, 230)}
OK, BAD, GREY = (60, 200, 60), (40, 40, 230), (160, 160, 160)


def ppe_label(tr) -> tuple[str, tuple]:
    text, verdict = ppe_summary(tr)
    return f'#{tr["track_id"]} {text}', {"ok": OK, "bad": BAD}.get(verdict, GREY)


CONE, MACHINE = (0, 140, 255), (0, 215, 255)
HAZARD_TEXT = {"near_machine": "near machine", "in_cone_area": "in cone area"}


def draw(frame, tracks, zones: list[Zone], fps: float, trails: dict,
         alert: str | None = None, n_violations: int | None = None, hull=None):
    H, W = frame.shape[:2]
    k = max(1.0, W / 960)                 # scale text and lines with the video size
    font = cv2.FONT_HERSHEY_SIMPLEX
    t1, t2 = max(1, round(k)), max(2, round(2 * k))
    overlay = frame.copy()
    for z in zones:
        pts = z.polygon.astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    for z in zones:
        cv2.polylines(frame, [z.polygon.astype(np.int32)], True, (0, 200, 255), t2)
    if hull is not None:                                          # area marked by cones
        ov = frame.copy()
        cv2.fillPoly(ov, [hull.astype(np.int32)], CONE)
        frame = cv2.addWeighted(ov, 0.20, frame, 0.80, 0)
        cv2.polylines(frame, [hull.astype(np.int32)], True, CONE, t2)
    by_id = {tr["track_id"]: tr for tr in tracks}
    for tr in tracks:                                             # worker -> machine warning line
        m = by_id.get(tr.get("near_machine"))
        if m is not None:
            mx = min(max(foot(tr["box"])[0], m["box"][0]), m["box"][2])
            cv2.line(frame, tuple(map(int, foot(tr["box"]))), (int(mx), int(m["box"][3])), BAD, t2)

    person_boxes = [tr["box"] for tr in tracks if tr["cls_name"] == "person"]

    def on_a_person(b):        # hide stray PPE detections (e.g. sun glare taken for a helmet)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return any(p[0] <= cx <= p[2] and p[1] - 0.15 * (p[3] - p[1]) <= cy <= p[3]
                   for p in person_boxes)

    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr["box"])
        if tr["cls_name"] in PPE_COLORS:                       # PPE evidence: thin box, no label
            if on_a_person(tr["box"]):
                cv2.rectangle(frame, (x1, y1), (x2, y2), PPE_COLORS[tr["cls_name"]], t1)
            continue
        if tr["cls_name"] == "safety_cone":                    # cones: thin box, no label
            cv2.rectangle(frame, (x1, y1), (x2, y2), CONE, t1)
            continue
        col = color_for(tr["track_id"])
        label = f'#{tr["track_id"]} {tr["cls_name"]} {tr["score"]:.2f}'
        if "moving" in tr:                                     # tracked machine / vehicle
            col = MACHINE
            label = f'#{tr["track_id"]} {tr["cls_name"]}' + (" moving" if tr["moving"] else "")
        if "ppe" in tr:
            label, col = ppe_label(tr)
        if tr.get("hazard"):
            label += " | " + HAZARD_TEXT[tr["hazard"]]
            col = BAD
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, round(1.5 * t2) if col == BAD else t2)
        fs = 0.5 * k
        (tw, th), _ = cv2.getTextSize(label, font, fs, t1)
        pad = round(3 * k)
        cv2.rectangle(frame, (x1, y1 - th - 2 * pad), (x1 + tw + 2 * pad, y1), col, -1)
        cv2.putText(frame, label, (x1 + pad, y1 - pad), font, fs, (255, 255, 255), t1, cv2.LINE_AA)
        pts = trails.get(tr["track_id"], [])
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, col, t2)

    # HUD
    lines = [f"FPS: {fps:5.1f}"] + [
        f'{z.name}: in={z.occupancy}  enter={z.entries}  exit={z.exits}' for z in zones]
    if n_violations is not None:
        lines.append(f"violations: {n_violations}")
    fs, lh = 0.55 * k, round(22 * k)
    width = max(cv2.getTextSize(l, font, fs, t1)[0][0] for l in lines) + round(24 * k)
    cv2.rectangle(frame, (8, 8), (8 + width, 8 + lh * len(lines) + round(10 * k)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (round(16 * k), round(8 + lh * (i + 1))), font, fs,
                    (255, 255, 255), t1, cv2.LINE_AA)
    if alert:
        bh = round(44 * k)
        cv2.rectangle(frame, (0, H - bh), (W, H), (40, 40, 230), -1)
        cv2.putText(frame, alert, (round(14 * k), H - round(14 * k)), font, 0.75 * k,
                    (255, 255, 255), t2, cv2.LINE_AA)
    return frame


def alert_message(v: dict) -> str:
    if v["rule"] == "machine_proximity":
        return (f'DANGER: worker #{v["track_id"]} close to {"moving " if v["machine_moving"] else ""}'
                f'{v["machine_class"]} #{v["machine_id"]}')
    if v["rule"] == "cone_area_entry":
        return f'VIOLATION: {v["class"]} #{v["track_id"]} inside the coned-off area'
    what = ("wears a non-work helmet / hat" if v.get("headwear") == "other_headwear"
            else f'without {v["missing"]}')
    return f'VIOLATION: worker #{v["track_id"]} {what} in {v["zone"]}'


def results_to_tracks(res, names) -> list[dict]:
    tracks = []
    b = res.boxes
    if b is None or b.id is None:
        return tracks
    for xyxy, tid, conf, cls in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy(),
                                    b.conf.cpu().numpy(), b.cls.cpu().numpy()):
        tracks.append({"track_id": int(tid), "box": xyxy.tolist(), "score": float(conf),
                       "cls_id": int(cls), "cls_name": names[int(cls)]})
    return tracks


def run(source: str, model_path: str, zones_path: str | None, out_dir: str,
        conf: float = 0.3, imgsz: int = 640, classes: list[int] | None = None,
        tracker: str = "bytetrack.yaml", max_frames: int | None = None,
        show: bool = False, keyframe_every: float = 2.0, safety: bool = False,
        min_ppe_width: float | None = None, hazards: bool = False,
        near_factor: float = 1.0, only_moving: bool = False,
        cone_keep_out: tuple = ("person",)) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kf_dir = out / "keyframes"
    kf_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    zones = load_zones(zones_path, W, H) if zones_path else []
    model = YOLO(model_path, task="detect")
    names = model.names
    monitor = None
    if safety:
        required = set().union(*(z.requires for z in zones)) if zones else set()
        need = {"person", "helmet", "no_helmet"}          # other_headwear / no_vest are optional
        if "vest" in required:
            need.add("vest")
        if "mask" in required:
            need |= {"mask", "no_mask"}
        missing = need - set(names.values())
        if missing:
            raise ValueError(f"--safety needs a PPE model with classes {sorted(need)}; "
                             f"{model_path} lacks {sorted(missing)} (see scripts/train_ppe.py)")
        # person size is judged at the detector's resolution, not the video's, and the
        # smoothing window is a fixed duration (1.5 s) whatever the frame rate
        monitor = SafetyMonitor(zones, SafetyConfig(input_scale=imgsz / max(W, H),
                                                    window=max(5, round(1.5 * src_fps)),
                                                    min_ppe_width=min_ppe_width),
                                evidence_dir=out / "evidence")
        if not monitor.zones:
            print('warning: no zone has "requires": ["helmet"] or ["vest"], no violation can be raised')
    hazard_mon = None
    if hazards:
        have = set(names.values())
        hcfg = HazardConfig(near_factor=near_factor, only_moving=only_moving,
                            cone_area_keep_out=tuple(cone_keep_out),
                            proximity=bool(have & {"machinery", "vehicle"}),
                            cones="safety_cone" in have)
        if not (hcfg.proximity or hcfg.cones):
            raise ValueError(f"--hazards needs a model with machinery / vehicle / safety_cone "
                             f"classes; {model_path} has {sorted(have)}")
        hazard_mon = HazardMonitor(hcfg, violations=monitor.violations if monitor else None,
                                   evidence_dir=out / "evidence")

    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (W, H))
    events_f = open(out / "events.jsonl", "w")
    tracks_f = open(out / "tracks.jsonl", "w")
    violations_f = open(out / "violations.jsonl", "w") if (monitor or hazard_mon) else None
    alert_until, alert_text = -1.0, None
    keyframes: list[dict] = []
    next_kf_t = 0.0

    trails: dict[int, list] = {}
    track_classes: dict[int, str] = {}
    latencies = []
    fps_ema = 0.0
    frame_idx = 0
    n_events = 0

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_idx >= max_frames):
            break
        t0 = time.perf_counter()
        res = model.track(frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz,
                          classes=classes, verbose=False)[0]
        tracks = results_to_tracks(res, names)
        dt = time.perf_counter() - t0
        latencies.append(dt * 1e3)
        fps_ema = (1.0 / dt) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1.0 / dt)

        t_sec = frame_idx / src_fps
        for tr in tracks:
            track_classes[tr["track_id"]] = tr["cls_name"]
            ax, ay = anchor_point(tr["box"])
            trails.setdefault(tr["track_id"], []).append((int(ax), int(ay)))
            trails[tr["track_id"]] = trails[tr["track_id"]][-30:]
        frame_events = []
        for z in zones:
            for ev in z.update(frame_idx, t_sec, tracks):
                events_f.write(json.dumps(ev) + "\n")
                frame_events.append(ev)
                n_events += 1
        new_v = monitor.update(frame_idx, t_sec, tracks, frame) if monitor else []
        new_v += hazard_mon.update(frame_idx, t_sec, tracks, frame) if hazard_mon else []
        for v in new_v:
            violations_f.write(json.dumps(v) + "\n")
            frame_events.append(v)
            alert_until = t_sec + 2.0
            alert_text = alert_message(v)
        tracks_f.write(json.dumps({
            "frame": frame_idx, "t": round(t_sec, 3),
            "tracks": [{"id": tr["track_id"], "class": tr["cls_name"],
                        "score": round(tr["score"], 3),
                        "box": [round(v, 1) for v in tr["box"]],
                        **({"ppe": tr["ppe"]} if "ppe" in tr else {}),
                        **{k: tr[k] for k in ("vest", "mask", "hazard", "near_machine", "moving")
                           if k in tr}} for tr in tracks],
            "occupancy": {z.name: z.occupancy for z in zones},
        }) + "\n")

        # Keyframes for the vision-language model: on every event + periodically
        reason = None
        if frame_events:
            reason = "event"
        elif t_sec >= next_kf_t:
            reason = "periodic"
        if reason:
            name = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(kf_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            keyframes.append({"frame": frame_idx, "t": round(t_sec, 3), "file": name,
                              "reason": reason})
            if reason == "periodic":
                next_kf_t = t_sec + keyframe_every

        vis = draw(frame, tracks, zones, fps_ema, trails,
                   alert=alert_text if t_sec <= alert_until else None,
                   n_violations=(len(monitor.violations) if monitor else
                                 len(hazard_mon.violations) if hazard_mon else None),
                   hull=hazard_mon.hull if hazard_mon else None)
        writer.write(vis)
        if show:
            cv2.imshow("Edge Perception Agent", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    writer.release()
    events_f.close()
    tracks_f.close()
    if violations_f:
        violations_f.close()
    (kf_dir / "index.json").write_text(json.dumps(keyframes, indent=1))
    if show:
        cv2.destroyAllWindows()

    lat = np.array(latencies) if latencies else np.array([0.0])
    summary = {
        "source": source,
        "model": model_path,
        "frames": frame_idx,
        "fps_source": src_fps,
        "duration_s": round(frame_idx / src_fps, 2),
        "zone_definitions": [{"name": z.name, "classes": sorted(z.classes) if z.classes else None,
                              "requires": sorted(z.requires)} for z in zones],
        "keyframes": len(keyframes),
        "resolution": [W, H],
        "unique_tracks": len(track_classes),
        "tracks_per_class": dict(Counter(track_classes.values())),
        "zones": [z.stats() for z in zones],
        "events": n_events,
        "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                       "p95": float(np.percentile(lat, 95))},
        "fps_mean": float(1000.0 / lat.mean()) if lat.mean() > 0 else 0.0,
    }
    if monitor:
        summary["safety"] = monitor.stats()
    if hazard_mon:
        summary["hazards"] = hazard_mon.stats()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description="Detection + tracking + zone analytics")
    p.add_argument("--source", required=True, help="video file, RTSP URL or webcam index")
    p.add_argument("--model", default="models/yolov8n.onnx", help=".pt or .onnx")
    p.add_argument("--zones", default=None, help="zones JSON config")
    p.add_argument("--out", default="results/run")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="COCO class ids to keep, e.g. 0 for person")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--keyframe-every", type=float, default=2.0,
                   help="seconds between periodic keyframes (events always save one)")
    p.add_argument("--safety", action="store_true",
                   help="PPE compliance in zones with \"requires\": [\"helmet\"] / [\"helmet\", \"vest\"]")
    p.add_argument("--min-ppe-width", type=float, default=None,
                   help="ignore helmet boxes narrower than this fraction of the person width "
                        "(near workers in crowds: helmets of people behind them), e.g. 0.25")
    p.add_argument("--hazards", action="store_true",
                   help="worker-machine proximity and coned-off areas (model with machinery / "
                        "vehicle / safety_cone classes)")
    p.add_argument("--near-factor", type=float, default=1.0,
                   help="'near a machine' = feet closer than this many person heights")
    p.add_argument("--only-moving", action="store_true", help="proximity alerts for moving machines only")
    p.add_argument("--cone-keep-out", nargs="+", default=["person"], choices=["person", "vehicle"],
                   help="who must stay out of the area marked by cones")
    a = p.parse_args()
    s = run(a.source, a.model, a.zones, a.out, a.conf, a.imgsz, a.classes,
            max_frames=a.max_frames, show=a.show, keyframe_every=a.keyframe_every, safety=a.safety,
            min_ppe_width=a.min_ppe_width, hazards=a.hazards, near_factor=a.near_factor,
            only_moving=a.only_moving, cone_keep_out=tuple(a.cone_keep_out))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
