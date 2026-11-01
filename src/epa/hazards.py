"""Site hazards around people: workers too close to machines, and areas marked
off with safety cones.

Detector classes (scripts/prepare_site_safety.py): person, machinery, vehicle,
safety_cone (+ the PPE classes, used by safety.py).

1. machine_proximity
   A worker whose feet are closer to a machine / vehicle than `near_factor`
   person heights, for >= `min_alert_s`  ->  one violation (re-armed once he
   has been clear of every machine for `rearm_s`).
   Distances are measured in the image, in units of the worker's own height,
   so they adapt to how far from the camera he stands. A single camera has no
   depth: "near" is approximate. To avoid the main 2D trap -- a worker far
   BEHIND a machine overlaps it in the image -- the gap is measured on the
   ground: the worker's feet are compared with the machine's lower part (its
   footprint), not with the whole box. Operators inside a cab (box inside the
   machine box, feet above its footprint) are ignored.
   Machines are tracked, so each one is also labelled moving / stationary
   (centre speed over the last second, in machine widths per second); the
   violation records it, and `only_moving=True` restricts alerts to moving ones.

2. cone_area_entry
   Cones seen in the last `cone_memory_s` seconds (they flicker in and out of
   detection) form an area: the convex hull of their ground points. A person
   (or a vehicle, `cone_area_keep_out`) whose ground point stays inside it for
   >= `min_alert_s` -> one violation. Fewer than 3 cones, or cones in a line
   (a hull without area), define no area.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class HazardConfig:
    person_class: str = "person"
    machine_classes: tuple = ("machinery", "vehicle")
    cone_class: str = "safety_cone"
    proximity: bool = True
    near_factor: float = 1.0           # "near": feet closer than this many person heights
    footprint: float = 0.5             # lower fraction of the machine box taken as its footprint
    operator_contain: float = 0.8      # person box this much inside a machine box: operator
    min_machine_scale: float = 0.5     # machine box at least this x the person's height: a
                                       # machine much smaller than the person is far behind him
    border_px: int = 10                # person box reaching the bottom of the frame: feet not
                                       # visible (close-up), so the ground gap is unknown
    only_moving: bool = False
    moving_speed: float = 0.15         # machine widths per second -> "moving"
    speed_window_s: float = 1.0
    min_alert_s: float = 0.5           # proximity / entry must last this long
    rearm_s: float = 2.0               # clear for this long before the same person can re-alert
    cones: bool = True
    min_cones: int = 3
    cone_memory_s: float = 3.0
    min_area_frac: float = 0.002       # hull area / frame area below this: cones in a line
    cone_area_keep_out: tuple = ("person",)


def foot(box):
    return (box[0] + box[2]) / 2, box[3]


def _contain(inner, outer) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / a if a > 0 else 0.0


def is_operator(p_box, m_box, cfg: HazardConfig) -> bool:
    """Person inside the machine box with feet above its footprint: in the cab."""
    mh = m_box[3] - m_box[1]
    ph = max(1.0, p_box[3] - p_box[1])
    if _contain(p_box, m_box) < cfg.operator_contain:
        return False
    # feet above the machine's footprint, or (seated in a cab: only the upper body is
    # visible) feet more than one person height above the machine's ground line
    return p_box[3] < m_box[3] - cfg.footprint * mh or m_box[3] - p_box[3] > 0.5 * ph


def ground_gap(p_box, m_box, cfg: HazardConfig) -> float:
    """Gap between the person's feet and the machine's footprint, in person heights
    (0 = touching or overlapping)."""
    fx, fy = foot(p_box)
    ph = max(1.0, p_box[3] - p_box[1])
    mx1, my1, mx2, my2 = m_box
    top = my2 - cfg.footprint * (my2 - my1)          # upper edge of the footprint band
    dx = max(mx1 - fx, 0.0, fx - mx2)
    dy = max(top - fy, 0.0, fy - my2)
    return float(np.hypot(dx, dy) / ph)


def nearest_machine(p_box, machines: list[dict], cfg: HazardConfig):
    """(machine, gap) of the closest machine that is not the person's own cab."""
    best, best_gap = None, float("inf")
    for m in machines:
        if is_operator(p_box, m["box"], cfg):
            return None, float("inf")                 # an operator is not "near" a machine
        if (m["box"][3] - m["box"][1]) < cfg.min_machine_scale * (p_box[3] - p_box[1]):
            continue                                   # depth mismatch: machine far behind
        g = ground_gap(p_box, m["box"], cfg)
        if g < best_gap:
            best, best_gap = m, g
    return best, best_gap


def cone_hull(points, frame_area: float, cfg: HazardConfig):
    """Convex hull (N, 2) int32 of the cone ground points, or None."""
    if len(points) < cfg.min_cones:
        return None
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) < cfg.min_area_frac * frame_area:
        return None
    return hull


def inside(hull, pt) -> bool:
    return hull is not None and cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), False) >= 0


def feet_visible(p_box, frame_h: int, cfg: HazardConfig) -> bool:
    return p_box[3] < frame_h - cfg.border_px


def frame_hazards(dets: list[dict], frame_shape, cfg: HazardConfig | None = None) -> dict:
    """Single-image version (no tracking: no speed, no duration). Returns
    {"near": {person index: (machine index, gap)}, "in_cone_area": set(indices),
     "hull": hull or None}. Indices refer to `dets`."""
    cfg = cfg or HazardConfig()
    machines = [dict(d, _i=i) for i, d in enumerate(dets) if d["cls_name"] in cfg.machine_classes]
    near, in_area = {}, set()
    cones = [foot(d["box"]) for d in dets if d["cls_name"] == cfg.cone_class]
    hull = cone_hull(cones, frame_shape[0] * frame_shape[1], cfg) if cfg.cones else None
    for i, d in enumerate(dets):
        if d["cls_name"] in cfg.cone_area_keep_out and inside(hull, foot(d["box"])):
            in_area.add(i)
        if (d["cls_name"] != cfg.person_class or not cfg.proximity
                or not feet_visible(d["box"], frame_shape[0], cfg)):
            continue
        m, g = nearest_machine(d["box"], machines, cfg)
        if m is not None and g < cfg.near_factor:
            near[i] = (m["_i"], g)
    return {"near": near, "in_cone_area": in_area, "hull": hull}


@dataclass
class _Alert:
    since: float | None = None
    last_bad: float = -1e9
    alerted: bool = False


@dataclass
class _Machine:
    trail: deque = field(default_factory=lambda: deque(maxlen=120))
    moving: bool = False


class HazardMonitor:
    """Video version: tracked machines (speed), remembered cones, durations.
    Shares the violation list with SafetyMonitor so ids stay unique."""

    def __init__(self, cfg: HazardConfig | None = None, violations: list | None = None,
                 evidence_dir: str | Path | None = None):
        self.cfg = cfg or HazardConfig()
        self.violations = violations if violations is not None else []
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.machines: dict[int, _Machine] = {}
        self.cones: dict[int, tuple] = {}             # track id -> (ground point, last seen t)
        self.prox: dict[int, _Alert] = {}
        self.area: dict[int, _Alert] = {}
        self.hull = None
        self.n_alerts = {"machine_proximity": 0, "cone_area_entry": 0}

    def _speed(self, tid, box, t) -> bool:
        st = self.machines.setdefault(tid, _Machine())
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        st.trail.append((t, c))
        while st.trail and t - st.trail[0][0] > self.cfg.speed_window_s:
            st.trail.popleft()
        t0, c0 = st.trail[0]
        if t - t0 < 0.5 * self.cfg.speed_window_s:
            return st.moving                           # not enough history yet
        w = max(1.0, box[2] - box[0])
        speed = np.hypot(c[0] - c0[0], c[1] - c0[1]) / w / (t - t0)
        st.moving = speed >= self.cfg.moving_speed
        return st.moving

    def _step(self, table, key, bad: bool, t) -> bool:
        """Duration / re-arm bookkeeping. True when a new alert must be raised."""
        a = table.setdefault(key, _Alert())
        if bad:
            a.last_bad = t
            a.since = t if a.since is None else a.since
            if not a.alerted and t - a.since >= self.cfg.min_alert_s:
                a.alerted = True
                return True
        else:
            a.since = None
            if a.alerted and t - a.last_bad >= self.cfg.rearm_s:
                a.alerted = False                      # clear long enough: re-arm
        return False

    def update(self, frame_idx: int, t: float, tracks: list[dict],
               frame: np.ndarray | None = None) -> list[dict]:
        """Annotates tracks in place: machines get "moving", people (or vehicles)
        get "hazard" = "near_machine" / "in_cone_area". Returns new violations."""
        cfg = self.cfg
        machines = [tr for tr in tracks if tr["cls_name"] in cfg.machine_classes]
        for m in machines:
            m["moving"] = self._speed(m["track_id"], m["box"], t)
        new = []

        if cfg.proximity:
            for p in (tr for tr in tracks if tr["cls_name"] == cfg.person_class):
                m, g = nearest_machine(p["box"], machines, cfg)
                if frame is not None and not feet_visible(p["box"], frame.shape[0], cfg):
                    m = None
                bad = m is not None and g < cfg.near_factor and (m["moving"] or not cfg.only_moving)
                if bad:
                    p["hazard"] = "near_machine"
                    p["near_machine"] = m["track_id"]
                if self._step(self.prox, p["track_id"], bad, t):
                    new.append(self._violation(frame_idx, t, "machine_proximity", p, frame, other=m,
                                               extra={"machine_id": m["track_id"],
                                                      "machine_class": m["cls_name"],
                                                      "machine_moving": bool(m["moving"]),
                                                      "gap_person_heights": round(g, 2)}))

        if cfg.cones:
            for c in (tr for tr in tracks if tr["cls_name"] == cfg.cone_class):
                self.cones[c["track_id"]] = (foot(c["box"]), t)
            self.cones = {k: v for k, v in self.cones.items() if t - v[1] <= cfg.cone_memory_s}
            area = frame.shape[0] * frame.shape[1] if frame is not None else 1920 * 1080
            self.hull = cone_hull([v[0] for v in self.cones.values()], area, cfg)
            for o in (tr for tr in tracks if tr["cls_name"] in cfg.cone_area_keep_out):
                bad = inside(self.hull, foot(o["box"]))
                if bad:
                    o["hazard"] = "in_cone_area"
                if self._step(self.area, o["track_id"], bad, t):
                    new.append(self._violation(frame_idx, t, "cone_area_entry", o, frame,
                                               extra={"class": o["cls_name"],
                                                      "cones": len(self.cones)}))
        return new

    def _violation(self, frame_idx, t, rule, obj, frame, other=None, extra=None) -> dict:
        v = {"id": len(self.violations) + 1, "frame": frame_idx, "t": round(t, 3),
             "t_start": round(self._since(rule, obj["track_id"], t), 3), "zone": "site",
             "track_id": obj["track_id"], "rule": rule, "box": [round(c, 1) for c in obj["box"]],
             "evidence_type": "detected", **(extra or {})}
        if frame is not None and self.evidence_dir is not None:
            box = obj["box"]
            if other is not None:                      # crop shows both the person and the machine
                ob = other["box"]
                box = [min(box[0], ob[0]), min(box[1], ob[1]), max(box[2], ob[2]), max(box[3], ob[3])]
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = (int(max(0, box[0] - 20)), int(max(0, box[1] - 20)),
                              int(min(W, box[2] + 20)), int(min(H, box[3] + 20)))
            name = f"violation_{v['id']:03d}_{rule}_track{obj['track_id']}.jpg"
            cv2.imwrite(str(self.evidence_dir / name), frame[y1:y2, x1:x2])
            v["evidence"] = name
        self.violations.append(v)
        self.n_alerts[rule] += 1
        return v

    def _since(self, rule, tid, t):
        a = (self.prox if rule == "machine_proximity" else self.area).get(tid)
        return a.since if a and a.since is not None else t

    def stats(self) -> dict:
        return {"hazard_violations": dict(self.n_alerts),
                "machines_tracked": len(self.machines),
                "machines_seen_moving": sum(m.moving for m in self.machines.values()),
                "cone_area_active": self.hull is not None}
