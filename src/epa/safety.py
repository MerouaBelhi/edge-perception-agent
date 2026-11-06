"""Worker-safety rules: PPE compliance (hard hat, high-visibility vest, mask) in zones.

A zone lists what it requires, e.g.  "requires": ["helmet", "vest"].
Detector classes (see scripts/prepare_ppe_dataset.py, scripts/prepare_site_safety.py):

    person
    helmet / no_helmet / other_headwear   head    (other_headwear: cap, sun hat,
                                                   bicycle helmet... NOT a hard hat)
    vest / no_vest                        torso   (no_vest is optional: the
                                                   Construction-PPE data has none)
    mask / no_mask                        face    (optional)

Persons are tracked (stable IDs); PPE boxes are per-frame evidence that gets
attached to a person, then smoothed over time.

Pipeline per frame and per item:
  1. associate   each PPE box -> the person whose head / torso region contains its
                 centre (closest wins, one box per person and item)
  2. smooth      per track, vote over the last `window` observations
                 helmet: helmet | no_helmet | other_headwear | unknown
                 vest:   vest | no_vest | unknown          mask: mask | no_mask | unknown
                 A missing helmet / vest is also evidence when the head / torso is
                 clearly visible (person tall enough at the detector's resolution,
                 not cut by the border, not hidden behind a nearer person) and no
                 PPE box sits on it. Masks are never inferred: faces turn away.
  3. rule        a missing item inside a zone that requires it for >= `min_violation_s`
                 -> one violation (+ evidence crop), re-armed only after the person
                 leaves the zone or wears the item for `rearm_helmet_s`

"unknown" never raises a violation: false alarms erode trust in a safety system
faster than misses are noticed, and unknowns are logged for review (or the VLM).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from epa.zones import Zone


@dataclass
class SafetyConfig:
    person_class: str = "person"
    ok_class: str = "helmet"
    missing_class: str = "no_helmet"
    wrong_class: str = "other_headwear"  # headwear that is not a hard hat (counts as missing)
    window: int = 15               # frames of evidence kept per track
    min_observations: int = 3      # below this the status stays "unknown"
    vote_ratio: float = 0.6        # share of observations needed to decide
    min_violation_s: float = 1.0   # time without the item inside the zone before alerting
    head_top: float = -0.10        # head region, as fractions of the person box height
    head_bottom: float = 0.35
    head_bottom_partial: float = 0.55   # upper-body box (close-up): the head sits lower and is
                                        # bigger (found: a hard hat centred at 40 % of the box)
    head_margin_x: float = 0.10    # widen the head region sideways (fraction of width)
    max_center_dx: float | None = None  # if set: PPE box centre must be within this fraction
                                        # of the person width from the person's centre line
    min_ppe_width: float | None = None  # if set: head PPE box must be at least this fraction of
                                        # the person width; a helmet of a farther worker seen inside
                                        # the large head region of a near one is ~0.1-0.25 wide
    infer_from_absence: bool = True     # nothing found on a clearly visible head -> "no helmet"
    min_person_height: int = 100        # px AT THE DETECTOR INPUT; smaller: too small to judge
    input_scale: float = 1.0            # detector input px per frame px (imgsz / max(W, H)):
                                        # a 250 px worker in a 1920 px frame is only 54 px for
                                        # a 416 px detector, and his helmet ~5 px (undetectable)
    border_px: int = 10                 # person box near the top border: head may be cut off
                                        # (3 px missed a legs-only box starting at y=5)
    rearm_helmet_s: float = 2.0         # an item must be worn this long to re-arm an alert: a
                                        # shorter run is usually another worker's PPE behind
    reid_window_s: float = 3.0          # duplicate-alert guard: a violating track that vanished
    reid_iou: float = 0.3               # and a new track at the same place = same worker (ID switch)
    # ---- vest ("requires": ["vest"]); the helmet settings above are unchanged
    check_vest: bool = False            # set automatically when a zone requires a vest
    vest_class: str = "vest"
    no_vest_class: str = "no_vest"
    torso_top: float = 0.15             # torso region, fractions of the person box height
    torso_bottom: float = 0.65          # full body: the vest ends around mid-height
    torso_bottom_partial: float = 1.0   # upper-body box (close-up, cut by the frame or hidden
    full_body_ratio: float = 2.0        # legs): height/width below this -> torso reaches the bottom
    torso_margin_x: float = 0.05
    vest_absent_ratio: float = 0.9      # INFERRED "no vest" needs >= 90 % of the window without a
                                        # vest: vest detections flicker (a vest seen now and then
                                        # IS worn); a detected no_vest box only needs vote_ratio
    min_vest_width: float = 0.30        # vest box >= this x person width: the vest of a worker
                                        # behind, seen through a near worker's torso, is narrower
    min_vest_conf: float = 0.6          # a vest box below this score is not trusted: the worker is
                                        # "unknown" (not OK, no alarm). Found on external images: a red
                                        # and a pink shirt taken for vests at 0.43 and 0.57 -> "OK".
                                        # Test split: no new false alarms, but vests recognised
                                        # 87 % -> 67 % (the rest become "unknown")
    dedupe_persons: bool = True         # single images: drop a second, nested box on the same
                                        # person (upper body inside full body); the video uses
                                        # tracking + the duplicate-alert guard instead
    # ---- mask ("requires": ["mask"]): detected evidence only, never inferred
    check_mask: bool = False
    mask_class: str = "mask"
    face_bottom_partial: float = 0.75   # upper-body box (close-up): the face sits lower in the
                                        # box than on a full body (found: masks at 40-70 %)
    no_mask_class: str = "no_mask"

    @property
    def head_classes(self) -> tuple:
        return (self.ok_class, self.missing_class, self.wrong_class)

    @property
    def torso_classes(self) -> tuple:
        return (self.vest_class, self.no_vest_class)

    @property
    def face_classes(self) -> tuple:
        return (self.mask_class, self.no_mask_class)


# status values per item: (worn, missing) -- anything else is "unknown"
ITEM_STATUS = {"helmet": ({"helmet"}, {"no_helmet", "other_headwear"}),
               "vest": ({"vest"}, {"no_vest"}),
               "mask": ({"mask"}, {"no_mask"})}
STATUS_KEY = {"helmet": "ppe", "vest": "vest", "mask": "mask"}   # key on the person dict

INFERRED = "inferred_no_helmet"
INFERRED_NO_VEST = "inferred_no_vest"


# ------------------------------------------------------------------- geometry
def _centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _inside(pt, r) -> bool:
    return r[0] <= pt[0] <= r[2] and r[1] <= pt[1] <= r[3]


def head_region(box, cfg: SafetyConfig):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    bottom = cfg.head_bottom if h >= cfg.full_body_ratio * w else cfg.head_bottom_partial
    return (x1 - cfg.head_margin_x * w, y1 + cfg.head_top * h,
            x2 + cfg.head_margin_x * w, y1 + bottom * h)


def torso_region(box, cfg: SafetyConfig):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    bottom = cfg.torso_bottom if h >= cfg.full_body_ratio * w else cfg.torso_bottom_partial
    return (x1 - cfg.torso_margin_x * w, y1 + cfg.torso_top * h,
            x2 + cfg.torso_margin_x * w, y1 + bottom * h)


def face_region(box, cfg: SafetyConfig):
    """Head region for masks; reaches lower on upper-body boxes, like the torso."""
    x1, y1, x2, y2 = head_region(box, cfg)
    w, h = box[2] - box[0], box[3] - box[1]
    if h < cfg.full_body_ratio * w:
        y2 = box[1] + cfg.face_bottom_partial * h
    return x1, y1, x2, y2


def scale_ok(det: dict, person_box, cfg: SafetyConfig) -> bool:
    """Head PPE box plausible in size for this person (depth consistency)."""
    if cfg.min_ppe_width is None:
        return True
    return (det["box"][2] - det["box"][0]) >= cfg.min_ppe_width * (person_box[2] - person_box[0])


def vest_ok_size(det: dict, person_box, cfg: SafetyConfig) -> bool:
    return (det["box"][2] - det["box"][0]) >= cfg.min_vest_width * (person_box[2] - person_box[0])


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_footprint(a, b, min_contain: float = 0.7, max_dy: float = 0.05) -> bool:
    """Two person boxes that are the same worker: the smaller one lies mostly
    inside the other AND both touch the ground at the same height. A different
    person behind the worker overlaps too, but stands higher in the image."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    if smaller <= 0 or inter / smaller < min_contain:
        return False
    h = max(a[3] - a[1], b[3] - b[1])
    return abs(a[3] - b[3]) <= max_dy * h


# ---------------------------------------------------------------- association
def _match(persons, dets, region_fn, size_ok, cfg) -> dict[int, str]:
    """Greedy one-to-one matching, most confident boxes first: a box goes to the
    person whose region contains its centre, the closest region centre winning."""
    regions = {p["track_id"]: region_fn(p["box"], cfg) for p in persons}
    boxes = {p["track_id"]: p["box"] for p in persons}
    taken: dict[int, str] = {}
    for det in sorted(dets, key=lambda d: -d["score"]):
        c = _centre(det["box"])
        best, best_d = None, float("inf")
        for tid, r in regions.items():
            if tid in taken or not _inside(c, r) or not size_ok(det, boxes[tid], cfg):
                continue
            if cfg.max_center_dx is not None and region_fn is head_region:
                bx1, _, bx2, _ = boxes[tid]
                if abs(c[0] - (bx1 + bx2) / 2) > cfg.max_center_dx * (bx2 - bx1):
                    continue
            rc = _centre(r)
            d = np.hypot(c[0] - rc[0], c[1] - rc[1])
            if d < best_d:
                best, best_d = tid, d
        if best is not None:
            taken[best] = det["cls_name"]
    return taken


def associate(persons: list[dict], ppe: list[dict], cfg: SafetyConfig) -> dict[int, str]:
    """{track_id: "helmet" | "no_helmet" | "other_headwear"} for persons with head
    evidence in this frame."""
    return _match(persons, [d for d in ppe if d["cls_name"] in cfg.head_classes],
                  head_region, scale_ok, cfg)


WEAK_VEST = "weak_vest"


def associate_torso(persons: list[dict], dets: list[dict], cfg: SafetyConfig) -> dict[int, str]:
    """{track_id: "vest" | "no_vest" | "weak_vest"}. A vest box below cfg.min_vest_conf
    still occupies the torso (so no "no vest" is inferred) but is returned as
    "weak_vest": the caller treats it as "unknown", never as a worn vest."""
    boxes = []
    for d in dets:
        if d["cls_name"] not in cfg.torso_classes:
            continue
        if d["cls_name"] == cfg.vest_class and d["score"] < cfg.min_vest_conf:
            d = dict(d, cls_name=WEAK_VEST)
        boxes.append(d)
    return _match(persons, boxes, torso_region, vest_ok_size, cfg)


def associate_vests(persons: list[dict], vests: list[dict], cfg: SafetyConfig) -> set[int]:
    """Track ids wearing a vest in this frame."""
    return {t for t, c in associate_torso(persons, vests, cfg).items() if c == cfg.vest_class}


def associate_masks(persons: list[dict], dets: list[dict], cfg: SafetyConfig) -> dict[int, str]:
    """{track_id: "mask" | "no_mask"}."""
    return _match(persons, [d for d in dets if d["cls_name"] in cfg.face_classes],
                  face_region, scale_ok, cfg)


def _hidden_by_nearer(p, persons, point) -> bool:
    """Someone nearer the camera (lower feet in the image) covers this point?"""
    return any(o is not p and o["box"][3] > p["box"][3] and _inside(point, o["box"])
               for o in persons)


def head_clearly_visible(p: dict, persons: list[dict], ppe: list[dict],
                         cfg: SafetyConfig) -> bool:
    """True only when the absence of a helmet box on this head is meaningful."""
    x1, y1, x2, y2 = p["box"]
    if (y2 - y1) * cfg.input_scale < cfg.min_person_height or y1 <= cfg.border_px:
        return False
    r = head_region(p["box"], cfg)
    # Any head PPE box on this head, even one matched to another (overlapping /
    # duplicate) person box, means the head is NOT empty: do not infer.
    if any(d["cls_name"] in cfg.head_classes and _inside(_centre(d["box"]), r)
           and scale_ok(d, p["box"], cfg) for d in ppe):
        return False
    return not _hidden_by_nearer(p, persons, ((x1 + x2) / 2, (y1 + max(r[3], y1)) / 2))


def torso_clearly_visible(p: dict, persons: list[dict], dets: list[dict],
                          cfg: SafetyConfig) -> bool:
    """True only when the absence of a vest box on this torso is meaningful."""
    x1, y1, x2, y2 = p["box"]
    if (y2 - y1) * cfg.input_scale < cfg.min_person_height:
        return False
    if y1 <= cfg.border_px:                 # cut by the top of the image: often only the legs
        return False                        # are in view, so the torso may not be there at all
    r = torso_region(p["box"], cfg)
    if any(d["cls_name"] in cfg.torso_classes and _inside(_centre(d["box"]), r)
           and vest_ok_size(d, p["box"], cfg) for d in dets):
        return False                        # a vest box is there (maybe matched to a duplicate)
    return not _hidden_by_nearer(p, persons, _centre(r))


# ---------------------------------------------------------------- smoothing
def smooth_head(votes, cfg: SafetyConfig) -> str:
    obs = [v for v in votes if v is not None]
    if len(obs) < cfg.min_observations:
        return "unknown"
    n_wrong = obs.count(cfg.wrong_class)
    n_missing = n_wrong + sum(v in (cfg.missing_class, INFERRED) for v in obs)
    if n_missing >= cfg.vote_ratio * len(obs):
        # the person IS wearing something, just not a hard hat
        return "other_headwear" if n_wrong >= 0.5 * n_missing else "no_helmet"
    if obs.count(cfg.ok_class) >= cfg.vote_ratio * len(obs):
        return "helmet"
    return "unknown"


def smooth_vest(votes, cfg: SafetyConfig) -> str:
    obs = [v for v in votes if v is not None]
    if len(obs) < cfg.min_observations:
        return "unknown"
    n_det, n_inf = obs.count(cfg.no_vest_class), obs.count(INFERRED_NO_VEST)
    if n_det and n_det + n_inf >= cfg.vote_ratio * len(obs):
        return "no_vest"                    # positive evidence: a detected no_vest box
    if n_inf >= cfg.vest_absent_ratio * len(obs):
        return "no_vest"
    if obs.count(cfg.vest_class) > 0:
        return "vest"
    return "unknown"


def smooth_mask(votes, cfg: SafetyConfig) -> str:
    obs = [v for v in votes if v is not None]
    if len(obs) < cfg.min_observations:
        return "unknown"
    for status in (cfg.no_mask_class, cfg.mask_class):
        if obs.count(status) >= cfg.vote_ratio * len(obs):
            return status
    return "unknown"


# ------------------------------------------------------------ single images
def same_person(small, big, min_contain: float = 0.8) -> bool:
    """`small` is a second box on the person in `big`: mostly inside it AND sharing
    its top (head) or its bottom (feet). A person standing behind is also inside the
    box of the one in front, but neither edge lines up."""
    ix = max(0.0, min(small[2], big[2]) - max(small[0], big[0]))
    iy = max(0.0, min(small[3], big[3]) - max(small[1], big[1]))
    area = (small[2] - small[0]) * (small[3] - small[1])
    if area <= 0 or ix * iy / area < min_contain:
        return False
    h = big[3] - big[1]
    return abs(small[1] - big[1]) <= 0.10 * h or abs(small[3] - big[3]) <= 0.05 * h


def dedupe_persons(persons: list[dict]) -> list[dict]:
    """Keep one box per person: most confident first, drop boxes nested in a kept one."""
    kept: list[dict] = []
    for p in sorted(persons, key=lambda d: -d["score"]):
        if not any(same_person(p["box"], k["box"]) or same_person(k["box"], p["box"]) for k in kept):
            kept.append(p)
    return kept


def frame_status(dets: list[dict], cfg: SafetyConfig) -> list[dict]:
    """Single-image version of the rules (no tracking, no smoothing): returns the
    person detections with "ppe" (helmet | no_helmet | other_headwear | unknown),
    plus "vest" / "mask" when cfg.check_vest / cfg.check_mask."""
    persons = [dict(d, track_id=i) for i, d in enumerate(dets) if d["cls_name"] == cfg.person_class]
    if cfg.dedupe_persons:
        persons = dedupe_persons(persons)
    seen = associate(persons, dets, cfg)
    for p in persons:
        vote = seen.get(p["track_id"])
        if vote is None and cfg.infer_from_absence and head_clearly_visible(p, persons, dets, cfg):
            vote = cfg.missing_class
        p["ppe"] = vote if vote is not None else "unknown"
    if cfg.check_vest:
        torso = associate_torso(persons, dets, cfg)
        for p in persons:
            v = torso.get(p["track_id"])
            if v == WEAK_VEST:
                p["vest"] = "unknown"
            else:
                p["vest"] = v or ("no_vest" if torso_clearly_visible(p, persons, dets, cfg) else "unknown")
    if cfg.check_mask:
        face = associate_masks(persons, dets, cfg)
        for p in persons:
            p["mask"] = face.get(p["track_id"], "unknown")
    return persons


def ppe_summary(p: dict) -> tuple[str, str]:
    """Display text and verdict ("ok" | "bad" | "unknown") for an annotated person,
    e.g. 'helmet', 'not a hard hat | vest'. "bad" if anything checked is missing,
    "ok" if everything checked is worn."""
    items = [i for i in ("helmet", "vest", "mask") if STATUS_KEY[i] in p]
    states = [p[STATUS_KEY[i]] for i in items]
    if any(s in ITEM_STATUS[i][1] for i, s in zip(items, states)):
        verdict = "bad"
    elif all(s in ITEM_STATUS[i][0] for i, s in zip(items, states)):
        verdict = "ok"
    else:
        verdict = "unknown"

    def word(item, s):
        if s == "other_headwear":
            return "not a hard hat"
        if s == "unknown":
            return "unknown" if len(items) == 1 else "? " + item
        return s.replace("_", " ")
    return " | ".join(word(i, s) for i, s in zip(items, states)), verdict


def short_label(p: dict) -> str:
    """Compact label for crowded images: 'OK', or only what is wrong / unknown,
    e.g. 'no vest', 'not hard hat', '? vest', '?'."""
    text, verdict = ppe_summary(p)
    if verdict == "ok":
        return "OK"
    items = [i for i in ("helmet", "vest", "mask") if STATUS_KEY[i] in p]
    states = [p[STATUS_KEY[i]] for i in items]
    if all(s == "unknown" for s in states):
        return "?"
    words = {"other_headwear": "not hard hat", "no_helmet": "no helmet", "no_vest": "no vest",
             "no_mask": "no mask"}
    bad = [words[s] for s in states if s in words]
    if bad:
        return " | ".join(bad)
    return " | ".join("? " + i for i, s in zip(items, states) if s == "unknown")


# --------------------------------------------------------------- video monitor
@dataclass
class _TrackState:
    votes: deque
    vest_votes: deque
    mask_votes: deque
    status: str = "unknown"              # helmet: helmet | no_helmet | other_headwear | unknown
    vest_status: str = "unknown"         # vest:   vest | no_vest | unknown
    mask_status: str = "unknown"         # mask:   mask | no_mask | unknown
    ok_since: dict = field(default_factory=dict)          # item -> start of current "worn" run
    last_box: list | None = None
    last_t: float = -1.0
    missing_since: dict = field(default_factory=dict)     # (item, zone) -> t first seen violating
    alerted: set = field(default_factory=set)             # (item, zone) already alerted


class SafetyMonitor:
    def __init__(self, zones: list[Zone], cfg: SafetyConfig | None = None,
                 evidence_dir: str | Path | None = None):
        cfg = cfg or SafetyConfig()
        self.zones = [z for z in zones if z.requires & set(ITEM_STATUS)]
        needs = set().union(*(z.requires for z in self.zones)) if self.zones else set()
        self.cfg = replace(cfg, check_vest=cfg.check_vest or "vest" in needs,
                           check_mask=cfg.check_mask or "mask" in needs)
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.states: dict[int, _TrackState] = {}
        self.violations: list[dict] = []
        self.duplicates: list[dict] = []    # alerts suppressed as tracker ID switches

    def _state(self, tid) -> _TrackState:
        w = self.cfg.window
        return self.states.setdefault(tid, _TrackState(deque(maxlen=w), deque(maxlen=w),
                                                       deque(maxlen=w)))

    def update(self, frame_idx: int, t: float, tracks: list[dict],
               frame: np.ndarray | None = None) -> list[dict]:
        """Annotate person tracks with their PPE status (in place: "ppe", and "vest" /
        "mask" when checked) and return new violations. Call AFTER the zones were
        updated for this frame."""
        cfg = self.cfg
        persons = [tr for tr in tracks if tr["cls_name"] == cfg.person_class]
        others = [tr for tr in tracks if tr["cls_name"] != cfg.person_class]
        head = associate(persons, others, cfg)
        torso = associate_torso(persons, others, cfg) if cfg.check_vest else {}
        face = associate_masks(persons, others, cfg) if cfg.check_mask else {}

        new = []
        visible_now = {p["track_id"] for p in persons}
        for p in persons:
            tid = p["track_id"]
            st = self._state(tid)
            st.last_box, st.last_t = p["box"], t

            vote = head.get(tid)
            if vote is None and cfg.infer_from_absence and head_clearly_visible(p, persons, others, cfg):
                vote = INFERRED
            st.votes.append(vote)
            st.status = p["ppe"] = smooth_head(st.votes, cfg)
            checks = {"helmet": st.status}

            if cfg.check_vest:
                vote = torso.get(tid)
                if vote is None and torso_clearly_visible(p, persons, others, cfg):
                    vote = INFERRED_NO_VEST
                if vote == WEAK_VEST:
                    vote = None                     # an untrusted vest: no evidence either way
                st.vest_votes.append(vote)
                st.vest_status = p["vest"] = smooth_vest(st.vest_votes, cfg)
                checks["vest"] = st.vest_status
            if cfg.check_mask:
                st.mask_votes.append(face.get(tid))
                st.mask_status = p["mask"] = smooth_mask(st.mask_votes, cfg)
                checks["mask"] = st.mask_status

            for item, status in checks.items():
                worn, missing = ITEM_STATUS[item]
                if status in worn:
                    st.ok_since.setdefault(item, t)
                else:
                    st.ok_since.pop(item, None)
                kept = item in st.ok_since and t - st.ok_since[item] >= cfg.rearm_helmet_s
                for z in self.zones:
                    if item not in z.requires:
                        continue
                    key = (item, z.name)
                    inside = z.contains(tid)
                    if not inside or status not in missing:
                        st.missing_since.pop(key, None)
                        if not inside or kept:
                            st.alerted.discard(key)      # re-arm: left, or really put it on
                        continue
                    since = st.missing_since.setdefault(key, t)
                    if t - since < cfg.min_violation_s or key in st.alerted:
                        continue
                    st.alerted.add(key)
                    v = self._violation(frame_idx, t, since, z.name, p, st, item, status,
                                        visible_now, persons, frame)
                    if v is not None:
                        new.append(v)
        return new

    def _violation(self, frame_idx, t, since, zone, p, st, item, status, visible_now,
                   persons, frame):
        tid, cfg = p["track_id"], self.cfg
        rule = f"{item}_required"
        dup_of, reason = self._same_worker_as(tid, p["box"], zone, rule, t, visible_now, persons)
        if dup_of is not None:
            self.duplicates.append({"t": round(t, 3), "zone": zone, "track_id": tid, "rule": rule,
                                    "same_as_violation": dup_of["id"], "reason": reason})
            return None
        # how the decision was reached, for review / the VLM check
        if item == "helmet":
            if status == "other_headwear":
                ev = "detected_other_headwear"
            else:
                ev = ("detected" if st.votes.count(cfg.missing_class) >= st.votes.count(INFERRED)
                      else "inferred_from_absence")
        elif item == "vest":
            ev = ("detected" if st.vest_votes.count(cfg.no_vest_class)
                  >= st.vest_votes.count(INFERRED_NO_VEST) else "inferred_from_absence")
        else:
            ev = "detected"
        v = {"id": len(self.violations) + 1, "frame": frame_idx, "t": round(t, 3),
             "t_start": round(since, 3), "zone": zone, "track_id": tid, "rule": rule,
             "missing": item, "box": [round(c, 1) for c in p["box"]], "evidence_type": ev}
        if status == "other_headwear":
            v["headwear"] = "other_headwear"    # wears a cap / hat / non-work helmet
        if frame is not None and self.evidence_dir is not None:
            v["evidence"] = self._save_crop(frame, p["box"], v["id"], tid)
        self.violations.append(v)
        return v

    def _same_worker_as(self, tid: int, box, zone: str, rule: str, t: float, visible_now: set,
                        persons: list[dict]):
        """Is this new violation the same worker as an earlier one (same zone, rule)?
        Returns (violation, reason) or (None, None).
          same_footprint: that worker is visible now, and this box is a second,
                          nested box on him (same ground line)
          id_switch:      that worker vanished < reid_window_s ago at the same place
                          and the tracker gave him a new ID"""
        boxes_now = {p["track_id"]: p["box"] for p in persons}
        for v in reversed(self.violations):
            if v["zone"] != zone or v["rule"] != rule or v["track_id"] == tid:
                continue                    # same track coming back = a new incident
            prev = self.states.get(v["track_id"])
            if prev is None or prev.last_box is None:
                continue
            if v["track_id"] in visible_now:
                if same_footprint(box, boxes_now[v["track_id"]]):
                    return v, "same_footprint"
                continue                    # visible elsewhere: a different person
            if t - prev.last_t <= self.cfg.reid_window_s and iou(box, prev.last_box) >= self.cfg.reid_iou:
                return v, "id_switch"
        return None, None

    def _save_crop(self, frame, box, vid: int, tid: int) -> str:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = 0.25 * (x2 - x1), 0.15 * (y2 - y1)
        x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        x2, y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
        name = f"violation_{vid:03d}_track{tid}.jpg"
        cv2.imwrite(str(self.evidence_dir / name), frame[y1:y2, x1:x2])
        return name

    def stats(self) -> dict:
        states = list(self.states.values())
        out = {"violations": len(self.violations),
               "violations_by_rule": {r: sum(v["rule"] == r for v in self.violations)
                                      for r in sorted({v["rule"] for v in self.violations})},
               "duplicate_alerts_suppressed": len(self.duplicates),
               "persons_tracked": len(states),
               "persons_by_final_status": {k: sum(s.status == k for s in states)
                                           for k in ("helmet", "no_helmet", "other_headwear",
                                                     "unknown")},
               "rule_zones": [z.name for z in self.zones]}
        if self.cfg.check_vest:
            out["persons_by_vest_status"] = {k: sum(s.vest_status == k for s in states)
                                             for k in ("vest", "no_vest", "unknown")}
        if self.cfg.check_mask:
            out["persons_by_mask_status"] = {k: sum(s.mask_status == k for s in states)
                                             for k in ("mask", "no_mask", "unknown")}
        return out
