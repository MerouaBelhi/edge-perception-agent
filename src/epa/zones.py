"""Polygon zones and entry/exit event logic.

A track "enters" a zone when its anchor point (bottom-centre of the box,
i.e. where an object touches the ground) moves from outside to inside the
polygon, and "exits" on the opposite transition. Events are kept as plain
dicts so they can be serialised to JSON and queried later (e.g. by an agent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Ray-casting test. `polygon` is an (N, 2) array of vertices."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def anchor_point(box_xyxy) -> tuple[float, float]:
    """Bottom-centre of a box: the object's footprint on the ground plane."""
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, y2


@dataclass
class Zone:
    name: str
    polygon: np.ndarray                      # (N, 2) in pixel coordinates
    classes: set[str] | None = None          # None = all classes
    requires: set[str] = field(default_factory=set)   # PPE rules, e.g. {"helmet"}
    _inside: dict[int, bool] = field(default_factory=dict, repr=False)
    occupancy: int = 0
    entries: int = 0
    exits: int = 0

    def update(self, frame_idx: int, t_sec: float, tracks: list[dict]) -> list[dict]:
        """Update zone state with the tracks of one frame; return new events."""
        events = []
        seen_inside = 0
        for tr in tracks:
            if self.classes is not None and tr["cls_name"] not in self.classes:
                continue
            tid = tr["track_id"]
            x, y = anchor_point(tr["box"])
            now_in = point_in_polygon(x, y, self.polygon)
            was_in = self._inside.get(tid)
            if now_in:
                seen_inside += 1
            if was_in is not None and now_in != was_in:
                kind = "enter" if now_in else "exit"
                if now_in:
                    self.entries += 1
                else:
                    self.exits += 1
                events.append({
                    "frame": frame_idx,
                    "t": round(t_sec, 3),
                    "zone": self.name,
                    "event": kind,
                    "track_id": tid,
                    "class": tr["cls_name"],
                })
            self._inside[tid] = now_in
        self.occupancy = seen_inside
        return events

    def contains(self, track_id: int) -> bool:
        """Whether the track was inside the zone at the last update."""
        return self._inside.get(track_id, False)

    def stats(self) -> dict:
        return {"zone": self.name, "occupancy": self.occupancy,
                "entries": self.entries, "exits": self.exits}


def load_zones(path: str | Path, frame_w: int, frame_h: int) -> list[Zone]:
    """Load zones from JSON. Coordinates may be absolute pixels or normalised
    to [0, 1] (set "normalized": true), so one config works at any resolution."""
    cfg = json.loads(Path(path).read_text())
    zones = []
    for z in cfg["zones"]:
        poly = np.asarray(z["polygon"], dtype=np.float32)
        # all coordinates in [0, 1] and no explicit flag: clearly normalised (a
        # 1-pixel zone is never intended), so do not silently monitor nothing
        norm = z.get("normalized", bool(poly.size) and float(poly.max()) <= 1.0)
        if norm:
            poly = poly * np.array([frame_w, frame_h], dtype=np.float32)
        classes = set(z["classes"]) if z.get("classes") else None
        zones.append(Zone(name=z["name"], polygon=poly, classes=classes,
                          requires=set(z.get("requires") or [])))
    return zones
