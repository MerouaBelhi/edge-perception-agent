"""Read-only query layer over a pipeline run (the agent's tools).

A run directory produced by `epa.pipeline` contains summary.json, events.jsonl,
tracks.jsonl and keyframes/. Every method here returns plain JSON-serialisable
data: the language model never sees raw files, only these answers, which keeps
its context small and its numbers grounded in what the tracker measured.
"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SceneStore:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        summary_path = self.dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary.json in {self.dir} — run epa.pipeline first")
        self.summary = json.loads(summary_path.read_text())
        self.events = _read_jsonl(self.dir / "events.jsonl")
        self.frames = _read_jsonl(self.dir / "tracks.jsonl")
        self._frame_times = [f["t"] for f in self.frames]
        kf_index = self.dir / "keyframes" / "index.json"
        self.keyframes = json.loads(kf_index.read_text()) if kf_index.exists() else []
        self.zone_names = [z["zone"] for z in self.summary.get("zones", [])]
        self.violations = _read_jsonl(self.dir / "violations.jsonl")
        self.has_safety = "safety" in self.summary or "hazards" in self.summary

    # ---------------------------------------------------------------- helpers
    def _nearest_frame(self, t: float) -> dict | None:
        if not self.frames:
            return None
        i = bisect.bisect_left(self._frame_times, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.frames)]
        return self.frames[min(cands, key=lambda j: abs(self._frame_times[j] - t))]

    def _check_zone(self, zone: str | None):
        if zone is not None and zone not in self.zone_names:
            return {"error": f"unknown zone '{zone}'", "available_zones": self.zone_names}
        return None

    # ------------------------------------------------------------------ tools
    def get_video_summary(self) -> dict:
        s = self.summary
        return {
            "duration_s": s.get("duration_s"),
            "fps": s.get("fps_source"),
            "frames": s.get("frames"),
            "zones": s.get("zone_definitions") or [{"name": n} for n in self.zone_names],
            "zone_totals": s.get("zones"),
            "unique_tracks_per_class": s.get("tracks_per_class"),
            "total_events": s.get("events"),
            "safety": s.get("safety"),       # None when run without --safety
        }

    def count_events(self, zone: str | None = None, event: str | None = None,
                     object_class: str | None = None, t_start: float | None = None,
                     t_end: float | None = None, max_listed: int = 20) -> dict:
        err = self._check_zone(zone)
        if err:
            return err
        if event is not None and event not in ("enter", "exit"):
            return {"error": "event must be 'enter' or 'exit'"}
        sel = [e for e in self.events
               if (zone is None or e["zone"] == zone)
               and (event is None or e["event"] == event)
               and (object_class is None or e["class"] == object_class)
               and (t_start is None or e["t"] >= t_start)
               and (t_end is None or e["t"] <= t_end)]
        return {"count": len(sel),
                "unique_tracks": len({e["track_id"] for e in sel}),
                "by_zone_and_type": dict(Counter(f'{e["zone"]}:{e["event"]}' for e in sel)),
                "events": sel[:max_listed],
                "truncated": len(sel) > max_listed}

    def zone_occupancy(self, zone: str, t: float | None = None) -> dict:
        err = self._check_zone(zone)
        if err:
            return err
        if not self.frames:
            return {"error": "no per-frame data (tracks.jsonl missing)"}
        if t is not None:
            f = self._nearest_frame(t)
            return {"zone": zone, "t": f["t"], "occupancy": f["occupancy"].get(zone, 0)}
        series = [(f["t"], f["occupancy"].get(zone, 0)) for f in self.frames]
        peak_t, peak = max(series, key=lambda x: x[1])
        mean = sum(v for _, v in series) / len(series)
        occupied = sum(1 for _, v in series if v > 0) / len(series)
        return {"zone": zone, "peak_occupancy": peak, "peak_at_t": peak_t,
                "mean_occupancy": round(mean, 2),
                "fraction_of_time_occupied": round(occupied, 3)}

    def objects_at(self, t: float) -> dict:
        f = self._nearest_frame(t)
        if f is None:
            return {"error": "no per-frame data (tracks.jsonl missing)"}
        return {"t": f["t"], "counts": dict(Counter(tr["class"] for tr in f["tracks"])),
                "tracks": [{"id": tr["id"], "class": tr["class"], "box": tr["box"]}
                           for tr in f["tracks"]],
                "zone_occupancy": f["occupancy"]}

    def track_history(self, track_id: int) -> dict:
        seen = [f["t"] for f in self.frames if any(tr["id"] == track_id for tr in f["tracks"])]
        if not seen:
            return {"error": f"track {track_id} not found"}
        cls = next(tr["class"] for f in self.frames for tr in f["tracks"] if tr["id"] == track_id)
        return {"track_id": track_id, "class": cls, "first_seen_t": seen[0],
                "last_seen_t": seen[-1], "visible_s": round(seen[-1] - seen[0], 2),
                "frames_visible": len(seen),
                "events": [e for e in self.events if e["track_id"] == track_id]}

    def list_violations(self, zone: str | None = None, t_start: float | None = None,
                        t_end: float | None = None) -> dict:
        if not self.has_safety:
            return {"error": "this run was processed without --safety / --hazards (no safety rules)"}
        err = self._check_zone(zone)
        if err:
            return err
        sel = [{k: v[k] for k in ("id", "t", "t_start", "zone", "track_id", "rule", "headwear", "evidence_type",
                          "machine_id", "machine_class", "machine_moving", "gap_person_heights",
                          "class")
                if k in v}
               for v in self.violations
               if (zone is None or v["zone"] == zone)
               and (t_start is None or v["t"] >= t_start)
               and (t_end is None or v["t"] <= t_end)]
        return {"count": len(sel), "unique_workers": len({v["track_id"] for v in sel}),
                "violations": sel}

    def ppe_status_at(self, t: float) -> dict:
        if not self.has_safety:
            return {"error": "this run was processed without --safety / --hazards (no safety rules)"}
        f = self._nearest_frame(t)
        if f is None:
            return {"error": "no per-frame data (tracks.jsonl missing)"}
        persons = [tr for tr in f["tracks"] if "ppe" in tr]
        out = {"t": f["t"],
               "counts": dict(Counter(tr["ppe"] for tr in persons)),
               "workers": [{"id": tr["id"], "ppe": tr["ppe"],
                            **{k: tr[k] for k in ("vest", "mask") if k in tr}} for tr in persons]}
        for k in ("vest", "mask"):
            if any(k in tr for tr in persons):
                out[f"{k}_counts"] = dict(Counter(tr[k] for tr in persons if k in tr))
        return out

    def violation_evidence(self, violation_id: int) -> dict:
        v = next((v for v in self.violations if v["id"] == violation_id), None)
        if v is None:
            return {"error": f"violation {violation_id} not found",
                    "available_ids": [x["id"] for x in self.violations]}
        if "evidence" not in v:
            return {"error": "no evidence image saved for this violation"}
        return {**v, "path": str(self.dir / "evidence" / v["evidence"])}

    def keyframe_near(self, t: float) -> dict | None:
        """Closest saved keyframe to time t (the VLM only sees saved frames)."""
        if not self.keyframes:
            return None
        kf = min(self.keyframes, key=lambda k: abs(k["t"] - t))
        return {**kf, "path": str(self.dir / "keyframes" / kf["file"])}
