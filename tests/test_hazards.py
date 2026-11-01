"""Machine proximity and cone-area rules on synthetic tracks."""

import numpy as np

from epa.hazards import HazardConfig, HazardMonitor, frame_hazards, ground_gap, is_operator

FPS = 10
FRAME = np.zeros((600, 800, 3), dtype=np.uint8)
MACHINE = [400, 300, 600, 450]            # footprint: y 375-450


def person(tid, x, y_feet, h=150):
    return {"track_id": tid, "cls_name": "person", "score": 0.9,
            "box": [x - 25, y_feet - h, x + 25, y_feet]}


def machine(tid=50, box=MACHINE, cls="machinery"):
    return {"track_id": tid, "cls_name": cls, "score": 0.9, "box": list(box)}


def cone(tid, x, y):
    return {"track_id": tid, "cls_name": "safety_cone", "score": 0.8, "box": [x - 8, y - 20, x + 8, y]}


def run(frames, **cfg):
    mon = HazardMonitor(HazardConfig(**cfg))
    for i, tracks in enumerate(frames):
        mon.update(i, i / FPS, tracks, FRAME)
    return mon


# ------------------------------------------------------------------ geometry
def test_gap_on_the_ground():
    cfg = HazardConfig()
    assert ground_gap(person(1, 380, 440)["box"], MACHINE, cfg) < 0.2   # beside it
    assert ground_gap(person(1, 200, 440)["box"], MACHINE, cfg) > 1.0   # 175 px away, 150 px tall
    # behind the machine: overlaps its box in the image, but feet are far above its footprint
    far_behind = person(1, 500, 310, h=60)
    assert ground_gap(far_behind["box"], MACHINE, cfg) > 1.0


def test_operator_in_cab_is_not_near():
    cfg = HazardConfig()
    cab = person(1, 450, 360, h=50)                 # inside the box, feet above the footprint
    assert is_operator(cab["box"], MACHINE, cfg)
    out = frame_hazards([cab, machine()], FRAME.shape)
    assert out["near"] == {}


# ------------------------------------------------------------------ proximity
def test_worker_next_to_machine_alerts_once():
    mon = run([[person(1, 380, 440), machine()] for _ in range(40)])
    v, = mon.violations
    assert v["rule"] == "machine_proximity" and v["machine_id"] == 50
    assert v["machine_moving"] is False and v["gap_person_heights"] < 0.2


def test_far_worker_no_alert():
    mon = run([[person(1, 150, 440), machine()] for _ in range(40)])
    assert mon.violations == []


def test_moving_machine_detected_and_only_moving_mode():
    frames = [[person(1, 380, 440), machine(box=[400 + 3 * i, 300, 600 + 3 * i, 450])]
              for i in range(40)]                     # 30 px/s = 0.15 widths/s
    mon = run(frames)
    assert mon.violations[0]["machine_moving"] is True
    still = [[person(1, 380, 440), machine()] for _ in range(40)]
    assert run(still, only_moving=True).violations == []


def test_short_pass_no_alert_and_rearm_after_clear():
    near, far = [person(1, 380, 440), machine()], [person(1, 100, 440), machine()]
    assert run([near] * 3 + [far] * 30).violations == []            # 0.3 s only
    mon = run([near] * 10 + [far] * 25 + [near] * 10)               # away 2.5 s, back
    assert len(mon.violations) == 2


# -------------------------------------------------------------------- cones
CONES = [cone(60, 100, 300), cone(61, 300, 300), cone(62, 300, 500), cone(63, 100, 500)]


def test_person_inside_cone_area_alerts():
    mon = run([CONES + [person(1, 200, 420)] for _ in range(20)])
    v, = mon.violations
    assert v["rule"] == "cone_area_entry" and v["class"] == "person"


def test_person_outside_or_cones_in_a_line():
    assert run([CONES + [person(1, 600, 420)] for _ in range(20)]).violations == []
    line = [cone(60, 100, 300), cone(61, 200, 301), cone(62, 300, 302)]
    assert run([line + [person(1, 200, 301)] for _ in range(20)]).violations == []


def test_flickering_cones_are_remembered():
    frames = [(CONES if i % 4 == 0 else CONES[:1]) + [person(1, 200, 420)] for i in range(20)]
    assert len(run(frames).violations) == 1


def test_vehicle_keep_out_option():
    car = machine(70, [170, 380, 230, 430], cls="vehicle")
    mon = run([CONES + [car] for _ in range(20)], cone_area_keep_out=("vehicle",))
    assert [v["class"] for v in mon.violations] == ["vehicle"]


def test_agent_sees_hazard_violations(tmp_path):
    import json
    from epa.scene_store import SceneStore
    (tmp_path / "summary.json").write_text(json.dumps({"zones": [], "hazards": {}}))
    (tmp_path / "violations.jsonl").write_text(json.dumps(
        {"id": 1, "t": 2.0, "t_start": 1.5, "zone": "site", "track_id": 3,
         "rule": "machine_proximity", "machine_id": 9, "machine_class": "machinery",
         "machine_moving": True, "gap_person_heights": 0.4, "evidence_type": "detected"}))
    v, = SceneStore(tmp_path).list_violations()["violations"]
    assert v["rule"] == "machine_proximity" and v["machine_moving"] is True


def test_close_up_with_feet_out_of_frame_is_not_near_a_machine():
    # real case: a close-up worker (box down to the image bottom) and a small machine far behind
    worker = {"cls_name": "person", "score": 0.9, "box": [30, 30, 470, 595]}
    far = {"cls_name": "machinery", "score": 0.6, "box": [520, 440, 580, 495]}
    assert frame_hazards([worker, far], FRAME.shape)["near"] == {}


def test_small_far_machine_behind_a_near_worker_is_not_near():
    # the same real case, box ending 15 px above the bottom: feet "visible", depth disagrees
    worker = {"cls_name": "person", "score": 0.9, "box": [30, 30, 470, 585]}
    far = {"cls_name": "machinery", "score": 0.6, "box": [520, 440, 580, 495]}
    assert frame_hazards([worker, far], FRAME.shape)["near"] == {}


def test_seated_operator_in_tall_machine_cab():
    # real case (telehandler): the operator box (upper body, 55 px) ends below the top of
    # the footprint band but ~3 person heights above the machine's ground line
    op = {"cls_name": "person", "score": 0.7, "box": [240, 375, 265, 430]}
    tele = {"cls_name": "machinery", "score": 0.9, "box": [50, 250, 500, 585]}
    assert frame_hazards([op, tele], FRAME.shape)["near"] == {}
    walker = {"cls_name": "person", "score": 0.9, "box": [300, 430, 350, 580]}   # on the ground
    assert 0 in frame_hazards([walker, tele], FRAME.shape)["near"]


def test_operator_in_excavator_cab():
    # real case: operator box ends ~0.7 of his height above the machine's ground line
    op = {"cls_name": "person", "score": 0.7, "box": [500, 330, 540, 447]}
    exc = {"cls_name": "machinery", "score": 0.9, "box": [275, 98, 638, 530]}
    assert frame_hazards([op, exc], FRAME.shape)["near"] == {}
