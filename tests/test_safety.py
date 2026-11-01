"""Helmet-compliance rules on synthetic tracks (no model needed)."""

import json

import numpy as np

from epa.agent import SceneAgent
from epa.safety import SafetyConfig, SafetyMonitor, associate, associate_vests, same_footprint
from epa.scene_store import SceneStore
from epa.zones import Zone

FPS = 10
# zone covering the bottom half of a 200x200 frame
ZONE = np.array([[0, 100], [200, 100], [200, 200], [0, 200]], dtype=np.float32)


def person(tid, x, y_feet):
    """Person 20 px wide, 60 px tall, feet at (x, y_feet)."""
    return {"track_id": tid, "cls_name": "person", "score": 0.9,
            "box": [x - 10, y_feet - 60, x + 10, y_feet]}


def ppe(cls, x, y_feet, tid=100):
    """Helmet / bare-head box on the head of a person at (x, y_feet)."""
    top = y_feet - 60
    return {"track_id": tid, "cls_name": cls, "score": 0.8,
            "box": [x - 6, top - 2, x + 6, top + 10]}


def make(zone_requires=("helmet",), **cfg):
    z = Zone("site", ZONE, classes={"person"}, requires=set(zone_requires))
    return z, SafetyMonitor([z], SafetyConfig(**cfg))


def step(z, mon, i, tracks):
    t = i / FPS
    z.update(i, t, tracks)
    return mon.update(i, t, tracks)


# ---------------------------------------------------------------- association
def test_helmet_goes_to_the_right_person():
    a, b = person(1, 50, 150), person(2, 120, 150)
    got = associate([a, b], [ppe("no_helmet", 120, 150), ppe("helmet", 50, 150)], SafetyConfig())
    assert got == {1: "helmet", 2: "no_helmet"}


def test_optional_centring_constraint():
    # helmet at the right edge of the head region, far from the person's centre line
    pers = [{"track_id": 1, "cls_name": "person", "score": 0.9, "box": [0, 0, 100, 300]}]
    off = [{"track_id": 9, "cls_name": "helmet", "score": 0.9, "box": [95, 10, 105, 20]}]
    assert associate(pers, off, SafetyConfig()) == {1: "helmet"}
    assert associate(pers, off, SafetyConfig(max_center_dx=0.35)) == {}


def test_box_far_from_any_head_is_ignored():
    got = associate([person(1, 50, 150)], [ppe("helmet", 50, 210)], SafetyConfig())
    assert got == {}


# ---------------------------------------------------------------------- rules
def test_violation_after_min_duration_inside_zone():
    z, mon = make()
    raised = []
    for i in range(30):                                # 3 s inside, bare head
        raised += step(z, mon, i, [person(1, 50, 150), ppe("no_helmet", 50, 150)])
    assert len(raised) == 1                            # alerted once, not every frame
    v = raised[0]
    assert v["track_id"] == 1 and v["zone"] == "site"
    # status needs 3 observations (t=0.2), then 1 s inside without helmet
    assert v["t"] - v["t_start"] >= 1.0 and v["t"] <= 1.4


def test_no_violation_with_helmet_or_outside_zone():
    z, mon = make()
    for i in range(30):
        step(z, mon, i, [person(1, 50, 150), ppe("helmet", 50, 150),     # inside, helmet
                         person(2, 150, 80), ppe("no_helmet", 150, 80)])  # bare head, outside
    assert mon.violations == []


def test_unknown_head_never_alerts():
    z, mon = make()
    for i in range(30):
        step(z, mon, i, [person(1, 50, 150)])          # no head evidence at all
    assert mon.violations == []
    assert mon.states[1].status == "unknown"


def test_single_noisy_frame_does_not_flip_status():
    z, mon = make()
    for i in range(30):
        cls = "no_helmet" if i == 15 else "helmet"     # one wrong detection
        step(z, mon, i, [person(1, 50, 150), ppe(cls, 50, 150)])
    assert mon.violations == [] and mon.states[1].status == "helmet"


def test_rearms_after_leaving_zone():
    z, mon = make()
    for i in range(70):
        y = 150 if (i < 25 or i >= 45) else 60         # in, out for 2 s, back in
        step(z, mon, i, [person(1, 50, y), ppe("no_helmet", 50, y)])
    assert len(mon.violations) == 2


def test_brief_helmet_flicker_does_not_rearm():
    # walking past helmeted workers, a bare-headed worker is judged "helmet" for 0.6 s
    z, mon = make()
    for i in range(60):
        cls = "helmet" if 25 <= i < 31 else "no_helmet"
        step(z, mon, i, [person(1, 50, 150), ppe(cls, 50, 150)])
    assert len(mon.violations) == 1


def test_helmet_put_on_for_real_then_removed_alerts_again():
    z, mon = make()
    for i in range(90):
        cls = "helmet" if 20 <= i < 50 else "no_helmet"     # 3 s with a helmet, then off again
        step(z, mon, i, [person(1, 50, 150), ppe(cls, 50, 150)])
    assert len(mon.violations) == 2


def test_zone_without_rule_is_ignored():
    z, mon = make(zone_requires=())
    for i in range(30):
        step(z, mon, i, [person(1, 50, 150), ppe("no_helmet", 50, 150)])
    assert mon.zones == [] and mon.violations == []


# ----------------------------------------------------- inference from absence
def tall(tid, x, y_feet, h=150):
    return {"track_id": tid, "cls_name": "person", "score": 0.9,
            "box": [x - 25, y_feet - h, x + 25, y_feet]}


def test_inferred_violation_when_head_clearly_visible():
    z, mon = make()
    for i in range(30):
        step(z, mon, i, [tall(1, 60, 190)])            # tall, head visible, no helmet found
    assert len(mon.violations) == 1
    assert mon.violations[0]["evidence_type"] == "inferred_from_absence"


def test_no_inference_when_head_cut_by_border():
    z, mon = make()
    for i in range(30):
        step(z, mon, i, [tall(1, 60, 190, h=190)])     # box starts at y=0
    assert mon.violations == []


def test_no_inference_when_head_hidden_behind_closer_person():
    z, mon = make()
    for i in range(30):
        back = tall(1, 60, 150, h=110)                  # farther away: feet higher in the image
        front = tall(2, 70, 198, h=150)                 # nearer: its body covers #1's head
        step(z, mon, i, [back, front])                  # no PPE boxes at all
    assert mon.states[1].status == "unknown"            # hidden head: never inferred
    assert all(v["track_id"] != 1 for v in mon.violations)


def test_duplicate_person_box_sharing_one_helmet_is_not_flagged():
    # the detector sometimes outputs two overlapping boxes for one worker;
    # the helmet can be matched to only one of them
    z, mon = make()
    for i in range(30):
        a = tall(1, 60, 190)
        b = {"track_id": 2, "cls_name": "person", "score": 0.6, "box": [30, 35, 92, 192]}
        helmet = {"track_id": 50, "cls_name": "helmet", "score": 0.9, "box": [52, 38, 68, 50]}
        step(z, mon, i, [a, b, helmet])
    assert mon.violations == []


def test_no_inference_when_person_is_small_for_the_detector():
    # 150 px tall in the frame, but the detector sees the frame at 1/4 scale: ~37 px
    z = Zone("site", ZONE, classes={"person"}, requires={"helmet"})
    mon = SafetyMonitor([z], SafetyConfig(input_scale=0.25))
    for i in range(30):
        step(z, mon, i, [tall(1, 60, 190)])
    assert mon.violations == [] and mon.states[1].status == "unknown"


def test_inference_can_be_disabled():
    z, mon = make(infer_from_absence=False)
    for i in range(30):
        step(z, mon, i, [tall(1, 60, 190)])
    assert mon.violations == []


# ------------------------------------------------- tracker ID switch (dedupe)
def test_id_switch_does_not_raise_a_second_alert():
    z, mon = make()
    for i in range(50):
        tid = 1 if i < 25 else 2                       # tracker loses #1, same worker becomes #2
        if i in (25, 26):
            continue                                   # 2 frames with no detection in between
        step(z, mon, i, [tall(tid, 60, 190)])
    assert len(mon.violations) == 1
    assert len(mon.duplicates) == 1 and mon.duplicates[0]["same_as_violation"] == 1


def test_new_worker_elsewhere_still_alerts():
    z, mon = make()
    for i in range(50):
        tracks = [tall(1, 40, 190)] if i < 25 else [tall(2, 160, 190)]   # different place
        step(z, mon, i, tracks)
    assert len(mon.violations) == 2 and mon.duplicates == []


def test_two_workers_side_by_side_both_alert():
    z, mon = make()
    for i in range(30):
        step(z, mon, i, [tall(1, 40, 190), tall(2, 100, 190)])          # both visible
    assert len(mon.violations) == 2


# ------------------------ nested duplicate box (real case from the GPU run)
FRAME = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=np.float32)


def _p(tid, box):
    return {"track_id": tid, "cls_name": "person", "score": 0.8, "box": box}


def _run_nested(order_new_first: bool):
    z = Zone("site", FRAME, classes={"person"}, requires={"helmet"})
    mon = SafetyMonitor([z], SafetyConfig())
    for i in range(50):
        tracks = [_p(25, [30, 84, 703, 524])]               # the worker, whole body
        if i >= 20:
            tracks.append(_p(30, [0, 75, 385, 520]))         # 2nd box on his left half
            tracks.append({"track_id": 90, "cls_name": "no_helmet", "score": 0.6,
                           "box": [175, 82, 205, 108]})      # bare head detected in it
            if order_new_first:
                tracks.reverse()
        step(z, mon, i, tracks)
    return mon


def test_nested_box_with_same_footprint_is_one_worker():
    for order_new_first in (False, True):                  # result must not depend on order
        mon = _run_nested(order_new_first)
        assert len(mon.violations) == 1, order_new_first
        assert mon.violations[0]["track_id"] == 25
        assert len(mon.duplicates) == 1 and mon.duplicates[0]["reason"] == "same_footprint"


def test_same_footprint_rule():
    worker = [30, 84, 703, 524]
    assert same_footprint([0, 75, 385, 520], worker)        # nested, feet on the same line
    assert not same_footprint([150, 120, 300, 420], [100, 100, 400, 580])  # farther person
    assert not same_footprint([0, 75, 100, 520], [600, 80, 700, 520])      # side by side


# ------------------------------------------------------------- agent + store
class VisionStub:
    def __init__(self):
        self.seen = []

    def describe_image(self, path, q):
        self.seen.append(path)
        return "no - bare head, no hard hat visible"


def test_agent_safety_tools(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "violation_001_track4.jpg").write_bytes(b"\xff\xd8x")
    (tmp_path / "summary.json").write_text(json.dumps({
        "zones": [{"zone": "site"}], "safety": {"violations": 1}}))
    (tmp_path / "violations.jsonl").write_text(json.dumps(
        {"id": 1, "frame": 30, "t": 3.0, "t_start": 2.0, "zone": "site", "track_id": 4,
         "rule": "helmet_required", "evidence": "violation_001_track4.jpg"}))
    (tmp_path / "tracks.jsonl").write_text(json.dumps(
        {"frame": 30, "t": 3.0, "occupancy": {"site": 1},
         "tracks": [{"id": 4, "class": "person", "box": [0, 0, 1, 1], "ppe": "no_helmet"},
                    {"id": 9, "class": "helmet", "box": [0, 0, 1, 1]}]}))

    store = SceneStore(tmp_path)
    assert store.list_violations()["count"] == 1
    assert store.list_violations(t_start=5)["count"] == 0
    assert store.ppe_status_at(3.0)["counts"] == {"no_helmet": 1}

    stub = VisionStub()
    agent = SceneAgent(store, stub)
    r = agent.execute("inspect_violation", {"violation_id": 1})
    assert r["answer"].startswith("no") and stub.seen[0].endswith("violation_001_track4.jpg")
    assert "error" in agent.execute("inspect_violation", {"violation_id": 7})


def test_safety_tools_on_a_normal_run(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"zones": []}))
    assert "error" in SceneStore(tmp_path).list_violations()


def test_small_helmet_of_a_farther_worker_is_not_given_to_a_near_one():
    near = {"track_id": 1, "cls_name": "person", "score": 0.9, "box": [500, 250, 800, 700]}
    behind = {"track_id": 7, "cls_name": "helmet", "score": 0.7, "box": [600, 330, 640, 360]}
    assert associate([near], [behind], SafetyConfig()) == {1: "helmet"}          # old behaviour
    assert associate([near], [behind], SafetyConfig(min_ppe_width=0.25)) == {}
    own = {"track_id": 8, "cls_name": "helmet", "score": 0.7, "box": [600, 210, 720, 290]}
    assert associate([near], [own], SafetyConfig(min_ppe_width=0.25)) == {1: "helmet"}


# ----------------------------------------------------------------------- vest
def vest_on(p, tid=300):
    """Vest box on the torso of person dict p."""
    x1, y1, x2, y2 = p["box"]
    h = y2 - y1
    return {"track_id": tid, "cls_name": "vest", "score": 0.8,
            "box": [x1 + 2, y1 + 0.25 * h, x2 - 2, y1 + 0.6 * h]}


def helmet_on(p, tid=200):
    x1, y1, x2, y2 = p["box"]
    cx = (x1 + x2) / 2
    return {"track_id": tid, "cls_name": "helmet", "score": 0.8, "box": [cx - 12, y1 - 3, cx + 12, y1 + 15]}


def make_vest(requires=("helmet", "vest")):
    z = Zone("site", ZONE, classes={"person"}, requires=set(requires))
    return z, SafetyMonitor([z], SafetyConfig())


def test_no_vest_alerts_with_its_own_rule():
    z, mon = make_vest()
    for i in range(30):
        p = tall(1, 60, 190)
        step(z, mon, i, [p, helmet_on(p)])                # helmet on, no vest
    assert [v["rule"] for v in mon.violations] == ["vest_required"]
    assert mon.violations[0]["missing"] == "vest"
    assert mon.states[1].status == "helmet" and mon.states[1].vest_status == "no_vest"


def test_helmet_and_vest_worn_no_alert():
    z, mon = make_vest()
    for i in range(30):
        p = tall(1, 60, 190)
        step(z, mon, i, [p, helmet_on(p), vest_on(p)])
    assert mon.violations == [] and mon.states[1].vest_status == "vest"


def test_missing_both_gives_two_violations():
    z, mon = make_vest()
    for i in range(30):
        step(z, mon, i, [tall(1, 60, 190)])
    assert sorted(v["rule"] for v in mon.violations) == ["helmet_required", "vest_required"]


def test_vest_not_checked_when_zone_only_requires_helmet():
    z, mon = make_vest(requires=("helmet",))
    for i in range(30):
        p = tall(1, 60, 190)
        step(z, mon, i, [p, helmet_on(p)])
    assert mon.violations == [] and not mon.cfg.check_vest


def test_small_vest_of_worker_behind_is_not_given_to_near_worker():
    near = {"track_id": 1, "cls_name": "person", "score": 0.9, "box": [0, 0, 100, 300]}
    behind_vest = {"track_id": 9, "cls_name": "vest", "score": 0.9, "box": [40, 100, 60, 130]}
    assert associate_vests([near], [behind_vest], SafetyConfig()) == set()
    own = {"track_id": 8, "cls_name": "vest", "score": 0.9, "box": [10, 70, 90, 180]}
    assert associate_vests([near], [own], SafetyConfig()) == {1}


def test_no_vest_inference_when_torso_hidden():
    z, mon = make_vest(requires=("vest",))
    for i in range(30):
        back = tall(1, 60, 150, h=110)
        front = tall(2, 65, 198, h=150)
        step(z, mon, i, [back, front, vest_on(front)])
    assert mon.states[1].vest_status == "unknown"
    assert all(v["track_id"] != 1 for v in mon.violations)


def test_flickering_vest_detection_is_still_a_vest():
    # real case: the detector finds the vest in only ~40 % of the frames
    z, mon = make_vest(requires=("vest",))
    for i in range(60):
        p = tall(1, 60, 190)
        step(z, mon, i, [p] + ([vest_on(p)] if i % 5 in (0, 2) else []))
    assert mon.violations == [] and mon.states[1].vest_status == "vest"


# ------------------------------------------- non-work headwear, no_vest, mask
def box_on_head(p, cls, tid=400):
    b = helmet_on(p)
    return dict(b, cls_name=cls, track_id=tid)


def test_cap_is_not_a_hard_hat():
    z, mon = make()
    for i in range(30):
        p = tall(1, 60, 190)
        step(z, mon, i, [p, box_on_head(p, "other_headwear")])
    assert mon.states[1].status == "other_headwear"
    v, = mon.violations
    assert v["rule"] == "helmet_required" and v["headwear"] == "other_headwear"
    assert v["evidence_type"] == "detected_other_headwear"


def test_hard_hat_mistaken_once_for_a_cap_is_still_a_hard_hat():
    z, mon = make()
    for i in range(30):
        p = tall(1, 60, 190)
        step(z, mon, i, [p, box_on_head(p, "other_headwear" if i == 10 else "helmet")])
    assert mon.violations == [] and mon.states[1].status == "helmet"


def test_detected_no_vest_needs_only_a_majority():
    z, mon = make_vest(requires=("vest",))
    for i in range(30):
        p = tall(1, 60, 190)
        # a no_vest box in 2 frames out of 3, nothing in the others
        extra = [dict(vest_on(p), cls_name="no_vest")] if i % 3 else []
        step(z, mon, i, [p] + extra)
    v, = mon.violations
    assert v["rule"] == "vest_required" and v["evidence_type"] == "detected"


def test_mask_rule_uses_detected_evidence_only():
    z = Zone("site", ZONE, classes={"person"}, requires={"mask"})
    mon = SafetyMonitor([z], SafetyConfig())
    for i in range(30):
        a, b = tall(1, 40, 190), tall(2, 120, 190)
        step(z, mon, i, [a, b, box_on_head(a, "no_mask", 401)])   # b: no mask evidence at all
    assert [v["track_id"] for v in mon.violations] == [1]
    assert mon.states[2].mask_status == "unknown"


def test_labels():
    from epa.safety import ppe_summary
    assert ppe_summary({"ppe": "helmet"}) == ("helmet", "ok")
    assert ppe_summary({"ppe": "unknown"}) == ("unknown", "unknown")
    assert ppe_summary({"ppe": "other_headwear", "vest": "vest"}) == ("not a hard hat | vest", "bad")
    assert ppe_summary({"ppe": "helmet", "vest": "unknown"}) == ("helmet | ? vest", "unknown")


# ------------------------------------------------------- single-image helpers
def test_nested_upper_body_box_is_dropped_but_person_behind_is_kept():
    from epa.safety import dedupe_persons
    full = {"cls_name": "person", "score": 0.9, "box": [100, 100, 200, 400]}
    upper = {"cls_name": "person", "score": 0.6, "box": [105, 102, 195, 250]}   # same head
    behind = {"cls_name": "person", "score": 0.8, "box": [120, 150, 180, 330]}  # inside, no edge
    kept = dedupe_persons([upper, full, behind])
    assert full in kept and behind in kept and upper not in kept


def test_short_labels():
    from epa.safety import short_label
    assert short_label({"ppe": "helmet", "vest": "vest"}) == "OK"
    assert short_label({"ppe": "helmet", "vest": "no_vest"}) == "no vest"
    assert short_label({"ppe": "other_headwear", "vest": "unknown"}) == "not hard hat"
    assert short_label({"ppe": "helmet", "vest": "unknown"}) == "? vest"
    assert short_label({"ppe": "unknown", "vest": "unknown"}) == "?"


def test_no_vest_inference_on_legs_only_box_cut_by_top_border():
    # real case: a person seen from the knees down (box starts at y=0) was called "no vest"
    from epa.safety import frame_status
    legs = {"cls_name": "person", "score": 0.8, "box": [330, 0, 640, 580]}
    assert frame_status([legs], SafetyConfig(check_vest=True))[0]["vest"] == "unknown"
    whole = {"cls_name": "person", "score": 0.8, "box": [330, 20, 480, 580]}
    assert frame_status([whole], SafetyConfig(check_vest=True))[0]["vest"] == "no_vest"


def test_mask_found_on_upper_body_close_up():
    # real case: selfie, person box = head + chest, mask centre at ~45 % of the box height
    from epa.safety import frame_status
    person = {"cls_name": "person", "score": 0.9, "box": [0, 90, 640, 640]}
    mask = {"cls_name": "mask", "score": 0.8, "box": [180, 260, 515, 410]}
    p, = frame_status([person, mask], SafetyConfig(check_mask=True))
    assert p["mask"] == "mask"
    # full body: the face region stays at the top
    body = {"cls_name": "person", "score": 0.9, "box": [100, 0, 200, 400]}
    low = {"cls_name": "mask", "score": 0.8, "box": [130, 200, 170, 230]}       # at the waist
    p, = frame_status([body, low], SafetyConfig(check_mask=True))
    assert p["mask"] == "unknown"


def test_hard_hat_on_close_up_head_is_found():
    # real case: close-up, box 425x492, hard hat centred at ~40 % of the box height
    from epa.safety import frame_status
    p = {"cls_name": "person", "score": 0.9, "box": [213, 58, 638, 550]}
    hat = {"cls_name": "helmet", "score": 0.8, "box": [210, 40, 380, 460]}
    assert frame_status([p, hat], SafetyConfig())[0]["ppe"] == "helmet"
