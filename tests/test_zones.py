import numpy as np

from epa.zones import Zone, anchor_point, point_in_polygon

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)


def test_point_in_polygon():
    assert point_in_polygon(5, 5, SQUARE)
    assert not point_in_polygon(15, 5, SQUARE)
    assert not point_in_polygon(-1, -1, SQUARE)


def test_anchor_is_bottom_centre():
    assert anchor_point([0, 0, 4, 10]) == (2.0, 10)


def _track(tid, x, y, cls="person"):
    # box whose bottom-centre is (x, y)
    return {"track_id": tid, "box": [x - 1, y - 5, x + 1, y], "cls_name": cls, "score": 0.9}


def test_enter_and_exit_events():
    z = Zone("z", SQUARE)
    assert z.update(0, 0.0, [_track(1, 20, 5)]) == []          # first sighting: no event
    ev = z.update(1, 0.1, [_track(1, 5, 5)])
    assert [e["event"] for e in ev] == ["enter"] and z.occupancy == 1
    ev = z.update(2, 0.2, [_track(1, 20, 5)])
    assert [e["event"] for e in ev] == ["exit"] and z.occupancy == 0
    assert (z.entries, z.exits) == (1, 1)


def test_class_filter():
    z = Zone("people_only", SQUARE, classes={"person"})
    z.update(0, 0.0, [_track(1, 20, 5, "car")])
    assert z.update(1, 0.1, [_track(1, 5, 5, "car")]) == []
    assert z.entries == 0


def test_normalised_coordinates_detected_without_flag(tmp_path):
    import json
    from epa.zones import load_zones
    p = tmp_path / "z.json"
    p.write_text(json.dumps({"zones": [{"name": "a", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}]}))
    z = load_zones(p, 1280, 720)[0]
    assert z.polygon.max() == 1280 and z.polygon[0][1] == 360
