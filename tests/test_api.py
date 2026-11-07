"""REST API with a fake detector (no model file needed)."""

import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")


class FakeDetector:
    class_names = ["person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask",
                   "other_headwear", "machinery", "vehicle", "safety_cone"]
    input_size, conf_thres = 960, 0.25
    last_timing = {"pre_ms": 1.0, "infer_ms": 2.0, "post_ms": 0.1}

    def __init__(self, *a, **k):
        pass

    def __call__(self, img):
        return [{"cls_name": "person", "score": 0.9, "box": [100, 100, 200, 400]},
                {"cls_name": "helmet", "score": 0.8, "box": [130, 90, 170, 125]},
                {"cls_name": "vest", "score": 0.85, "box": [105, 160, 195, 260]},
                {"cls_name": "person", "score": 0.9, "box": [300, 100, 400, 400]},
                {"cls_name": "helmet", "score": 0.8, "box": [330, 90, 370, 125]},
                {"cls_name": "vest", "score": 0.4, "box": [305, 160, 395, 260]}]   # weak vest


@pytest.fixture()
def client(monkeypatch):
    import epa.detector_onnx as dm
    from fastapi.testclient import TestClient
    monkeypatch.setattr(dm, "OnnxYoloDetector", FakeDetector)
    sys.modules.pop("epa.api", None)
    import epa.api as api
    yield TestClient(api.app)
    sys.modules.pop("epa.api", None)


def _jpg():
    ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 128, np.uint8))
    return {"file": ("a.jpg", buf.tobytes(), "image/jpeg")}


def test_detect_returns_worker_decisions(client):
    r = client.post("/detect", files=_jpg())
    assert r.status_code == 200
    labels = [w["label"] for w in r.json()["workers"]]
    assert labels == ["OK", "? vest"]          # the weak vest is not accepted


@pytest.mark.parametrize("view", ["decisions", "boxes"])
def test_annotated_returns_a_jpeg(client, view):
    r = client.post(f"/detect/annotated?view={view}", files=_jpg())
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is not None


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
