"""Post-processing of the hand-written ONNX detector (no model file needed)."""

import numpy as np

from epa.detector_onnx import OnnxYoloDetector


def _detector(nc=2):
    d = OnnxYoloDetector.__new__(OnnxYoloDetector)        # skip loading an ONNX session
    d.conf_thres, d.iou_thres, d.class_names = 0.3, 0.45, [f"c{i}" for i in range(nc)]
    return d


def _output(rows):
    """rows: [(cx, cy, w, h, cls, score)] in letterboxed input space -> (1, 4 + nc, N)."""
    nc = 2
    out = np.zeros((1, 4 + nc, len(rows)), np.float32)
    for j, (cx, cy, w, h, c, s) in enumerate(rows):
        out[0, :4, j] = cx, cy, w, h
        out[0, 4 + c, j] = s
    return out


def test_class_aware_nms_on_a_wide_photo():
    # 6000 px wide photo, scale 0.1: a class-1 box at x ~ 500 px and a class-0 box at
    # x ~ 4596 px. A fixed per-class offset of 4096 px would put them on top of each
    # other and NMS would drop one; different classes must never suppress each other.
    det = _detector()
    out = _output([(459.6, 100, 20, 40, 0, 0.9), (50.0, 100, 20, 40, 1, 0.8)])
    dets = det.postprocess(out, scale=0.1, px=0, py=0, frame_shape=(4000, 6000, 3))
    assert sorted(d["cls_name"] for d in dets) == ["c0", "c1"]


def test_nms_still_merges_duplicates_of_one_class():
    det = _detector()
    out = _output([(100, 100, 20, 40, 0, 0.9), (101, 100, 20, 40, 0, 0.7)])
    dets = det.postprocess(out, scale=1.0, px=0, py=0, frame_shape=(640, 640, 3))
    assert len(dets) == 1 and abs(dets[0]["score"] - 0.9) < 1e-6
