"""REST inference service (FastAPI + ONNX Runtime, no PyTorch at runtime).

    uvicorn epa.api:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /                 redirects to the interactive docs (/docs)
    GET  /health           model info and readiness
    POST /detect           image file -> JSON: raw detections, per-worker safety
                           decisions (hard hat, vest) and per-stage latency
    POST /detect/annotated image file -> JPEG: each worker's decision drawn
                           (view=decisions, default) or the raw boxes (view=boxes)
"""

from __future__ import annotations

import os
import zlib

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response

from epa.detector_onnx import OnnxYoloDetector
from epa.safety import SafetyConfig, frame_status, short_label

# released safety model first (INT8, then FP32), COCO model as a fallback
MODEL_PATH = os.getenv("EPA_MODEL") or next(
    (m for m in ("models/site11hh_v2_int8.onnx", "models/site11hh_v2.onnx",
                 "models/yolov8n_int8.onnx", "models/yolov8n.onnx") if os.path.exists(m)),
    "models/yolov8n.onnx")

app = FastAPI(title="Edge Perception Agent", version="0.1.0",
              description="Worker-safety detection (hard hat, vest) served with a hand-written ONNX Runtime pipeline.")
detector = OnnxYoloDetector(MODEL_PATH, conf_thres=0.25)


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")          # the browser lands on the interactive docs


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH),
            "input_size": detector.input_size}


GREEN, RED, GREY, YELLOW = (60, 180, 60), (40, 40, 220), (150, 150, 150), (0, 200, 255)
SAFETY = "person" in detector.class_names and "helmet" in detector.class_names


def _workers(img: np.ndarray, dets: list[dict]) -> list[dict]:
    """Per-worker decisions with the same single-image rules as scripts/detect_images.py."""
    if not SAFETY:
        return []
    cfg = SafetyConfig(check_vest="vest" in detector.class_names,
                       input_scale=detector.input_size / max(img.shape[:2]))
    out = []
    for p in frame_status(dets, cfg):
        label = short_label(p)
        out.append({"box": [round(v, 1) for v in p["box"]], "score": round(p["score"], 3),
                    "hard_hat": p["ppe"], "vest": p.get("vest"), "label": label})
    return out


def _put_label(img, text, x1, y1, color, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    ty = max(y1, th + 6)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
    cv2.putText(img, text, (x1 + 3, ty - 4), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.0, le=1.0)):
    img = _decode(await file.read())
    detector.conf_thres = conf
    dets = detector(img)
    return {"image_size": [img.shape[1], img.shape[0]], "count": len(dets),
            "workers": _workers(img, dets), "detections": dets, "timing_ms": detector.last_timing}


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...),
                           conf: float = Query(0.3, ge=0.0, le=1.0),
                           view: str = Query("decisions", pattern="^(decisions|boxes)$")):
    img = _decode(await file.read())
    detector.conf_thres = conf
    dets = detector(img)
    W = img.shape[1]
    scale, thick = max(0.4, min(W / 1400, 0.9)), max(1, round(W / 700))
    if view == "decisions" and SAFETY:
        for d in dets:                                         # machines / vehicles for context
            if d["cls_name"] in ("machinery", "vehicle"):
                x1, y1, x2, y2 = map(int, d["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, thick)
        # farthest first (feet highest in the image): labels of nearer workers end on top
        for w in sorted(_workers(img, dets), key=lambda w: w["box"][3]):
            x1, y1, x2, y2 = map(int, w["box"])
            col = GREEN if w["label"] == "OK" else GREY if w["label"].startswith("?") else RED
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick + 1)
            _put_label(img, w["label"], x1, y1, col, scale)
    else:
        for d in sorted(dets, key=lambda d: -(d["box"][2] - d["box"][0])):   # small boxes on top
            x1, y1, x2, y2 = map(int, d["box"])
            rng = np.random.default_rng(zlib.crc32(d["cls_name"].encode()))
            col = tuple(int(c) for c in rng.integers(60, 230, 3))          # one colour per class
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
            _put_label(img, f'{d["cls_name"]} {d["score"]:.2f}', x1, y1, col, scale * 0.8)
    ok, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
