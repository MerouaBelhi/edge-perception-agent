"""REST inference service (FastAPI + ONNX Runtime, no PyTorch at runtime).

    uvicorn epa.api:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health           model info and readiness
    POST /detect           image file -> JSON detections + per-stage latency
    POST /detect/annotated image file -> JPEG with boxes drawn
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from epa.detector_onnx import OnnxYoloDetector

# released safety model first (INT8, then FP32), COCO model as a fallback
MODEL_PATH = os.getenv("EPA_MODEL") or next(
    (m for m in ("models/site11hh_v2_int8.onnx", "models/site11hh_v2.onnx",
                 "models/yolov8n_int8.onnx", "models/yolov8n.onnx") if os.path.exists(m)),
    "models/yolov8n.onnx")

app = FastAPI(title="Edge Perception Agent", version="0.1.0",
              description="Object detection served with a hand-written ONNX Runtime pipeline.")
detector = OnnxYoloDetector(MODEL_PATH, conf_thres=0.25)


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH),
            "input_size": detector.input_size}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.0, le=1.0)):
    img = _decode(await file.read())
    detector.conf_thres = conf
    dets = detector(img)
    return {"image_size": [img.shape[1], img.shape[0]], "count": len(dets),
            "detections": dets, "timing_ms": detector.last_timing}


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...),
                           conf: float = Query(0.25, ge=0.0, le=1.0)):
    img = _decode(await file.read())
    detector.conf_thres = conf
    for d in detector(img):
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f'{d["cls_name"]} {d["score"]:.2f}', (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
