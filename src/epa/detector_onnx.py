"""YOLOv8 inference with ONNX Runtime, written by hand.

No Ultralytics at inference time: pre-processing (letterbox), output decoding
and NMS are implemented here, which is exactly what has to be re-done when a
model is moved to an edge device or to C++ (see ../../cpp/main.cpp).

YOLOv8 ONNX output: (1, 4 + num_classes, num_anchors), e.g. (1, 84, 8400)
  rows 0..3  -> box centre x, centre y, width, height (in input pixels)
  rows 4..   -> per-class scores (already sigmoid-activated, no objectness)
"""

from __future__ import annotations

import ast
import time

import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def letterbox(img: np.ndarray, new_size: int = 640, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a square. Returns the padded
    image plus (scale, pad_x, pad_y) needed to map boxes back.
    copyMakeBorder pads in one C++ call (~5x faster than allocating a canvas
    and copying into it with NumPy)."""
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    out = cv2.copyMakeBorder(resized, pad_y, new_size - nh - pad_y, pad_x, new_size - nw - pad_x,
                             cv2.BORDER_CONSTANT, value=color)
    return out, scale, pad_x, pad_y


class OnnxYoloDetector:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45,
                 class_names: list[str] | None = None, num_threads: int | None = None,
                 device: str = "cpu"):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            opts.intra_op_num_threads = num_threads
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider not available: install onnxruntime-gpu "
                                   "(and uninstall onnxruntime) to run ONNX on the GPU")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        # the provider ONNX Runtime actually uses (it silently falls back to CPU on errors)
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = class_names or self._names_from_metadata() or COCO_CLASSES
        self.last_timing: dict[str, float] = {}

    def _names_from_metadata(self) -> list[str] | None:
        """Ultralytics stores class names in the ONNX metadata, e.g.
        names = "{0: 'person', 1: 'helmet', 2: 'no_helmet'}"."""
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return None
        try:
            d = ast.literal_eval(raw)
            return [d[i] for i in sorted(d)]
        except (ValueError, SyntaxError):
            return None

    def preprocess(self, frame_bgr: np.ndarray):
        img, scale, px, py = letterbox(frame_bgr, self.input_size)
        # BGR->RGB, /255 and HWC->NCHW in one C++ call (same as the C++ engine)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)
        return blob, scale, px, py

    def postprocess(self, output: np.ndarray, scale: float, px: int, py: int,
                    frame_shape) -> list[dict]:
        preds = output[0].T                       # (8400, 84)
        boxes_cxcywh = preds[:, :4]
        scores_all = preds[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(cls_ids)), cls_ids]

        keep = scores > self.conf_thres
        if not keep.any():
            return []
        boxes_cxcywh, scores, cls_ids = boxes_cxcywh[keep], scores[keep], cls_ids[keep]

        # cxcywh (letterboxed input space) -> xyxy (original image space)
        cx, cy, w, h = boxes_cxcywh.T
        x1 = (cx - w / 2 - px) / scale
        y1 = (cy - h / 2 - py) / scale
        x2 = (cx + w / 2 - px) / scale
        y2 = (cy + h / 2 - py) / scale
        H, W = frame_shape[:2]
        x1, x2 = np.clip(x1, 0, W), np.clip(x2, 0, W)
        y1, y2 = np.clip(y1, 0, H), np.clip(y2, 0, H)

        # Class-aware NMS: offset boxes per class so different classes never suppress each other
        offset = cls_ids.astype(np.float32) * 4096.0
        nms_boxes = np.stack([x1 + offset, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.conf_thres, self.iou_thres)
        idx = np.array(idx).flatten() if len(idx) else []

        dets = []
        for i in idx:
            c = int(cls_ids[i])
            dets.append({
                "box": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "score": float(scores[i]),
                "cls_id": c,
                "cls_name": self.class_names[c] if c < len(self.class_names) else str(c),
            })
        return dets

    def __call__(self, frame_bgr: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()
        blob, scale, px, py = self.preprocess(frame_bgr)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        dets = self.postprocess(out, scale, px, py, frame_bgr.shape)
        t3 = time.perf_counter()
        self.last_timing = {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3,
                            "post_ms": (t3 - t2) * 1e3}
        return dets
