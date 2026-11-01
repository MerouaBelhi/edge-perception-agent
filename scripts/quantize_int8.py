"""Post-training static INT8 quantization of the ONNX model (ONNX Runtime).

    python scripts/quantize_int8.py --model models/yolov8n.onnx --calib assets/ --n 50

Calibration uses real frames (images in a folder, or frames sampled from a
video) so the activation ranges match the deployment data.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epa.detector_onnx import letterbox  # noqa: E402


def load_frames(src: str, n: int) -> list[np.ndarray]:
    p = Path(src)
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        return [cv2.imread(str(f)) for f in files[:n]]
    cap = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    frames = []
    for idx in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    return frames


class FrameReader(CalibrationDataReader):
    def __init__(self, frames, input_name, size):
        self.data = []
        for f in frames:
            img, *_ = letterbox(f, size)
            blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.data.append({input_name: blob.transpose(2, 0, 1)[None]})
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/yolov8n.onnx")
    p.add_argument("--calib", required=True, help="image folder or video")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()

    src = Path(a.model)
    prep = src.with_name(src.stem + "_prep.onnx")
    dst = src.with_name(src.stem + "_int8.onnx")
    # Per-channel QDQ (DequantizeLinear with `axis`) requires opset >= 13
    import onnx
    from onnx import version_converter
    m = onnx.load(str(src))
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    if opset < 13:
        m = version_converter.convert_version(m, 13)
        src13 = src.with_name(src.stem + "_op13.onnx")
        onnx.save(m, str(src13))
        quant_pre_process(str(src13), str(prep))
        src13.unlink()
    else:
        quant_pre_process(str(src), str(prep))

    import onnxruntime as ort
    input_name = ort.InferenceSession(str(prep), providers=["CPUExecutionProvider"]) \
        .get_inputs()[0].name
    frames = load_frames(a.calib, a.n)
    if not frames:
        raise SystemExit("No calibration frames found")

    # Keep the detection-head decoding in float. YOLOv8 concatenates box
    # coordinates (0..640) with class scores (0..1) in one tensor: a single
    # INT8 scale for both wipes out the scores (0 detections). We therefore
    # exclude the final 1x1 prediction convs (cv2.i.2 / cv3.i.2) and the DFL /
    # box-decoding ops; the backbone, neck and head trunk stay INT8.
    graph = onnx.load(str(prep)).graph
    exclude = []
    for n in graph.node:
        if "/model.22/" not in n.name:
            continue
        in_trunk = any(f"/{b}.{i}/{b}.{i}.{j}/" in n.name
                       for b in ("cv2", "cv3") for i in range(3) for j in (0, 1))
        if not in_trunk:
            exclude.append(n.name)

    quantize_static(str(prep), str(dst), FrameReader(frames, input_name, a.imgsz),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    nodes_to_exclude=exclude)
    print(f"Excluded {len(exclude)} head/decoding nodes from quantization")
    prep.unlink(missing_ok=True)
    print(f"INT8 model -> {dst}  ({len(frames)} calibration frames)")


if __name__ == "__main__":
    main()
