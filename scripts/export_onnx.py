"""Export a YOLOv8 PyTorch model to ONNX.

    python scripts/export_onnx.py --weights yolov8n.pt --imgsz 640

opset 12 + static input shape keeps the graph compatible with OpenCV DNN (C++)
and ONNX Runtime on CPU / ARM edge devices.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--out-dir", default="models")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    onnx_path = model.export(format="onnx", imgsz=a.imgsz, opset=a.opset,
                             simplify=True, dynamic=False)
    target = out_dir / Path(onnx_path).name
    shutil.move(onnx_path, target)
    pt = Path(a.weights)
    if pt.exists() and pt.parent.resolve() != out_dir.resolve():
        shutil.move(str(pt), out_dir / pt.name)
    print(f"Exported -> {target}")


if __name__ == "__main__":
    main()
