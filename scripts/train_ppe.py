"""Fine-tune YOLOv8n (COCO-pretrained) on the 3-class helmet dataset, then
export to ONNX for the same edge pipeline as the rest of the project.

    # NVIDIA GPU: ~10-20 min for 100 epochs
    python scripts/train_ppe.py --epochs 100 --imgsz 640
    # CPU only (laptop without NVIDIA GPU): roughly 1-2 h for 20 epochs
    python scripts/train_ppe.py --epochs 20 --imgsz 416

The device is chosen automatically (GPU if PyTorch can see one, otherwise CPU).

Writes models/<name>.pt, models/<name>.onnx and results/<name>_metrics.json
(default name: ppe_yolov8n; use --name for extra runs so they do not overwrite it).
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/ppe3/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto", help="auto | cpu | 0 (first NVIDIA GPU)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default="ppe_yolov8n")
    p.add_argument("--patience", type=int, default=30,
                   help="early stopping after N epochs without improvement (0 = never stop early)")
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="switch mosaic augmentation off for the last N epochs (0 = keep it on)")
    p.add_argument("--eval-only", metavar="BEST_PT",
                   help="skip training: evaluate + export an existing best.pt")
    a = p.parse_args()
    if a.device == "auto":
        import torch
        a.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"device: {'GPU ' + torch.cuda.get_device_name(0) if a.device == '0' else 'CPU'}")

    Path("models").mkdir(exist_ok=True)
    if a.eval_only:
        best = Path(a.eval_only)
    else:
        model = YOLO(a.weights)
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                    workers=a.workers, project="runs", name=a.name, exist_ok=True,
                    patience=a.patience, close_mosaic=a.close_mosaic, plots=True)
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, f"models/{a.name}.pt")

    # Evaluate on the held-out test split, per class
    m = YOLO(f"models/{a.name}.pt")
    res = m.val(data=a.data, split="test", imgsz=a.imgsz, device=a.device, plots=False)
    per_class = {m.names[c]: {"precision": round(float(res.box.p[i]), 3),
                              "recall": round(float(res.box.r[i]), 3),
                              "mAP50": round(float(res.box.ap50[i]), 3),
                              "mAP50-95": round(float(res.box.ap[i]), 3)}
                 for i, c in enumerate(res.box.ap_class_index)}
    metrics = {"split": "test", "imgsz": a.imgsz, "epochs": a.epochs,
               "mAP50": round(float(res.box.map50), 3),
               "mAP50-95": round(float(res.box.map), 3), "per_class": per_class}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{a.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    onnx_path = m.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False)
    shutil.move(onnx_path, f"models/{a.name}.onnx")
    print(f"-> models/{a.name}.pt, models/{a.name}.onnx, results/{a.name}_metrics.json")


if __name__ == "__main__":
    main()
