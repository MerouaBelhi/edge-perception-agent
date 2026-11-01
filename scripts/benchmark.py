"""Latency benchmark on one machine: CPU section + GPU section.

    python scripts/benchmark.py --images assets --runs 50

CPU rows (speed-up vs PyTorch FP32 on CPU):
    PyTorch FP32 | ONNX Runtime FP32 (hand-written pipeline) | ONNX Runtime INT8
GPU rows, only if CUDA is available (speed-up vs PyTorch FP32 on GPU):
    PyTorch FP32 | PyTorch FP16 | ONNX Runtime FP32 on the CUDA provider (needs onnxruntime-gpu)

Methodology
- Every backend runs in its OWN process. In one shared process, PyTorch's CPU
  thread pool competed with ONNX Runtime's and made the ONNX numbers erratic.
- Devices are pinned explicitly: Ultralytics would otherwise silently use the GPU.
- Batch 1, 640x640 input, 5 warm-up runs, end-to-end time (pre + inference + post).

Writes results/benchmark.json and results/benchmark.md (tables for the README).
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epa.detector_onnx import OnnxYoloDetector  # noqa: E402

BACKENDS = {  # key: (section, label)
    "pt_cpu": ("cpu", "PyTorch FP32 (Ultralytics)"),
    "onnx_cpu": ("cpu", "ONNX Runtime FP32 (hand-written pipeline)"),
    "int8_cpu": ("cpu", "ONNX Runtime INT8 (static PTQ)"),
    "pt_gpu": ("gpu", "PyTorch FP32 (Ultralytics)"),
    "pt_gpu_fp16": ("gpu", "PyTorch FP16 (Ultralytics, half=True)"),
    "onnx_gpu": ("gpu", "ONNX Runtime FP32, CUDA provider (hand-written pipeline)"),
}


def cpu_name() -> str:
    try:                                                # Linux
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()  # Windows / macOS


def stats(ms):
    a = np.array(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "fps": float(1000.0 / a.mean())}


def bench(fn, frames, runs, warmup=5):
    for i in range(warmup):
        fn(frames[i % len(frames)])
    ms = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(frames[i % len(frames)])
        ms.append((time.perf_counter() - t0) * 1e3)
    return stats(ms)


def run_backend(kind: str, a) -> dict:
    """Benchmark ONE backend in this process. Returns stats or {"skipped": reason}."""
    files = [f for f in sorted(Path(a.images).iterdir())
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    frames = [cv2.imread(str(f)) for f in files]
    if not frames:
        raise SystemExit(f"No images in {a.images}")

    if kind.startswith("pt_"):
        if not Path(a.pt).exists():
            return {"skipped": f"{a.pt} not found"}
        import torch
        gpu = kind != "pt_cpu"
        if gpu and not torch.cuda.is_available():
            return {"skipped": "no CUDA GPU"}
        from ultralytics import YOLO
        m = YOLO(a.pt)
        kw = {"imgsz": 640, "device": 0 if gpu else "cpu", "verbose": False,
              "half": kind == "pt_gpu_fp16"}
        r = bench(lambda f: m(f, **kw), frames, a.runs)
        r["device"] = torch.cuda.get_device_name(0) if gpu else cpu_name()
        return r

    path = a.int8 if kind == "int8_cpu" else a.onnx
    if not Path(path).exists():
        return {"skipped": f"{path} not found"}
    if kind == "onnx_gpu":
        try:
            d = OnnxYoloDetector(path, device="cuda")
        except RuntimeError as e:
            return {"skipped": str(e)}
        if d.provider != "CUDAExecutionProvider":
            return {"skipped": f"ONNX Runtime fell back to {d.provider}"}
    else:
        d = OnnxYoloDetector(path)
    stages = {"pre_ms": [], "infer_ms": [], "post_ms": []}

    def call(f):
        d(f)
        for k, v in d.last_timing.items():
            stages[k].append(v)

    r = bench(call, frames, a.runs)
    r["stages_ms"] = {k: round(float(np.mean(v[-a.runs:])), 2) for k, v in stages.items()}
    r["device"] = d.provider
    return r


def table(rows: dict, base_label: str) -> list[str]:
    base = rows[base_label]["mean_ms"] if base_label in rows else None
    out = ["| Backend | Mean (ms) | p95 (ms) | FPS | Speed-up |", "|---|---:|---:|---:|---:|"]
    for label, v in rows.items():
        sp = f"{base / v['mean_ms']:.2f}x" if base else "–"
        out.append(f"| {label} | {v['mean_ms']:.1f} | {v['p95_ms']:.1f} | {v['fps']:.1f} | {sp} |")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="assets")
    p.add_argument("--pt", default="models/yolov8n.pt")
    p.add_argument("--onnx", default="models/yolov8n.onnx")
    p.add_argument("--int8", default="models/yolov8n_int8.onnx")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--out", default="results")
    p.add_argument("--worker", choices=list(BACKENDS), help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.worker:                       # child process: one backend, JSON on the last line
        print(json.dumps(run_backend(a.worker, a)))
        return

    sections = {"cpu": {}, "gpu": {}}
    devices = {"cpu": cpu_name(), "gpu": None}
    skipped = {}
    for kind, (section, label) in BACKENDS.items():
        print(f"measuring {kind} ...", flush=True)
        cmd = [sys.executable, __file__, "--worker", kind, "--images", a.images, "--pt", a.pt,
               "--onnx", a.onnx, "--int8", a.int8, "--runs", str(a.runs)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines_out = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if proc.returncode != 0 or not lines_out:
            err = proc.stderr.strip().splitlines()
            skipped[kind] = "failed: " + (err[-1] if err else "no output")
            continue
        r = json.loads(lines_out[-1])
        if "skipped" in r:
            skipped[kind] = r["skipped"]
            continue
        if section == "gpu" and kind.startswith("pt_"):
            devices["gpu"] = r["device"]
        sections[section][label] = r

    md = [f"**CPU: {devices['cpu']}**", ""]
    md += table(sections["cpu"], BACKENDS["pt_cpu"][1])
    if sections["gpu"]:
        md += ["", f"**GPU: {devices['gpu']}**", ""]
        md += table(sections["gpu"], BACKENDS["pt_gpu"][1])
    stage_lines = [f"- {sec.upper()} {label}: pre {v['stages_ms']['pre_ms']} ms, "
                   f"inference {v['stages_ms']['infer_ms']} ms, post {v['stages_ms']['post_ms']} ms"
                   for sec, rows in sections.items() for label, v in rows.items() if "stages_ms" in v]
    if stage_lines:
        md += ["", "Time per stage (ONNX rows):", ""] + stage_lines
    text = "\n".join(md)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "benchmark.md").write_text(text + "\n")
    (out / "benchmark.json").write_text(json.dumps(
        {"meta": {"cpu": devices["cpu"], "gpu": devices["gpu"], "runs": a.runs,
                  "input": "640x640, batch 1", "python": platform.python_version()},
         "results": sections, "skipped": skipped}, indent=2))
    print("\n" + text)
    if skipped:
        print("\nSkipped:")
        for k, why in skipped.items():
            print(f"- {k}: {why}")


if __name__ == "__main__":
    main()
