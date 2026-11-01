**CPU: AMD Ryzen 9 7900X 12-Core Processor**

| Backend | Mean (ms) | p95 (ms) | FPS | Speed-up |
|---|---:|---:|---:|---:|
| PyTorch FP32 (Ultralytics) | 13.4 | 15.3 | 74.4 | 1.00x |
| ONNX Runtime FP32 (hand-written pipeline) | 31.5 | 40.4 | 31.7 | 0.43x |
| ONNX Runtime INT8 (static PTQ) | 19.8 | 24.5 | 50.5 | 0.68x |

**GPU: NVIDIA GeForce RTX 4090**

| Backend | Mean (ms) | p95 (ms) | FPS | Speed-up |
|---|---:|---:|---:|---:|
| PyTorch FP32 (Ultralytics) | 3.6 | 4.1 | 280.9 | 1.00x |
| PyTorch FP16 (Ultralytics, half=True) | 3.6 | 4.0 | 279.8 | 1.00x |

Time per stage (ONNX rows):

- CPU ONNX Runtime FP32 (hand-written pipeline): pre 5.8 ms, inference 24.27 ms, post 0.96 ms
- CPU ONNX Runtime INT8 (static PTQ): pre 4.94 ms, inference 13.6 ms, post 0.75 ms
