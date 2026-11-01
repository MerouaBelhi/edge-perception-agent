| Image | Object | FP32 score | INT8 score | Box IoU FP32 vs INT8 |
|---|---|---:|---:|---:|
| bus.jpg | person | 0.89 | 0.86 | 0.99 |
| bus.jpg | person | 0.88 | 0.89 | 0.98 |
| bus.jpg | person | 0.88 | 0.89 | 1.00 |
| bus.jpg | bus | 0.84 | 0.86 | 0.97 |
| bus.jpg | person | 0.44 | 0.31 | 0.98 |
| zidane.jpg | person | 0.83 | 0.78 | 0.98 |
| zidane.jpg | person | 0.83 | 0.79 | 0.97 |

7/7 FP32 detections found again by INT8 (INT8 total: 7); largest confidence drop: 0.13.
