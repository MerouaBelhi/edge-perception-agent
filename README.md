# Edge Perception Agent

**Real-time worker-safety monitoring for construction sites, built to run on a CPU.** A YOLOv8n detector, multi-object tracking and per-worker decision rules flag missing hard hats, high-visibility vests and masks, non-work headwear (caps, sun hats, bicycle helmets), workers too close to machinery and people inside coned-off areas. Inference runs with ONNX Runtime in Python and C++, with INT8 quantization, and an LLM + vision-language agent answers questions about the video and double-checks each alert.

![results on construction-site images](assets/results_panel.jpg)

*Released model on held-out construction-site images. Green `OK`: hard hat and vest worn. Red: what is missing (`no vest`, `no helmet`, `not hard hat` = non-work headwear). Grey `?`: cannot be judged (operator in a cab, person too small or hidden). Yellow: machines and vehicles.*

| Part | What is implemented |
|---|---|
| **Detection** | YOLOv8n fine-tuned on 11 classes: `person`, `helmet`, `no_helmet`, `other_headwear`, `vest`, `no_vest`, `mask`, `no_mask`, `machinery`, `vehicle`, `safety_cone` |
| **Per-worker decisions** | Hard hat / vest / mask attached to the right worker (head, torso and face regions), missing items inferred from a clearly visible body part, temporal voting, `unknown` when the system cannot judge |
| **Hazards** | Worker-machine proximity measured on the ground in person heights, operators in a cab ignored, coned-off areas from the cones' convex hull |
| **Tracking and zones** | ByteTrack IDs, polygon zones that each list the PPE they require, one alert per incident, evidence crops, JSON logs |
| **Evaluation** | Per-worker evaluation (not only mAP), an external dataset with a held-out part, before / after per image, a control training run |
| **Edge deployment** | ONNX export, hand-written pre/post-processing and NMS in Python and C++, static INT8 quantization (1.9× faster, 3.3× smaller), FastAPI service in Docker without PyTorch |
| **Scene agent** | LLM with tool calling over the tracker logs + a VLM that checks each violation's evidence crop; runs locally with Ollama |
| **Quality** | 81 unit tests (detector post-processing, rules, hazards, zones, agent, dataset scripts), GitHub Actions CI |

---

## Results of the released model (`site11hh_v2`)

YOLOv8n at 960 px. Training data: [Construction Site Safety](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) (CC BY 4.0) + headwear and vehicle images from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) + 4.5k images of [Hard Hat Workers](https://huggingface.co/datasets/Voxel51/hard-hat-detection) (CC0); 500 more Hard Hat Workers images were set aside before training and never used.

mAP scores boxes; what matters for safety is the decision taken for each worker. `scripts/eval_safety.py` matches the system's people to the labelled ones and compares its decision with the label:

| Test split (Construction Site Safety + Open Images) | Missing item caught ↑ | Missing item called "worn" ↓ | False alarms when worn ↓ | Worn item recognised ↑ |
|---|---:|---:|---:|---:|
| Hard hat (40 bare heads, 101 worn) | 65 % | **0 %** | 2 % | 87 % |
| Hard hat: non-work headwear (191) | 71 % | 2 % | | |
| Vest (87 without, 61 with) | 79 % | **0 %** | 2 % | 87 % |
| Mask (69 without, 26 with; never inferred) | 77 % | **0 %** | 0 % | 73 % |

| **External** held-out images: new sites, cameras and people (1995 hard hats, 591 bare heads) | |
|---|---:|
| Bare heads caught ↑ | 70 % |
| Bare heads called "helmet" ↓ | 3 % |
| False alarms on hard hats ↓ | 1 % |
| Hard hats recognised ↑ | 70 % |

Most remaining misses are **people the detector does not find at all** (tight groups, top-down views, people partly hidden or cut off), not wrong decisions: a worker who is found is rarely judged wrongly. The splits are small (40 bare heads in the test split), so one head moves a percentage by 2.5 points.

Detector, mAP50 per class on the test split:

| person | helmet | no_helmet | other_headwear | vest | no_vest | mask | no_mask | machinery | vehicle | safety_cone | **all** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.84 | 0.52 | 0.73 | 0.89 | 0.77 | 0.79 | 0.84 | 0.82 | 0.50 | 0.46 | **0.72** |

Vehicles and cones are the weakest classes (recall 0.36 and 0.41: few examples in the data); the cone rule remembers cones for 3 s to bridge missed detections. `no_helmet` is weak too, which is why a missing hard hat is also inferred from a clearly visible head (next section).

---

## How a decision is made

1. **Detect** every frame (11 classes above).
2. **Associate** each PPE box to one tracked person: hard hats and headwear to the **head** region, vests to the **torso**, masks to the **face**. The regions adapt to upper-body close-ups (a box less than twice as tall as it is wide). With workers close together, the nearest region wins and each box is used once. A PPE box must be wide enough for the person (`--min-ppe-width`), so the helmet of a worker standing behind is not given to the one in front.
3. **Infer what is missing.** The `no_helmet` class is weak (mAP50 0.52), so a **missing** hard hat on a **clearly visible** head (person large enough at the detector's resolution, not cut by the image border, not hidden behind someone nearer the camera, no PPE box of any kind on the head) also counts as evidence. The same holds for vests. Masks are never inferred from absence. A worker in `other_headwear` has no hard hat, and the alert says *non-work headwear*, not *no helmet*.
4. **Vote over time.** Per worker, a vote over the last 1.5 s gives `helmet`, `no_helmet` or `unknown` (and the same for vest and mask). A vest does not come off in 1.5 s, so "no vest" needs ≥ 90 % of the window without one.
5. **Apply the zone rule.** Each zone lists what it requires (`configs/site_safety_zones.json`: `"requires": ["helmet", "vest"]`, a paint room adds `"mask"`). A missing item inside such a zone for ≥ 1 s gives **one** violation, re-armed only when the worker leaves or the item is seen again for 2 s.
6. **Save evidence.** The worker's crop goes to `evidence/`, and the violation records how it was decided (`detected` or `inferred_from_absence`). The agent's `inspect_violation` tool asks the vision model to check the crop.

`unknown` (head hidden, too small, turned away) never raises an alert: in a safety system, false alarms make people ignore the alerts. Unknowns are still logged for review.

**Hazards around people** (`--hazards`, `src/epa/hazards.py`):

- `machine_proximity`: a worker closer to a machine or vehicle than one of his own body heights, for ≥ 0.5 s. Measuring in person heights adapts the threshold to the distance from the camera. One camera gives no depth, so the gap is measured *on the ground*: the worker's feet against the machine's lower half (its footprint), so a worker standing far behind a machine, who overlaps it in the image, is not "near". Operators in the cab (feet above the footprint) are ignored, the worker's feet must be visible, and a worker much smaller than the machine's scale is treated as further away. Machines are tracked, so each alert says whether the machine was moving (`--only-moving` keeps only those).
- `cone_area_entry`: cones seen in the last 3 s (cone detections flicker) define an area, their convex hull; a person (or, with `--cone-keep-out vehicle`, a vehicle) inside it for ≥ 0.5 s raises a violation. Cones in a line define no area.
- Limits: distances are approximate (no depth), "moving" comes from image motion (a panning camera makes everything move), and the data has fewer machines and cones than people.

---

## Evaluation on new sites

The test split comes from the same sources as the training data. To measure generalisation, the system was run on Hard Hat Workers (~5k images, mostly Chinese construction and power sites), which no model had seen (`scripts/external_test.py`). It exposed a real weakness: **coloured hard hats (red, blue, purple, pink) were often judged `other_headwear` or `no_helmet`, giving 18 % false alarms.**

Fix (`scripts/prepare_hardhat.py`): 500 images were set aside first and never trained on; the other 4.5k were added to training. That dataset labels only helmets, heads and a few people, so the classes it does not label (vests, machines, unlabelled people) were filled in with the current model's confident predictions; otherwise they would be learned as background. Same 500 held-out images, before / after:

| Held-out external images (1995 hard hats, 591 bare heads) | Before (`site11_960`) | v1 (`site11hh`) | **v2 (`site11hh_v2`)** |
|---|---:|---:|---:|
| Bare heads caught ↑ | 61 % | 67 % | **70 %** |
| Bare heads called "helmet" ↓ | 4 % | 4 % | **3 %** |
| **False alarms on hard hats** ↓ | 18 % | 1 % | **1 %** |
| Hard hats recognised ↑ | 51 % | 66 % | **70 %** |
| Hard hats on people not detected ↓ | 578 | 659 | **560** |
| Heads fixed / broken vs before (`scripts/compare_models.py`) | | 458 / 126 | **518 / 69** |
| Images better / worse / unchanged | | 189 / 45 / 266 | **214 / 25 / 261** |

**v1 fixed the hard hats but lost people.** `scripts/compare_models.py` saves the most improved and the most degraded images automatically, so the review is not cherry-picked. The degraded ones showed one main cause: more workers were not detected at all (578 → 659). v1 added the previous model's people only at confidence ≥ 0.5, so many real people stayed unlabelled and were learned as background. **v2 lowers that threshold to 0.3** (`--person-conf 0.3`), everything else unchanged: missed people drop below the original model (560), and the broken heads fall from 126 to 69. In the reviewed worst v2 cases, no worker with a hard hat was flagged and no bare head was accepted: the remaining regressions are missed people. Part of them are an artefact of the dataset, whose 416 px images are padded with mirrored copies: heads in the mirrored borders count as ground truth.

![before / after on held-out external images](assets/before_after.jpg)

*Held-out external images, before (left) and after (right), with the full decision drawn (hard hat + vest). Before, coloured hard hats were judged "no helmet" or "not hard hat"; after, they are recognised, and the alert that remains, "no vest", is correct: these workers wear no high-visibility vest. The title bars count hard-hat decisions only, since this dataset labels heads, not vests. Visible remaining errors: two workers in a red and a pink shirt accepted as wearing a vest (`OK`, second row), a few people too small to judge (`?`), and mirrored copies at the bottom of each image, which come from the dataset's padding. Pairs chosen automatically (most improved), made with `scripts/compare_models.py --check-vest --panel 4`.*

On the original test split, v2 keeps its behaviour: bare heads called "helmet" 0 %, false alarms on hard hats 2 % (4 % before), hard hats recognised 87 % (83 %), bare heads caught 65 % (68 %). Two costs: 4 of the 191 workers in non-work headwear (2 %) are now accepted as wearing a hard hat (0 before), and the `helmet` *box* mAP50 on the test split fell from 0.96 to 0.84 while the per-worker decisions improved. With 16k extra hard hats in training, the helmet class became more permissive and partly learned the external dataset's box style (see the control experiment below).

**Control experiment: one clean training run.** `site11hh_v2` is the end of a chain of fine-tunings (640 px → 960 px → + external data). To check that the chain does not hide anything, the same final dataset was trained once from the COCO weights for 100 epochs with no early stopping (`site_final`):

| | `site11hh_v2` (released) | `site_final` (single run) |
|---|---:|---:|
| External: bare heads caught ↑ | 70 % | **72 %** |
| External: bare heads called "helmet" ↓ | **3 %** | 4 % |
| External: false alarms on hard hats ↓ | 1 % | 1 % |
| External: hard hats recognised ↑ | 70 % | **72 %** |
| External: hard hats on people not detected ↓ | 560 | **538** |
| Test: bare heads called "helmet" ↓ | **0 %** | 2 % (1 of 40) |
| Test: hard hats recognised ↑ | **87 %** | 82 % |
| Test: non-work headwear called "helmet" ↓ | 2 % | **1 %** |
| Test: `helmet` box mAP50 | **0.84** | 0.78 |
| Best validation mAP50 | **0.787** (epoch 38/40) | 0.774 (epoch 97/100) |

The two are close: the single run is slightly better on the external sites and worse on the original data, mostly on the `helmet` box. A likely cause is **box style**: Hard Hat Workers draws helmet boxes differently from Construction Site Safety. With 16k external helmet boxes, the single run learned the external style, while `site11hh_v2`, fine-tuned from a model that already knew the original style, kept it. The released model stays `site11hh_v2`, which is better on the error that matters most: a bare head accepted as a hard hat (0 % on the original test split).

### Training diagnostics

`site11hh_v2`: 40 epochs at 960 px from `site11_960`, best epoch 38 (validation mAP50 0.787, mAP50-95 0.515; its validation split includes 200 Hard Hat Workers images, so these numbers are not comparable to earlier models). Train and validation losses fall together to the end (val box 1.51 → 1.43, val cls 1.05 → 0.90): no overfitting. The step in train cls loss at epoch 31 is mosaic augmentation switched off for the last 10 epochs (`close_mosaic`), not an anomaly.

![training curves, released model](assets/training/site11hh_v2_curves.png)

The control run's curves are a clean reference: train and validation losses fall together for 100 epochs, validation box loss is flat from epoch ~85 (converged), and train cls loss drops at epoch 91 when mosaic is switched off.

![training curves, single 100-epoch run](assets/training/site_final_curves.png)

| Ultralytics plots (released model) | |
|---|---|
| ![results](assets/training/results.png) | ![confusion matrix](assets/training/confusion_matrix_normalized.png) |
| ![precision-recall](assets/training/BoxPR_curve.png) | ![F1-confidence](assets/training/BoxF1_curve.png) |

Per-epoch tables (box, cls and dfl losses, train and validation, and validation metrics): [`docs/training/site11hh_v2_epochs.md`](docs/training/site11hh_v2_epochs.md), [`docs/training/site_final_epochs.md`](docs/training/site_final_epochs.md). Interactive: `python scripts/training_report.py --run site11hh_v2 site_final --tensorboard && tensorboard --logdir runs/tensorboard`.

---

## Edge deployment: ONNX, INT8 and C++

**Released model on CPU** (AMD Ryzen 9 7900X, 12 cores; 960 px input, hand-written ONNX Runtime pipeline, `scripts/latency_onnx.py`):

| Model | Size | Pre-processing | Inference | Post-processing | Total (mean) | p95 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `site11hh_v2.onnx` (FP32) | 13.0 MB | 13.3 ms | 75.2 ms | 0.7 ms | 89.2 ms | 127.2 ms | 11.2 |
| **`site11hh_v2_int8.onnx`** | **3.9 MB** | 6.1 ms | **39.1 ms** | 0.7 ms | **45.9 ms** | 70.8 ms | **21.8** |

**INT8 is 1.9× faster and 3.3× smaller.** At 960 px the network itself is most of the time, so quantization pays off much more than in the 640 px engine benchmark below. The server is shared, which explains the wide p95 and the different pre-processing times (the same code runs in both rows).

**INT8 accuracy on the safety task** (same rules, same images, `scripts/eval_safety.py` and `scripts/external_test.py`):

| | FP32 (12.9 MB) | INT8 (3.9 MB) |
|---|---:|---:|
| Test: bare heads caught ↑ | 65 % | 65 % |
| Test: bare heads called "helmet" ↓ | 0 % | 0 % |
| Test: false alarms on hard hats ↓ | 2 % | 3 % |
| Test: hard hats recognised ↑ | 87 % | 81 % |
| Test: non-work headwear caught ↑ / called "helmet" ↓ | 71 % / 2 % | 74 % / 1 % |
| Test: missing vests caught ↑ / called "vest" ↓ | 79 % / 0 % | 74 % / 0 % |
| Test: **false alarms on vests** ↓ | 2 % | **7 %** |
| Test: vests recognised ↑ | 87 % | 80 % |
| Test: missing masks caught ↑ / masks recognised ↑ | 77 % / 73 % | 71 % / 69 % |
| External: bare heads caught ↑ / called "helmet" ↓ | 70 % / 3 % | 70 % / 4 % |
| External: false alarms on hard hats ↓ / hard hats recognised ↑ | 1 % / 70 % | 2 % / 69 % |
| External: hard hats on people not detected ↓ | 560 | 583 |

INT8 keeps the safety-critical behaviour (bare heads caught, nothing missing accepted as worn) but **loses some detections**: a few more people go undetected and fewer vests and hard hats are found. Vests suffer most: a vest that is not detected on a clearly visible torso becomes a "no vest" alarm (2 % → 7 %). The model is therefore shipped in both versions: FP32 where accuracy matters, INT8 on small devices. Next steps to close the gap: more calibration images, keeping the most sensitive layers in float, or quantization-aware training.

**Engine benchmark** (COCO `yolov8n`, 640 px, batch 1, end-to-end incl. pre/post-processing, `scripts/benchmark.py`):

| CPU: AMD Ryzen 9 7900X | Mean (ms) | p95 (ms) | FPS |
|---|---:|---:|---:|
| PyTorch FP32 (Ultralytics) | 13.4 | 15.3 | 74.4 |
| ONNX Runtime FP32 (hand-written pipeline) | 31.5 | 40.4 | 31.7 |
| **ONNX Runtime INT8 (static PTQ)** | **19.8** | **24.5** | **50.5** |

| GPU: NVIDIA RTX 4090 | Mean (ms) | p95 (ms) | FPS |
|---|---:|---:|---:|
| PyTorch FP32 (Ultralytics) | 3.6 | 4.1 | 280.9 |
| PyTorch FP16 (`half=True`) | 3.6 | 4.0 | 279.8 |

- **INT8 almost halves the ONNX inference time** (24.3 → 13.6 ms) and makes the model 3.4× smaller (12.9 → 3.7 MB).
- **On this 12-core desktop CPU, PyTorch is still faster than the ONNX pipeline.** ONNX's advantage here is deployability (no PyTorch, a C++ runtime, a small model), not raw speed on a large CPU. The real test for INT8 is a small edge CPU (ARM, Jetson), the next step.
- **FP16 gives nothing on the GPU:** at batch 1, YOLOv8n is far too small to keep an RTX 4090 busy; the time goes into pre/post-processing and kernel launches.

**Engineering notes**

1. **INT8 quantization gave zero detections at first.** YOLOv8's output tensor concatenates box coordinates (0–640) with class probabilities (0–1); one INT8 scale for both rounds every score to zero. `scripts/quantize_int8.py` keeps the final 1×1 prediction convolutions and the box decoding in float, while the backbone, neck and head trunk stay INT8.
2. **Per-channel quantization requires opset ≥ 13**: the script upgrades the graph with `onnx.version_converter` first.
3. **OpenCV DNN (4.6) could not import the YOLOv8 head**, so the C++ engine uses the ONNX Runtime C++ API, the same runtime as the Python side, which gives identical results (checked box for box).
4. **Two bugs in my own benchmark.** Ultralytics silently ran on the GPU while ONNX ran on the CPU (3.0 vs 22.3 ms): every row is now pinned to its device. With all backends in one process, PyTorch's CPU thread pool competed with ONNX Runtime's and made its numbers erratic (22–43 ms): each backend now runs in its own process.

---

## Scene agent: ask questions about the video

```text
$ python -m epa.agent --run results/site -v \
    -q "Which workers entered the site without a hard hat, and is each alert real?"
  -> list_violations({"zone": "construction_site"})
  -> inspect_violation({"violation_id": 0})
  -> inspect_violation({"violation_id": 1})
```

| Question type | Answered by | Why |
|---|---|---|
| Counts, times, violations, PPE status, track IDs | Tools over the run's logs (`events.jsonl`, `tracks.jsonl`, violations) | Exact numbers from the tracker; the LLM never estimates a count |
| Is the alert real? What is the worker wearing? | `inspect_violation` / `look_at_frame` → vision-language model on the evidence crop or keyframe | The tracker records boxes and classes, not what things look like |

**Tools:** `get_video_summary`, `count_events`, `zone_occupancy`, `objects_at`, `track_history`, `list_violations`, `ppe_status_at`, `inspect_violation`, `look_at_frame`.

- The LLM sees tool results, never raw logs: the context stays small for long videos, and every number in an answer can be traced to a tool call (`-v` prints them).
- Alerts inferred from absence are sent to the VLM first; it looks at the head itself, which also catches cases one camera cannot resolve (the helmet of a worker hidden behind another one).
- The client speaks the OpenAI-compatible chat API: a local Ollama server (default, free, offline), vLLM, LM Studio or a hosted API.

---

## Quick start

```bash
git clone https://github.com/MerouaBelhi/edge-perception-agent.git
cd edge-perception-agent
pip install -r requirements.txt
# model files: download site11hh_v2.onnx and site11hh_v2_int8.onnx from the Releases page into models/

# still images: per-worker decisions drawn on each image
python scripts/detect_images.py --source path/to/images --model models/site11hh_v2.onnx --out results/images

# video: tracking, zones, PPE rules and hazards
PYTHONPATH=src python -m epa.pipeline --source site.mp4 --model models/site11hh_v2.onnx --imgsz 960 \
    --zones configs/site_safety_zones.json --safety --hazards --out results/site
#   (models/site11hh_v2_int8.onnx: 1.9x faster, 3.3x smaller, slightly less accurate: see Edge deployment)
#   -> results/site/annotated.mp4, events.jsonl, tracks.jsonl, violations.jsonl, evidence/

# questions about the run (needs Ollama: ollama pull qwen2.5:7b && ollama pull qwen2.5vl:3b)
python -m epa.agent --run results/site
```

**Reproduce the model and the evaluation**

```bash
pip install fiftyone pyyaml
python scripts/prepare_site_safety.py --css datasets/css --dst datasets/site11     # + Open Images
python scripts/train_ppe.py --data datasets/site11/data.yaml --imgsz 640 --name site11
python scripts/train_ppe.py --data datasets/site11/data.yaml --weights models/site11.pt --imgsz 960 --name site11_960
python scripts/prepare_hardhat.py --site datasets/site11 --dst datasets/site11hh \
    --model models/site11_960.onnx --person-conf 0.3                                 # keeps 500 images out
python scripts/train_ppe.py --data datasets/site11hh/data.yaml --weights models/site11_960.pt \
    --imgsz 960 --epochs 40 --name site11hh_v2
python scripts/eval_safety.py --data datasets/site11 --model models/site11hh_v2.onnx --split test
python scripts/external_test.py --model models/site11hh_v2.onnx --holdout results/hardhat_holdout.txt
python scripts/quantize_int8.py --model models/site11hh_v2.onnx --calib datasets/site11/images/val --imgsz 960
```

### C++ inference engine

```bash
# ONNX Runtime release: https://github.com/microsoft/onnxruntime/releases
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-<version>
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
./cpp/build/epa_infer models/site11hh_v2_int8.onnx path/to/site.jpg results/cpp_site.jpg
./cpp/build/epa_infer models/site11hh_v2_int8.onnx path/to/site.mp4 results/cpp_site.mp4
```

The engine reads the input size and class names from the ONNX file, so the same binary runs any of the models. Windows (Visual Studio 2022): install the *Desktop development with C++* workload, OpenCV and ONNX Runtime, then `cmake -S cpp -B cpp/build -DOpenCV_DIR=C:/opencv/build -DONNXRUNTIME_DIR=C:/onnxruntime/<version>` and `cmake --build cpp/build --config Release`. The build copies `onnxruntime.dll` next to the `.exe`, because Windows ships an older one in System32 that would otherwise be loaded and fail.

### REST API (Docker)

```bash
docker build -t edge-perception-agent .
docker run -p 8000:8000 edge-perception-agent
curl -F file=@path/to/site.jpg "localhost:8000/detect?conf=0.3"
curl -F file=@path/to/site.jpg localhost:8000/detect/annotated -o out.jpg
```

Interactive docs at `http://localhost:8000/docs`.

---

## Development history: what failed and how it was fixed

The released model is the fourth generation. The earlier steps are kept here because each one changed the design.

**1. First model: 3 classes, and a weak `no_helmet` class.** `ppe3` (person, helmet, no_helmet) was trained on [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) (Ultralytics, AGPL-3.0). Helmets were detected well (mAP50 0.94), bare heads poorly (0.14), and five times more training did not help (0.16 → 0.14): the dataset had about 400 bare heads, mostly in offices and sports scenes, so the model could partly learn "construction background → helmet". Relying on `no_helmet` boxes, the system caught only 36 % of bare heads. **Inferring a missing helmet from a clearly visible head** raised this to 89 % (validation split) without more false alarms, which is why step 3 above exists. Tighter matching rules (helmet centred above the person, shorter head region) were compared on the validation split and none improved the system, so they were not adopted ([`assets/safety_demo.gif`](assets/safety_demo.gif): this first model on a test video).

**2. Bugs found on videos and photos.**
- *Duplicate boxes.* A worker wearing a helmet was flagged: the detector gave two overlapping boxes for him, the helmet went to one, and the other "had no helmet". Absence is now inferred only when the head region contains no PPE box at all. A second case, one worker with two alerts, looked like a tracker ID switch; printing the per-frame tracks showed a second box nested on his left half with the same bottom edge. Boxes ≥ 70 % inside another one and touching the ground at the same height are now the same worker.
- *Resolution.* At 416 px, distant workers' helmets shrink to ~5 px and vanish while the workers still look large in a 1080p frame, so absence-inference flagged them. The "clearly visible" test now measures the person at the detector's input size: at low resolution the system says `unknown`.
- *Occlusion.* A bare-headed worker walking in front of helmeted colleagues was given their helmets for a moment, which re-armed his alert. A helmet must now be seen for 2 s before an alert re-arms, and voting uses a fixed 1.5 s window (15 frames was 0.5 s at 30 fps).
- *Crowds.* A bare-headed man close to the camera got the helmets of workers behind him: his large box made his head region large. Those helmets were only 0.11–0.26 of his width (a worker's own: ≥ 0.29), hence `--min-ppe-width`.
- *Vests on video.* Every worker wore a vest, yet some got alerts: the vest was detected in only 40–70 % of frames. Since a vest does not come off in 1.5 s, "no vest" now needs ≥ 90 % of the window without one (false alerts 6 → 2). On photos, half of the false vest alarms came from close-ups where the vest sits low in the box, hence the adaptive torso region (21 % → 12 %).
- *Machine proximity.* False alarms came from workers far behind a machine, operators in a cab and people whose feet were not visible, hence the ground-gap, footprint, cab and feet checks.

**3. Non-work headwear: `other_headwear`.** AI-generated test clips showed **a straw sun hat and a bicycle helmet accepted as hard hats**: a model that knows only "hard hat" and "bare head" takes anything rigid on a head for a helmet, and the VLM cannot catch an accepted helmet since only alerts are sent to it. `site11` adds an `other_headwear` class from Open Images (bicycle helmets, sun hats, cowboy hats, swim caps...; images with Open Images' mixed "Helmet" class are skipped), plus `no_vest`, `mask`, `no_mask` from Construction Site Safety and `machinery`, `vehicle`, `safety_cone`. Fine-tuning it at 960 px instead of 640 px (`site11_960`) recognised more hard hats (80 % → 83 %) and caught more bare heads (62 % → 68 %), at about twice the inference time.

**4. New sites: Hard Hat Workers** (section above): coloured hard hats, pseudo-labels, and the person threshold that made v2.

---

## Data and licenses

| Data | License | Used for |
|---|---|---|
| [Construction Site Safety](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) (Roboflow) | CC BY 4.0 | training and test (released model) |
| [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) | annotations CC BY 4.0, images CC BY 2.0 | non-work headwear and vehicles |
| [Hard Hat Workers](https://huggingface.co/datasets/Voxel51/hard-hat-detection) (Voxel51) | CC0 | training + 500 held-out images (external test) |
| [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) (Ultralytics) | AGPL-3.0 | first model only (`ppe3`), not the released one |

Training uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0), so this project, including the released models, is distributed under the same license.

---

## Project structure

```
src/epa/
  detector_onnx.py   hand-written ONNX Runtime inference (pre/post-processing, NMS)
  pipeline.py        video pipeline: detection -> ByteTrack -> zones -> safety rules -> outputs
  safety.py          hard hat / vest / mask association, inference from absence, voting, zone rules, evidence
  hazards.py         worker-machine proximity, machine motion, coned-off areas
  zones.py           polygon zones, enter/exit events
  agent.py           scene agent: tool schemas + LLM tool-calling loop
  scene_store.py     query layer over a run (the agent's tools)
  llm.py             OpenAI-compatible client (chat with tools, vision)
  api.py             FastAPI service
cpp/main.cpp         C++ inference engine (ONNX Runtime C++ API + OpenCV)
configs/             zones and the PPE each zone requires
scripts/
  prepare_site_safety.py  Construction Site Safety + Open Images -> 11 classes
  prepare_hardhat.py      Hard Hat Workers -> training, with a held-out part
  prepare_ppe_dataset.py  Construction-PPE -> first models (ppe3 / ppe4)
  train_ppe.py            fine-tune, test-split metrics, ONNX export
  eval_safety.py          per-worker evaluation of the hard hat / vest / mask decisions
  external_test.py        evaluation on the held-out Hard Hat Workers images
  compare_models.py       before / after per image, most improved and most degraded
  detect_images.py        rules on still images
  training_report.py      per-epoch table, loss curves, TensorBoard logs
  collect_assets.py       copy README figures and tables out of runs/ (git-ignored)
  export_onnx.py, quantize_int8.py, compare_int8.py, benchmark.py, latency_onnx.py
  concat_videos.py, to_h264.py, make_gif.py, zone_helper.py   video and zone utilities
tests/               unit tests (pytest)
docs/training/       per-epoch tables and evaluation results
```

---

## Roadmap

- [ ] Benchmarks on ARM (Raspberry Pi / Jetson) and with TensorRT
- [ ] Zone logic, safety rules and tracking in the C++ engine
- [ ] Ground-plane calibration (homography) for distances in metres
- [ ] More on-site bare heads, machines and cones in the training data; motorcycle helmets
- [ ] Pose estimation (YOLOv8-pose) for falls and unsafe postures
- [ ] Agent endpoint in the REST API

## License

[AGPL-3.0](LICENSE), the license of Ultralytics YOLOv8, which this project builds on. The datasets keep their own licenses (see *Data and licenses* above).
