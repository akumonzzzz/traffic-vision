---
title: Traffic Vision
emoji: 🚦
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# Traffic Vision

Real-time traffic object detection and vehicle tracking. Detects vehicles,
pedestrians and road signage in **images, video clips, and live camera feeds**,
assigns each vehicle a persistent ID, and counts it as it crosses a virtual line.

**[▶ Live demo](https://huggingface.co/spaces/KaiVQ/traffic-vision)** ·
**[API docs](https://huggingface.co/spaces/KaiVQ/traffic-vision/docs)**

![The console: clip analysis with tracking and line counting](docs/console.jpg)

Every vehicle carries a persistent ID and a motion trail, and is counted once as
it crosses the line:

![Tracking with persistent IDs and a counting line](docs/tracking.jpg)

---

## Why tracking, not just detection

Detection answers *"what is in this frame"*. A traffic camera needs *"is that the
same car I saw a second ago"* — without identity you cannot count a vehicle
without counting it again on the next frame, and flow rate is impossible.

So every video and live path runs ByteTrack rather than plain prediction. Each
vehicle gets a stable ID, a motion trail, and is counted **once**, in the
direction it crosses the line. That virtual tripwire is the primitive real
traffic deployments are built on.

---

## Three input modes

| Mode | Endpoint | What it does |
|---|---|---|
| **Image** | `POST /api/detect` | Single frame → boxes, per-class counts, annotated JPEG |
| **Video** | `POST /api/video` | Clip → async job → annotated H.264 MP4 + crossing counts |
| **Live camera** | `WS /ws/live` | Browser webcam → per-frame detections streamed back |

Plus a headless CLI for real deployments:

```bash
python scripts/stream.py --source "rtsp://user:pass@10.0.0.20:554/stream1" --csv counts.csv
```

| | |
|---|---|
| **Model** | Ultralytics YOLO11s (COCO-pretrained, filtered to traffic classes) |
| **Classes** | person, bicycle, car, motorcycle, bus, truck, train, traffic light, stop sign |
| **Tracker** | ByteTrack, one isolated instance per session |
| **Image inference** | ~93 ms median, CPU-only, 640 px |
| **Video throughput** | ~17 fps, CPU-only, 960 px |
| **Live throughput** | ~12–15 fps per session, two concurrent sessions |
| **Serving** | FastAPI + Uvicorn, REST + WebSocket |
| **Container** | Python 3.12-slim, CPU-only torch (2.29 GB image, measured) |

---

## Accuracy

Speed without accuracy says nothing about whether the boxes are right, so the
served weights are measured, not assumed. `scripts/evaluate.py` scores whatever
`MODEL_PATH` points at, restricted to the classes this service actually serves:

```bash
python scripts/evaluate.py                                  # your own dataset
python scripts/evaluate.py --data coco128.yaml --split train   # public smoke test
```

Stock `yolo11s.pt`, 640 px, `conf=0.001`, `iou=0.45`, CPU:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|--:|--:|--:|--:|--:|
| person | 254 | 0.868 | 0.699 | 0.824 | 0.586 |
| bicycle | 6 | 0.552 | 0.333 | 0.386 | 0.314 |
| car | 46 | 0.745 | 0.255 | 0.442 | 0.245 |
| motorcycle | 5 | 0.773 | 1.000 | 0.962 | 0.765 |
| bus | 7 | 0.902 | 0.714 | 0.766 | 0.700 |
| train | 3 | 0.797 | 1.000 | 0.995 | 0.819 |
| truck | 12 | 0.838 | 0.432 | 0.575 | 0.338 |
| traffic light | 14 | 0.682 | 0.143 | 0.220 | 0.185 |
| stop sign | 2 | 0.781 | 1.000 | 0.995 | 0.846 |
| **All** | **349** | **0.771** | **0.620** | **0.685** | **0.533** |

### Why `yolo11s` and not `yolo11n`

The first measurement ran on `yolo11n` and put `car` recall at 0.17 — the weakest
vehicle class and the most important one in a traffic detector. Rather than guess
at a fix, all three candidate levers were measured against the same split:

| Model | imgsz | car recall | car mAP50 | all mAP50 | latency |
|---|--:|--:|--:|--:|--:|
| yolo11n | 640 | 0.174 | 0.241 | 0.642 | 72 ms |
| yolo11n | 1280 | 0.304 | 0.432 | 0.571 | 181 ms |
| **yolo11s** | **640** | **0.255** | **0.442** | **0.685** | **93 ms** |
| yolo11s | 1280 | 0.408 | 0.550 | 0.635 | 343 ms |

`yolo11s` at 640 px wins on every accuracy measure for 21 ms and 14 MB. It is now
the default.

Two results in that table are worth reading carefully. Raising `imgsz` lifts `car`
sharply but *drops* overall mAP50 — because `person` supplies 254 of the 349
instances, so the "all" column is largely a person score, and upscaling past the
images' native resolution hurts the classes that were already resolved. And
`yolo11s` at 1280 has the best `car` recall of all, 0.408, but costs 3.7× the
latency for it; that trade is available via `IMAGE_SIZE=1280` for anyone who wants
recall over speed.

**What this measures, and what it does not.** These figures come from
`coco128` — 128 general COCO photos, not road scenes — because no labelled
traffic set ships with this repo. They describe stock weights on COCO's
distribution and are a floor, not a claim about camera footage. Several classes
carry only a handful of instances, far too few to be stable: `stop sign` scoring
0.995 on two instances means almost nothing.

`traffic light` remains weak at 0.143 recall, and `car` at 0.255 is still the
ceiling-setter. Both are dominated by small, distant instances. Fine-tuning on
real road data is the remaining lever, and the only one that would make these
numbers describe traffic rather than COCO.

Replacing these numbers with ones from real traffic footage is the single highest-value
change to this repo. See [docs/DATASET.md](docs/DATASET.md).

---

## Quick start

### Docker

```bash
docker compose up --build
```

Open <http://localhost:7860>.

### Local Python

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

---

## API

Interactive docs at `/docs`.

### Image — `POST /api/detect`

```bash
curl -X POST http://localhost:7860/api/detect \
  -F "file=@road.jpg" -F "conf=0.35" -F "classes=2,5,7"
```

```json
{
  "count": 8,
  "counts_by_class": { "car": 7, "truck": 1 },
  "detections": [
    { "class_id": 2, "class_name": "car", "confidence": 0.5955,
      "x1": 988.3, "y1": 252.3, "x2": 1049.7, "y2": 299.0,
      "width": 61.4, "height": 46.7 }
  ],
  "image": { "width": 1566, "height": 876 },
  "inference_ms": 91.7,
  "model": "yolo11s.pt"
}
```

`POST /api/detect/image` returns the annotated JPEG directly, for
`curl ... --output result.jpg`.

### Video — `POST /api/video`

Returns a job id immediately; a 30 s clip takes ~40 s on CPU, well past any
sensible HTTP timeout.

```bash
JOB=$(curl -s -X POST http://localhost:7860/api/video \
  -F "file=@traffic.mp4" -F "stride=2" -F "line_y=0.6" | jq -r .job_id)

curl -s "http://localhost:7860/api/video/$JOB"            # poll
curl -s "http://localhost:7860/api/video/$JOB/result" -o annotated.mp4
```

Poll until `state` is `done` or `failed`:

```json
{
  "state": "done", "progress": 1.0, "frames_done": 90, "elapsed_s": 5.8,
  "stats": {
    "frames": 90, "width": 768, "height": 428,
    "unique_objects": { "car": 10 },
    "peak_concurrent": 5,
    "counts": { "total_in": 3, "total_out": 5,
                "by_class": { "car": { "in": 3, "out": 5 } } }
  },
  "result_url": "/api/video/.../result"
}
```

### Live — `WS /ws/live`

Send a JSON config message, then alternate binary JPEG frames with JSON replies.

```javascript
ws.send(JSON.stringify({ conf: 0.35, line_y: 0.5, classes: [2, 5, 7] }));
// -> {"ack": true}
ws.send(jpegBytes);
// -> {"frame":1,"tracks":[{"id":3,"name":"car","conf":0.71,
//      "x":0.42,"y":0.31,"w":0.08,"h":0.06}],
//     "counts":{"car":6},"line":{"total_in":2,"total_out":1},
//     "unique_total":9,"inference_ms":68.2}
```

Box coordinates are normalised 0–1 so the client can scale them to any display size.

**The client must wait for a reply before sending the next frame.** That single
rule is the backpressure mechanism — without it a 60 Hz camera queues frames
faster than a CPU box can clear them and the overlay drifts seconds behind
reality. `{"reset": true}` clears tracker state and counters.

### Ops — `GET /api/health`, `GET /api/classes`

Health reports the loaded model, device, live-session capacity and job counts.

---

## Configuration

Every setting is an environment variable; swapping models needs no code change.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `yolo11s.pt` | Weights to serve — point at your own fine-tune |
| `DEFAULT_CONF` / `DEFAULT_IOU` | `0.35` / `0.45` | Detection thresholds |
| `IMAGE_SIZE` | `640` | Inference resolution |
| `DEVICE` | `cpu` | `cpu`, or `0` for the first CUDA device |
| `INFER_WORKERS` | `2` | Inference thread-pool size |
| `MAX_UPLOAD_BYTES` | `10 MB` | Image upload cap |
| `MAX_VIDEO_BYTES` | `60 MB` | Video upload cap |
| `VIDEO_MAX_WIDTH` | `960` | Frames are downscaled to this before inference |
| `VIDEO_MAX_FRAMES` | `900` | Hard cap per clip |
| `VIDEO_JOB_TTL_S` | `1800` | How long results are kept before cleanup |
| `MAX_LIVE_SESSIONS` | `4` | Concurrent websocket streams |
| `LIVE_MAX_WIDTH` | `960` | Live frames downscaled to this |
| `NIGHT_MODE` | `auto` | Low-light boost: `auto`, `on`, `off` |
| `NIGHT_GAMMA` | `0.6` | Brightening curve; lower is stronger |
| `NIGHT_CLAHE` | `0` | Local contrast. Off by default — it *cost* detections when measured |
| `TRACKER` | `bytetrack.yaml` | `botsort.yaml` handles fast motion better, slower |

---

## Project layout

```
app/
  config.py        environment-driven settings
  detector.py      stateless single-image detection
  video.py         tracking, line counting, ffmpeg encoding
  jobs.py          in-memory job store with TTL cleanup
  main.py          REST + WebSocket routes
  static/          browser console (image / video / live)
scripts/
  stream.py        headless RTSP / webcam / file runner
  autolabel_sam.py SAM 2.1 point-prompts → YOLO labels
  train.py         fine-tuning entry point
  evaluate.py      mAP / precision / recall for the served weights
tests/             64 tests: API contracts, tracking, counting, encoding
```

---

## Design notes

Three decisions that are not obvious from the code:

**One model per tracking session.** Ultralytics stores tracker state on the
model object, and `persist=True` deliberately keeps it between calls. That makes
the model single-tenant: two sessions sharing one inherit each other's frame
counter and open tracks, so the second silently loses its first frame and its
counts are polluted by the first session's vehicles. Each session therefore gets
its own instance from `detector.new_tracking_model()` — weights are ~6 MB and
load in tens of milliseconds. Concurrent live sessions are capped so this cannot
grow without bound.

**ffmpeg, not `cv2.VideoWriter`.** The codecs OpenCV can reach in a slim
container (`mp4v`) produce files Chrome and Safari refuse to play inline.
`imageio-ffmpeg` ships a static ffmpeg with libx264 on every platform, so output
plays in the browser and downloads cleanly.

**The inference pool is owned by the app lifespan, not the module.** A
module-level `ThreadPoolExecutor` is shut down by the first app teardown and can
never be used again, which breaks any second startup in the same process.

---

## Training your own weights

The served model is COCO-pretrained. To fine-tune:

1. Put images in `dataset/images/{train,val}/` and YOLO labels in
   `dataset/labels/{train,val}/`. **The trees must mirror each other** —
   Ultralytics locates labels by replacing `/images/` with `/labels/` in the
   image path, so a `.txt` beside the `.jpg` is silently ignored.
2. Update the class list in `dataset/data.yaml`.
3. Train on a GPU (CPU training is impractical):
   ```bash
   python scripts/train.py --model yolo11s.pt --epochs 100 --batch 16 --device 0
   ```
4. Serve it: `MODEL_PATH=weights/traffic-yolo.pt uvicorn app.main:app`

`scripts/autolabel_sam.py` turns one click per object into a bounding box via
SAM 2.1. See [docs/DATASET.md](docs/DATASET.md) for layout rules and sanity checks.

For how the pieces fit together, how to change them, and how to deploy a change,
see the [development guide](docs/DEVELOPMENT.md).

---

## Known limitations

Stated plainly, because they are the honest state of the project:

- **No accuracy metrics are published.** The COCO-pretrained model has not been
  evaluated against a labelled traffic-specific validation set, so no mAP is
  quoted. Publishing an unvalidated number would be worse than publishing none.
- **Small and distant objects are missed.** YOLO11n at 640 px drops vehicles
  beyond roughly 150 m. Raising `IMAGE_SIZE` to 1280 or moving to `yolo11s`
  recovers most of them at proportionally higher latency.
- **`truck` / `bus` / `car` confusion** on vans and SUVs, inherited from COCO's
  labelling conventions. A good reason to fine-tune.
- **Line counting assumes crossing traffic.** A camera pointed along the road
  rather than across it will undercount, because vehicles travel parallel to the
  line instead of through it.
- **Tracking loses identity through long occlusions.** ByteTrack recovers short
  gaps; a vehicle hidden behind a truck for several seconds returns as a new ID
  and is counted twice.
- **CPU inference.** Fine for images and short clips. Live streaming holds
  ~12–15 fps per session; real-time HD video needs a GPU.
- **Night performance is improved, not solved.** A gamma boost before inference
  recovers a meaningful share of lost detections (7 → 9 on a simulated dark
  frame), but COCO is a daytime dataset and the underlying weights have never
  seen a night road. Only fine-tuning fixes that properly.
- **Rail transport is heavy rail only.** COCO has a `train` class, now enabled,
  but no class for trams, light rail or on-street metro — those are typically
  misread as `bus`. Cityscapes and Mapillary Vistas both carry an on-rails class
  and are the datasets to fine-tune on.
- **Very fast objects still break tracking.** When a vehicle moves further
  between frames than its own box width, IoU association fails and it is issued a
  new ID, which double-counts it. `TRACKER=botsort.yaml` compensates for motion
  and helps; a higher frame rate helps more.

---

## Roadmap

- [ ] Fine-tune on a labelled traffic dataset and publish mAP50-95
- [ ] Speed estimation from track displacement and a calibrated ground plane
- [ ] User-drawn counting lines and polygon regions instead of a horizontal line
- [ ] ONNX export for faster CPU inference
- [ ] Persist counts to a time-series store for traffic-flow dashboards

---
## License

Copyright (C) 2026 Duy Vuong Quang

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See [LICENSE](LICENSE) for the full text.

AGPL is not a free choice here: it is inherited from
[Ultralytics](https://github.com/ultralytics/ultralytics), which this project
depends on for inference. AGPL's network clause covers software served over a
network, so a public deployment must offer its source — which is what this
repository is. For a permissive licence, replace the Ultralytics dependency with
an ONNX export served through `onnxruntime`, or obtain an Ultralytics Enterprise
License.
