---
title: Traffic Object Detection
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# Traffic Object Detection

Detects vehicles, pedestrians and road signage in traffic imagery. Ships as a
containerised FastAPI service with a REST API and a browser UI.

**[▶ Live demo](https://huggingface.co/spaces/YOUR_HF_USERNAME/traffic-object-detection)** ·
**[API docs](https://huggingface.co/spaces/YOUR_HF_USERNAME/traffic-object-detection/docs)**

![Detection output](docs/demo.jpg)

---

## What it does

Upload a road scene and get back bounding boxes, per-class counts, and inference
latency. Eight classes are reported: `person`, `bicycle`, `car`, `motorcycle`,
`bus`, `truck`, `traffic light`, `stop sign`.

Confidence threshold, IoU and the active class set are all adjustable per request,
so you can watch precision/recall trade off in real time.

| | |
|---|---|
| **Model** | Ultralytics YOLO11n (COCO-pretrained, filtered to traffic classes) |
| **Inference** | ~70–90 ms per image, CPU-only, 640 px |
| **Serving** | FastAPI + Uvicorn |
| **Container** | Python 3.12-slim, CPU-only torch (~1 GB image) |
| **Deploy** | Hugging Face Spaces (Docker SDK) |

---

## Quick start

### Docker (recommended)

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

Interactive docs are served at `/docs`.

### `POST /api/detect` → JSON

```bash
curl -X POST http://localhost:7860/api/detect \
  -F "file=@road.jpg" \
  -F "conf=0.35" \
  -F "classes=2,5,7"
```

```json
{
  "count": 8,
  "counts_by_class": { "car": 7, "truck": 1 },
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.6012,
      "x1": 988.4, "y1": 243.1, "x2": 1051.7, "y2": 292.6,
      "width": 63.3, "height": 49.5
    }
  ],
  "image": { "width": 1566, "height": 876 },
  "inference_ms": 91.7,
  "model": "yolo11n.pt"
}
```

### `POST /api/detect/image` → annotated JPEG

```bash
curl -X POST http://localhost:7860/api/detect/image \
  -F "file=@road.jpg" --output result.jpg
```

### `GET /api/health` · `GET /api/classes`

Health reports the loaded model and device. Classes lists what this deployment
reports, with box colours.

| Field | Default | Notes |
|---|---|---|
| `file` | — | JPEG or PNG, max 10 MB |
| `conf` | `0.35` | Confidence threshold, 0–1 |
| `iou` | `0.45` | NMS overlap threshold, 0–1 |
| `classes` | all traffic classes | Comma-separated class ids, e.g. `2,5,7` |
| `annotate` | `true` | Include a base64 annotated JPEG in the response |

---

## Configuration

Every setting is an environment variable, so swapping models needs no code change.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `yolo11n.pt` | Weights to serve — point at your own fine-tune |
| `DEFAULT_CONF` | `0.35` | Confidence threshold |
| `DEFAULT_IOU` | `0.45` | NMS IoU threshold |
| `IMAGE_SIZE` | `640` | Inference resolution |
| `DEVICE` | `cpu` | `cpu`, or `0` for the first CUDA device |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size cap |

---

## Project layout

```
app/
  config.py        environment-driven settings
  detector.py      YOLO wrapper — decode, predict, annotate
  main.py          FastAPI routes
  static/          browser demo UI
scripts/
  autolabel_sam.py SAM 2.1 point-prompts → YOLO labels
  train.py         fine-tuning entry point
dataset/data.yaml  Ultralytics dataset config
tests/test_api.py  API contract tests
```

---

## Training your own weights

The served model is COCO-pretrained. To fine-tune on your own traffic data:

1. Put images in `dataset/images/{train,val}/` and YOLO-format labels in
   `dataset/labels/{train,val}/`. **The two trees must mirror each other** —
   Ultralytics locates labels by replacing `/images/` with `/labels/` in the
   image path, so a `.txt` sitting next to the `.jpg` is silently ignored.
2. Update the class list in `dataset/data.yaml`.
3. Train (on a GPU — CPU training is impractical):
   ```bash
   python scripts/train.py --model yolo11s.pt --epochs 100 --batch 16 --device 0
   ```
4. Serve the result:
   ```bash
   MODEL_PATH=weights/traffic-yolo.pt uvicorn app.main:app
   ```

`scripts/autolabel_sam.py` speeds up labelling by turning a single click per
object into a bounding box via SAM 2.1:

```bash
python scripts/autolabel_sam.py --points points.json --split train
```

where `points.json` is `{"scene01.jpg": [[185, 421], [502, 487]], ...}`.

---

## Known limitations

Stated plainly, because they are the honest state of the current model:

- **Small/distant objects are missed.** YOLO11n at 640 px drops vehicles more
  than roughly 150 m out. Raising `IMAGE_SIZE` to 1280 or moving to `yolo11s`/`yolo11m`
  recovers most of them, at proportionally higher latency.
- **No accuracy metrics are published yet.** The COCO-pretrained model has not
  been evaluated on a labelled traffic-specific validation set, so no mAP is
  quoted here. Publishing an unvalidated number would be worse than publishing none.
- **`truck` vs `bus` vs `car` confusion** on vans and SUVs — inherited from COCO's
  labelling conventions, and a good reason to fine-tune.
- **Images only.** No video or multi-object tracking yet.
- **CPU inference.** Fine for single images; a GPU is needed for video framerates.

---

## Roadmap

- [ ] Fine-tune on a labelled traffic dataset and publish mAP50-95
- [ ] Video upload with ByteTrack multi-object tracking
- [ ] ONNX export for faster CPU inference
- [ ] Vehicle counting across a user-drawn line

---

## License

AGPL-3.0, inherited from [Ultralytics](https://github.com/ultralytics/ultralytics),
which this project depends on for inference. AGPL's network clause covers software
served over a network, so a public deployment of this app must offer its source —
which is what this repository is.

If you need a permissive license, swap the Ultralytics dependency for an ONNX
export served through `onnxruntime`, or obtain an Ultralytics Enterprise License.
