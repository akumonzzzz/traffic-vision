"""FastAPI service exposing the traffic detector as a JSON API + demo UI."""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config
from .detector import TrafficDetector, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("traffic-detector")

state: dict[str, TrafficDetector] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load weights once at startup so the first request is not the slow one.
    state["detector"] = TrafficDetector()
    yield
    state.clear()


app = FastAPI(
    title="Traffic Object Detection API",
    description="YOLO-based detection of vehicles, pedestrians and road signage.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_detector() -> TrafficDetector:
    detector = state.get("detector")
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return detector


def _parse_classes(raw: str | None, detector: TrafficDetector) -> list[int] | None:
    """Parse the `classes` form field: '2,5,7' -> [2, 5, 7]. Empty -> None."""
    if not raw or not raw.strip():
        return None
    try:
        ids = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise HTTPException(
            status_code=400, detail="`classes` must be comma-separated integers"
        ) from None

    unknown = [i for i in ids if i not in detector.names]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown class ids: {unknown}")
    return ids or None


@app.get("/api/health", tags=["ops"])
def health():
    """Liveness + which model is actually loaded. Useful in the README and in CI."""
    detector = state.get("detector")
    return {
        "status": "ok" if detector else "loading",
        "model": config.MODEL_PATH,
        "device": config.DEVICE,
        "image_size": config.IMAGE_SIZE,
        "classes": len(detector.names) if detector else 0,
    }


@app.get("/api/classes", tags=["model"])
def classes():
    """Classes this deployment reports, with their box colours."""
    return {"classes": get_detector().class_catalog()}


@app.post("/api/detect", tags=["inference"])
async def detect(
    file: UploadFile = File(..., description="JPEG or PNG road scene"),
    conf: float = Form(config.DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Form(config.DEFAULT_IOU, ge=0.0, le=1.0),
    classes: str | None = Form(None, description="Comma-separated class ids to keep"),
    annotate: bool = Form(True, description="Include a base64 annotated JPEG"),
):
    """Detect traffic objects and return boxes, per-class counts and latency."""
    detector = get_detector()

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > config.MAX_UPLOAD_BYTES:
        limit_mb = config.MAX_UPLOAD_BYTES / 1_048_576
        raise HTTPException(status_code=413, detail=f"Image exceeds {limit_mb:.0f} MB limit")

    wanted = _parse_classes(classes, detector)

    try:
        detections, image, latency_ms = detector.predict(
            payload, conf=conf, iou=iou, classes=wanted
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = {
        "count": len(detections),
        "counts_by_class": summarize(detections),
        "detections": [d.to_dict() for d in detections],
        "image": {"width": image.width, "height": image.height},
        "inference_ms": round(latency_ms, 1),
        "model": config.MODEL_PATH,
    }

    if annotate:
        jpeg = detector.encode_jpeg(detector.annotate(image, detections))
        body["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()

    return body


@app.post("/api/detect/image", tags=["inference"])
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(config.DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Form(config.DEFAULT_IOU, ge=0.0, le=1.0),
    classes: str | None = Form(None),
):
    """Same as /api/detect but responds with the annotated JPEG directly.

    Handy for `curl ... --output result.jpg` and for embedding in an <img> tag.
    """
    detector = get_detector()

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > config.MAX_UPLOAD_BYTES:
        limit_mb = config.MAX_UPLOAD_BYTES / 1_048_576
        raise HTTPException(status_code=413, detail=f"Image exceeds {limit_mb:.0f} MB limit")

    wanted = _parse_classes(classes, detector)

    try:
        detections, image, _ = detector.predict(payload, conf=conf, iou=iou, classes=wanted)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    jpeg = detector.encode_jpeg(detector.annotate(image, detections))
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"X-Detection-Count": str(len(detections))},
    )


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
