"""FastAPI service exposing the traffic detector as a JSON API + demo UI."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, video
from .detector import TrafficDetector, summarize
from .jobs import JobState, JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("traffic-detector")

state: dict = {}

# Inference is CPU-bound and releases the GIL inside torch, so it runs in a
# thread pool rather than blocking the event loop. Two workers: enough to keep
# a live viewer and a video job from starving each other, few enough that they
# do not thrash a 2-core free-tier box.
#
# The pool is created per-lifespan rather than at import. A module-level pool is
# shut down by the first app teardown and can never be used again, which breaks
# any second startup in the same process (tests, embedding, reload).
INFER_WORKERS = int(os.getenv("INFER_WORKERS", "2"))


def get_executor() -> ThreadPoolExecutor:
    executor = state.get("executor")
    if executor is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return executor

# Live sessions each own a model instance (see TrackedStream), so they are
# counted and capped rather than left unbounded.
LIVE_SESSIONS = itertools.count()
_live_lock = threading.Lock()
_live_active = 0


def _acquire_live_slot() -> bool:
    global _live_active
    with _live_lock:
        if _live_active >= config.MAX_LIVE_SESSIONS:
            return False
        _live_active += 1
        return True


def _release_live_slot() -> None:
    global _live_active
    with _live_lock:
        _live_active = max(0, _live_active - 1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load weights once at startup so the first request is not the slow one.
    state["detector"] = TrafficDetector()
    state["jobs"] = JobStore()
    state["executor"] = ThreadPoolExecutor(
        max_workers=INFER_WORKERS, thread_name_prefix="infer"
    )
    yield
    state["executor"].shutdown(wait=False, cancel_futures=True)
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


def get_jobs() -> JobStore:
    jobs = state.get("jobs")
    if jobs is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return jobs


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
        "night_mode": config.NIGHT_MODE,
        "tracker": config.TRACKER,
        "live_sessions": {"active": _live_active, "max": config.MAX_LIVE_SESSIONS},
        "jobs": state["jobs"].stats() if state.get("jobs") else None,
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
    night_mode: str = Form("auto", pattern="^(auto|on|off)$",
                           description="Low-light boost: auto, on, or off"),
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
        detections, image, latency_ms, enhanced = detector.predict(
            payload, conf=conf, iou=iou, classes=wanted, night_mode=night_mode
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
        "low_light_boost": enhanced,
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
        detections, image, _, _ = detector.predict(
            payload, conf=conf, iou=iou, classes=wanted
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    jpeg = detector.encode_jpeg(detector.annotate(image, detections))
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"X-Detection-Count": str(len(detections))},
    )


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _run_video_job(job_id: str, src, dst, classes, conf, iou, stride, line,
                   tracker=None, night_mode=None):
    """Executed on a worker thread; owns the job's lifecycle end to end."""
    jobs = state.get("jobs")
    detector = state.get("detector")
    if jobs is None or detector is None:
        return

    jobs.update(job_id, state=JobState.RUNNING, message="Tracking objects")
    try:
        stats = video.process_video(
            detector, src, dst,
            classes=classes, conf=conf, iou=iou, stride=stride,
            max_frames=config.VIDEO_MAX_FRAMES,
            line=line, tracker=tracker, night_mode=night_mode,
            on_progress=lambda done, total: jobs.set_progress(job_id, done, total),
        )
        jobs.update(
            job_id,
            state=JobState.DONE,
            progress=1.0,
            message="Complete",
            stats=vars(stats),
            result_path=dst,
            finished_at=time.time(),
        )
        log.info("Job %s done: %d frames", job_id, stats.frames)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via job state
        log.exception("Job %s failed", job_id)
        jobs.update(
            job_id,
            state=JobState.FAILED,
            error=str(exc),
            message="Failed",
            finished_at=time.time(),
        )
    finally:
        # The upload is only needed during processing.
        with contextlib.suppress(OSError):
            src.unlink(missing_ok=True)


@app.post("/api/video", tags=["video"])
async def submit_video(
    background: BackgroundTasks,
    file: UploadFile = File(..., description="MP4/MOV/AVI/MKV clip"),
    conf: float = Form(config.DEFAULT_CONF, ge=0.0, le=1.0),
    iou: float = Form(config.DEFAULT_IOU, ge=0.0, le=1.0),
    classes: str | None = Form(None),
    stride: int = Form(1, ge=1, le=10, description="Process every Nth frame"),
    line_y: float = Form(0.5, ge=0.0, le=1.0, description="Counting line height, 0-1"),
    tracker: str = Form("bytetrack.yaml", pattern=r"^(bytetrack|botsort)\.yaml$"),
    night_mode: str = Form("auto", pattern="^(auto|on|off)$"),
):
    """Queue a clip for tracking. Returns a job id to poll.

    Processing is asynchronous because a 30 s clip takes ~40 s on CPU, well past
    any sensible HTTP timeout.
    """
    detector = get_detector()
    jobs = get_jobs()

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > config.MAX_VIDEO_BYTES:
        limit_mb = config.MAX_VIDEO_BYTES / 1_048_576
        raise HTTPException(status_code=413, detail=f"Video exceeds {limit_mb:.0f} MB limit")

    wanted = _parse_classes(classes, detector)

    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix}")

    job = jobs.create()
    job_dir = jobs.dir_for(job)
    src = job_dir / ("input" + suffix)
    dst = job_dir / "annotated.mp4"
    src.write_bytes(payload)

    # Reject unreadable uploads now, while we can still answer with a 400.
    try:
        width, height, fps, frames = video.probe(src)
    except ValueError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    jobs.update(
        job.id,
        source_path=src,
        message="Queued",
        frames_total=min(frames // stride, config.VIDEO_MAX_FRAMES) if frames else 0,
    )

    line = ((0.0, line_y), (1.0, line_y))
    background.add_task(
        _submit_to_pool, job.id, src, dst, wanted, conf, iou, stride, line,
        tracker, night_mode,
    )

    body = job.to_dict()
    body["source"] = {"width": width, "height": height, "fps": round(fps, 2), "frames": frames}
    if frames and frames // stride > config.VIDEO_MAX_FRAMES:
        body["notice"] = (
            "Clip will be truncated to " + str(config.VIDEO_MAX_FRAMES) + " processed frames"
        )
    return body


def _submit_to_pool(*args) -> None:
    """Hand the job to the inference pool without awaiting it."""
    executor = state.get("executor")
    if executor is None:
        log.error("Executor gone; dropping job %s", args[0])
        return
    executor.submit(_run_video_job, *args)


@app.get("/api/video/{job_id}", tags=["video"])
def video_status(job_id: str):
    """Poll a job. Clients should stop polling on state 'done' or 'failed'."""
    job = get_jobs().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return job.to_dict()


@app.get("/api/video/{job_id}/result", tags=["video"])
def video_result(job_id: str):
    """The annotated MP4. Served inline so the browser can play it directly."""
    job = get_jobs().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    if job.state is not JobState.DONE or not job.result_path:
        raise HTTPException(status_code=409, detail="Job is " + job.state.value + ", not done")
    if not job.result_path.exists():
        raise HTTPException(status_code=410, detail="Result has been cleaned up")
    return FileResponse(
        job.result_path,
        media_type="video/mp4",
        filename="traffic-" + job_id + ".mp4",
    )


# ---------------------------------------------------------------------------
# Live camera
# ---------------------------------------------------------------------------


@app.websocket("/ws/live")
async def live_stream(ws: WebSocket):
    """Per-frame detection for a live camera feed.

    Protocol: the client sends a JSON config message, then alternates binary
    JPEG frames with JSON replies. The client must wait for a reply before
    sending the next frame -- that single rule is the backpressure mechanism,
    and it stops a fast camera queueing work a CPU box will never catch up on.
    """
    await ws.accept()

    detector = state.get("detector")
    if detector is None:
        await ws.close(code=1013, reason="Model still loading")
        return

    if not _acquire_live_slot():
        await ws.send_json({
            "error": f"Server is at capacity ({config.MAX_LIVE_SESSIONS} live "
                     "sessions). Try again shortly."
        })
        await ws.close(code=1013, reason="Too many live sessions")
        return

    session = next(LIVE_SESSIONS)
    log.info("Live session %d opened (%d/%d slots)",
             session, _live_active, config.MAX_LIVE_SESSIONS)

    executor = state.get("executor")
    if executor is None:
        _release_live_slot()
        await ws.close(code=1013, reason="Service is still starting")
        return

    stream = video.TrackedStream(detector)
    loop = asyncio.get_running_loop()
    frame_no = 0

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Text messages reconfigure the running stream.
            text = message.get("text")
            if text is not None:
                try:
                    cfg = json.loads(text)
                except json.JSONDecodeError:
                    await ws.send_json({"error": "Malformed config"})
                    continue

                if cfg.get("reset"):
                    stream = video.TrackedStream(detector)
                if cfg.get("conf") is not None:
                    stream.conf = max(0.01, min(float(cfg["conf"]), 0.99))
                if cfg.get("iou") is not None:
                    stream.iou = max(0.01, min(float(cfg["iou"]), 0.99))
                if cfg.get("classes") is not None:
                    valid = [int(i) for i in cfg["classes"] if int(i) in detector.names]
                    stream.classes = valid or detector.traffic_class_ids
                if cfg.get("night_mode") in {"auto", "on", "off"}:
                    stream.night_mode = cfg["night_mode"]
                if cfg.get("line_y") is not None:
                    y = float(cfg["line_y"])
                    stream.counter = video.LineCounter((0.0, y), (1.0, y))
                await ws.send_json({"ack": True})
                continue

            data = message.get("bytes")
            if not data:
                continue
            if len(data) > config.LIVE_MAX_FRAME_BYTES:
                await ws.send_json({"error": "Frame too large"})
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"error": "Undecodable frame"})
                continue

            if frame.shape[1] > config.LIVE_MAX_WIDTH:
                ratio = config.LIVE_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(
                    frame, (config.LIVE_MAX_WIDTH, int(frame.shape[0] * ratio)),
                    interpolation=cv2.INTER_AREA,
                )

            started = loop.time()
            tracks = await loop.run_in_executor(executor, stream.process, frame)
            latency_ms = (loop.time() - started) * 1000
            frame_no += 1

            height, width = frame.shape[:2]
            await ws.send_json({
                "frame": frame_no,
                # Normalised so the client can scale boxes to its own video size.
                "tracks": [
                    {
                        "id": t.track_id,
                        "name": t.class_name,
                        "conf": round(t.confidence, 3),
                        "x": round(t.x1 / width, 5),
                        "y": round(t.y1 / height, 5),
                        "w": round((t.x2 - t.x1) / width, 5),
                        "h": round((t.y2 - t.y1) / height, 5),
                    }
                    for t in tracks
                ],
                "counts": summarize_names([t.class_name for t in tracks]),
                "line": stream.counter.as_dict(),
                "unique_total": len(stream.seen_ids),
                "low_light_boost": stream.enhanced_frames > 0,
                "inference_ms": round(latency_ms, 1),
            })
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - never let one client kill the worker
        log.exception("Live stream error")
        with contextlib.suppress(RuntimeError):
            await ws.close(code=1011)
    finally:
        _release_live_slot()
        log.info("Live session %d closed after %d frame(s)", session, frame_no)


def summarize_names(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
