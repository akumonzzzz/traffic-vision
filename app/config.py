"""Runtime configuration, all overridable via environment variables."""

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"

# Which weights to serve. Point this at your own fine-tuned .pt to swap models
# without touching any other code:  MODEL_PATH=weights/traffic_yolo11s.pt
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt")

# Inference defaults. The API accepts per-request overrides.
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.35"))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", "0.45"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
DEVICE = os.getenv("DEVICE", "cpu")

# Reject oversized uploads before they reach the model.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# COCO class ids that are meaningful on a road scene. When the model is a
# custom fine-tune whose names do not match these, TRAFFIC_CLASSES is ignored
# and every class the model knows is kept (see detector.resolve_class_filter).
TRAFFIC_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)

# Stable colour per class so the same vehicle type is always the same colour.
CLASS_COLORS = {
    "person": (239, 68, 68),
    "bicycle": (168, 85, 247),
    "car": (34, 197, 94),
    "motorcycle": (249, 115, 22),
    "bus": (59, 130, 246),
    "truck": (234, 179, 8),
    "traffic light": (236, 72, 153),
    "stop sign": (14, 165, 233),
}
FALLBACK_COLOR = (148, 163, 184)
