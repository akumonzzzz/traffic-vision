"""Thin wrapper around an Ultralytics YOLO model.

Keeps the FastAPI layer free of any ultralytics/torch specifics so the model can
be swapped (pretrained -> fine-tuned) without changing the API.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One predicted box, in pixel coordinates of the submitted image."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["width"] = round(self.width, 1)
        d["height"] = round(self.height, 1)
        return d


class TrafficDetector:
    """Loads the model once and serves predictions."""

    def __init__(self, model_path: str = config.MODEL_PATH, device: str = config.DEVICE):
        self.model_path = model_path
        self.device = device
        log.info("Loading model %s on %s", model_path, device)
        self.model = YOLO(model_path)
        self.names: dict[int, str] = {int(k): v for k, v in self.model.names.items()}
        self.traffic_class_ids = self._resolve_traffic_ids()
        log.info("Model ready: %d classes, %d kept as traffic classes",
                 len(self.names), len(self.traffic_class_ids))

    def _resolve_traffic_ids(self) -> list[int]:
        """Map the configured traffic class names onto this model's label set.

        A COCO model matches 8 of them. A custom fine-tune trained only on
        vehicles will match few or none -- in that case every class is a
        traffic class, so we keep them all rather than filtering to nothing.
        """
        wanted = {n.lower() for n in config.TRAFFIC_CLASS_NAMES}
        matched = [i for i, n in self.names.items() if n.lower() in wanted]
        if not matched:
            log.info("No COCO traffic names in this model; keeping all classes")
            return sorted(self.names)
        return sorted(matched)

    def new_tracking_model(self) -> YOLO:
        """A private YOLO instance for one tracking session.

        Ultralytics stores tracker state on the model (``predictor.trackers``),
        and ``persist=True`` deliberately keeps it between calls. That makes the
        model object single-tenant: two sessions sharing one would inherit each
        other's frame counter and open tracks, so ids and counts from one user
        would leak into another's. Weights are small (yolo11n is ~6 MB) and load
        in tens of milliseconds, so a session gets its own instance.
        """
        return YOLO(self.model_path)

    def class_catalog(self) -> list[dict]:
        """Classes this deployment can report, for the UI's filter checkboxes."""
        return [
            {"id": i, "name": self.names[i], "color": self._color(self.names[i])}
            for i in self.traffic_class_ids
        ]

    @staticmethod
    def _color(name: str) -> tuple[int, int, int]:
        return config.CLASS_COLORS.get(name.lower(), config.FALLBACK_COLOR)

    def predict(
        self,
        image_bytes: bytes,
        conf: float = config.DEFAULT_CONF,
        iou: float = config.DEFAULT_IOU,
        classes: Iterable[int] | None = None,
    ) -> tuple[list[Detection], Image.Image, float]:
        """Run detection. Returns (detections, original RGB image, latency_ms)."""
        image = self._decode(image_bytes)
        frame = np.array(image)[:, :, ::-1]  # RGB -> BGR for ultralytics

        wanted = sorted(set(classes)) if classes else self.traffic_class_ids

        started = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=config.IMAGE_SIZE,
            classes=wanted,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        detections: list[Detection] = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self.names.get(cls_id, str(cls_id)),
                        confidence=round(float(box.conf.item()), 4),
                        x1=round(x1, 1),
                        y1=round(y1, 1),
                        x2=round(x2, 1),
                        y2=round(y2, 1),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections, image, latency_ms

    def _decode(self, image_bytes: bytes) -> Image.Image:
        """Bytes -> RGB PIL image, rejecting anything that is not an image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as 400
            raise ValueError("File is not a readable image") from exc
        return image.convert("RGB")

    def annotate(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        """Draw labelled boxes onto a copy of the image."""
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        # Scale line weight and text with image size so 4K and 640px both read well.
        scale = max(canvas.width, canvas.height) / 1000
        thickness = max(2, round(3 * scale))
        font = self._font(max(13, round(16 * scale)))

        for det in detections:
            color = self._color(det.class_name)
            draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=color, width=thickness)

            label = f"{det.class_name} {det.confidence:.0%}"
            tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, font=font)
            tw, th = tx1 - tx0, ty1 - ty0
            pad = max(3, round(4 * scale))

            # Put the label inside the box when it would run off the top edge.
            box_top = det.y1 - th - 2 * pad
            ly = box_top if box_top >= 0 else det.y1
            draw.rectangle([det.x1, ly, det.x1 + tw + 2 * pad, ly + th + 2 * pad], fill=color)
            draw.text((det.x1 + pad, ly + pad), label, fill=(255, 255, 255), font=font)

        return canvas

    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        for candidate in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def encode_jpeg(image: Image.Image, quality: int = 88) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def summarize(detections: list[Detection]) -> dict[str, int]:
    """Per-class counts, e.g. {'car': 11, 'truck': 2}, highest count first."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
