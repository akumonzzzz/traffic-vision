"""Video and live-stream processing.

The difference between this and ``detector.py`` is *identity*. Detection alone
answers "what is in this frame"; a traffic camera needs "is that the same car I
saw a second ago", which is what enables counting, flow rate and speed. So every
path in this module runs the tracker rather than plain prediction.
"""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    """One tracked object in one frame. ``track_id`` is stable across frames."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


def _side_of_line(point: tuple[float, float], a: tuple[float, float],
                  b: tuple[float, float]) -> float:
    """Signed area of triangle (a, b, point). Sign tells you which side."""
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


class LineCounter:
    """Counts tracks crossing a line, separated by direction of travel.

    This is the primitive real traffic deployments are built on: a virtual
    tripwire drawn across the lanes. A track is counted once, on the frame where
    the sign of its position relative to the line flips.

    Coordinates are normalised (0-1) so the line survives resolution changes.
    """

    def __init__(self, start=(0.0, 0.5), end=(1.0, 0.5)):
        self.start = start
        self.end = end
        self._last_side: dict[int, float] = {}
        self.counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0}
        )
        self.total_in = 0
        self.total_out = 0

    def update(self, tracks: list[Track], width: int, height: int) -> list[int]:
        """Returns the ids that crossed on this frame, for a visual pulse."""
        a = (self.start[0] * width, self.start[1] * height)
        b = (self.end[0] * width, self.end[1] * height)
        crossed: list[int] = []

        for track in tracks:
            side = _side_of_line(track.centroid, a, b)
            previous = self._last_side.get(track.track_id)
            self._last_side[track.track_id] = side

            # Need a previous observation, and a genuine sign flip. The epsilon
            # ignores tracks sitting exactly on the line jittering back and forth.
            if previous is None or abs(side) < 1e-6:
                continue
            if (previous > 0) == (side > 0):
                continue

            direction = "in" if side > 0 else "out"
            self.counts[track.class_name][direction] += 1
            if direction == "in":
                self.total_in += 1
            else:
                self.total_out += 1
            crossed.append(track.track_id)

        return crossed

    def as_dict(self) -> dict:
        return {
            "line": {"start": list(self.start), "end": list(self.end)},
            "total_in": self.total_in,
            "total_out": self.total_out,
            "by_class": {k: dict(v) for k, v in self.counts.items()},
        }


class TrackedStream:
    """Runs tracking over a sequence of frames, holding state between them.

    One instance per stream, and -- importantly -- one *model* per stream.
    ``persist=True`` tells Ultralytics to carry tracker state across calls so ids
    stay stable, but that state lives on the model object. Sharing a model
    between two streams therefore leaks one stream's frame counter and open
    tracks into the other: the second stream drops its first frame, and its
    vehicle counts are polluted by the first stream's tracks. So each stream
    gets a private model from ``detector.new_tracking_model()``.
    """

    def __init__(self, detector, classes: list[int] | None = None,
                 conf: float = config.DEFAULT_CONF, iou: float = config.DEFAULT_IOU,
                 tracker: str = "bytetrack.yaml"):
        self.detector = detector
        self.model = detector.new_tracking_model()
        self.classes = classes or detector.traffic_class_ids
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.counter = LineCounter()
        # Recent centroids per id, for motion trails.
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=24))
        self.seen_ids: set[int] = set()
        self.class_of_id: dict[int, str] = {}

    def process(self, frame_bgr: np.ndarray) -> list[Track]:
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=config.IMAGE_SIZE,
            device=self.detector.device,
            verbose=False,
        )

        tracks: list[Track] = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracks

        ids = boxes.id.int().tolist()
        for box, track_id in zip(boxes, ids, strict=False):
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            name = self.detector.names.get(cls_id, str(cls_id))
            tracks.append(
                Track(track_id, cls_id, name, float(box.conf.item()), x1, y1, x2, y2)
            )
            self.seen_ids.add(track_id)
            self.class_of_id[track_id] = name

        h, w = frame_bgr.shape[:2]
        self.counter.update(tracks, w, h)
        for track in tracks:
            self.trails[track.track_id].append(track.centroid)

        return tracks

    def unique_totals(self) -> dict[str, int]:
        """Unique objects seen for the whole stream, not per-frame counts."""
        totals: dict[str, int] = defaultdict(int)
        for name in self.class_of_id.values():
            totals[name] += 1
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def draw_overlay(frame_bgr: np.ndarray, tracks: list[Track], stream: TrackedStream,
                 show_line: bool = True, show_trails: bool = True) -> np.ndarray:
    """Burn boxes, ids, trails and the counting line into a frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 1000
    thickness = max(2, round(2.5 * scale))
    font_scale = max(0.45, 0.6 * scale)

    if show_line:
        counter = stream.counter
        a = (int(counter.start[0] * w), int(counter.start[1] * h))
        b = (int(counter.end[0] * w), int(counter.end[1] * h))
        cv2.line(out, a, b, (255, 255, 255), max(2, thickness))
        label = f"IN {counter.total_in}   OUT {counter.total_out}"
        cv2.putText(out, label, (a[0] + 10, max(28, a[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    max(1, thickness - 1), cv2.LINE_AA)

    for track in tracks:
        r, g, b_ = config.CLASS_COLORS.get(track.class_name.lower(), config.FALLBACK_COLOR)
        color = (b_, g, r)  # OpenCV is BGR

        if show_trails:
            points = list(stream.trails.get(track.track_id, ()))
            for i in range(1, len(points)):
                p0 = (int(points[i - 1][0]), int(points[i - 1][1]))
                p1 = (int(points[i][0]), int(points[i][1]))
                cv2.line(out, p0, p1, color, max(1, thickness - 1), cv2.LINE_AA)

        p1 = (int(track.x1), int(track.y1))
        p2 = (int(track.x2), int(track.y2))
        cv2.rectangle(out, p1, p2, color, thickness)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
        # Keep the label on-screen when the box touches the top edge.
        ty = p1[1] - th - baseline - 4
        if ty < 0:
            ty = p1[1] + 4
        cv2.rectangle(out, (p1[0], ty), (p1[0] + tw + 8, ty + th + baseline + 6), color, -1)
        cv2.putText(out, label, (p1[0] + 4, ty + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    max(1, thickness - 1), cv2.LINE_AA)

    return out


@dataclass
class VideoStats:
    frames: int = 0
    duration_s: float = 0.0
    fps_in: float = 0.0
    fps_processed: float = 0.0
    width: int = 0
    height: int = 0
    unique_objects: dict[str, int] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    peak_concurrent: int = 0


class FfmpegWriter:
    """Encodes BGR frames to H.264 MP4 by piping into the bundled ffmpeg.

    OpenCV's VideoWriter is avoided deliberately: the codecs it can reach in a
    slim container (mp4v) produce files Chrome and Safari refuse to play inline.
    imageio-ffmpeg ships a static ffmpeg with libx264 on every platform, so the
    output plays in the browser and downloads cleanly.
    """

    def __init__(self, path: Path, width: int, height: int, fps: float):
        # libx264 requires even dimensions.
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.path = path
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{max(fps, 1):.3f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # lets the browser start playing before full download
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = frame_bgr[: self.height, : self.width]
        self.proc.stdin.write(frame_bgr.astype(np.uint8).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            err = self.proc.stderr.read().decode(errors="replace")[-800:]
            raise RuntimeError(f"ffmpeg failed ({code}): {err}")


def iter_frames(path: Path, stride: int = 1, max_frames: int | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) from a video, skipping ``stride - 1`` frames each step."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video - unsupported or corrupt file")
    try:
        index = emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, frame
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            index += 1
    finally:
        cap.release()


def probe(path: Path) -> tuple[int, int, float, int]:
    """(width, height, fps, frame_count). frame_count can be 0 for some containers."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video - unsupported or corrupt file")
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def process_video(
    detector,
    src: Path,
    dst: Path,
    classes: list[int] | None = None,
    conf: float = config.DEFAULT_CONF,
    iou: float = config.DEFAULT_IOU,
    stride: int = 1,
    max_frames: int | None = None,
    line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> VideoStats:
    """Track through a video and write an annotated H.264 MP4 to ``dst``."""
    width, height, fps_in, total = probe(src)

    # Downscale big inputs: a 4K clip costs 4x the time for no accuracy gain at
    # a 640 px inference size.
    target_w = min(width, config.VIDEO_MAX_WIDTH)
    ratio = target_w / width if width else 1.0
    out_w, out_h = int(width * ratio), int(height * ratio)

    stream = TrackedStream(detector, classes=classes, conf=conf, iou=iou)
    if line:
        stream.counter = LineCounter(line[0], line[1])

    fps_out = fps_in / stride if stride > 1 else fps_in
    writer = FfmpegWriter(dst, out_w, out_h, fps_out)

    expected = (total // stride) if total else 0
    if max_frames:
        expected = min(expected, max_frames) if expected else max_frames

    processed = 0
    peak = 0
    try:
        for _, frame in iter_frames(src, stride=stride, max_frames=max_frames):
            if ratio != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            tracks = stream.process(frame)
            peak = max(peak, len(tracks))
            writer.write(draw_overlay(frame, tracks, stream))
            processed += 1
            if on_progress and processed % 5 == 0:
                on_progress(processed, expected)
    finally:
        writer.close()

    if on_progress:
        on_progress(processed, expected or processed)

    return VideoStats(
        frames=processed,
        duration_s=round(processed / fps_out, 2) if fps_out else 0.0,
        fps_in=round(fps_in, 2),
        fps_processed=round(fps_out, 2),
        width=out_w,
        height=out_h,
        unique_objects=stream.unique_totals(),
        counts=stream.counter.as_dict(),
        peak_concurrent=peak,
    )
