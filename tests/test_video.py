"""Tracking, counting and video-encoding tests.

The line counter is tested with synthetic tracks rather than real footage: it is
pure geometry, and hand-built positions let us assert exact counts instead of
"about the right number of cars".
"""

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from app import video
from app.detector import TrafficDetector
from app.video import LineCounter, Track

W = H = 1000


def track_at(track_id: int, cy: float, name: str = "car") -> Track:
    """A 40x40 box centred horizontally, at height ``cy``."""
    return Track(track_id, 2, name, 0.9, 480, cy - 20, 520, cy + 20)


def feed(counter: LineCounter, track_id: int, heights, name="car"):
    for cy in heights:
        counter.update([track_at(track_id, cy, name)], W, H)


# --------------------------------------------------------------------------
# LineCounter
# --------------------------------------------------------------------------

def test_downward_crossing_counts_inbound():
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    feed(c, 1, [200, 300, 400, 600, 700])
    assert (c.total_in, c.total_out) == (1, 0)
    assert c.counts["car"] == {"in": 1, "out": 0}


def test_upward_crossing_counts_outbound():
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    feed(c, 2, [800, 700, 600, 400, 300])
    assert (c.total_in, c.total_out) == (0, 1)


def test_track_that_never_crosses_is_not_counted():
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    feed(c, 3, [100, 200, 300, 400, 450])
    assert (c.total_in, c.total_out) == (0, 0)


def test_crossing_counts_once_not_once_per_frame():
    """The regression that matters: a vehicle past the line must not keep counting."""
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    feed(c, 4, [400, 600, 700, 800, 900])
    assert c.total_in == 1


def test_round_trip_counts_both_directions():
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    feed(c, 5, [400, 600, 400])
    assert (c.total_in, c.total_out) == (1, 1)


def test_counts_are_split_by_class():
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    c.update([track_at(6, 400), track_at(7, 400, "truck")], W, H)
    c.update([track_at(6, 600), track_at(7, 600, "truck")], W, H)
    assert c.counts["car"]["in"] == 1
    assert c.counts["truck"]["in"] == 1
    assert c.total_in == 2


def test_unseen_id_does_not_count_on_first_sighting():
    """A track first observed past the line has no previous side, so no crossing."""
    c = LineCounter((0.0, 0.5), (1.0, 0.5))
    c.update([track_at(8, 400)], W, H)
    c.update([track_at(9, 600)], W, H)
    assert (c.total_in, c.total_out) == (0, 0)


def test_line_position_is_respected():
    c = LineCounter((0.0, 0.8), (1.0, 0.8))
    feed(c, 10, [400, 600])          # both above a line at y=0.8
    assert c.total_in == 0
    feed(c, 10, [900])               # now below it
    assert c.total_in == 1


def test_as_dict_shape():
    c = LineCounter((0.0, 0.25), (1.0, 0.25))
    body = c.as_dict()
    assert body["line"] == {"start": [0.0, 0.25], "end": [1.0, 0.25]}
    assert body["total_in"] == 0 and body["total_out"] == 0


# --------------------------------------------------------------------------
# Tracker isolation
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    return TrafficDetector()


@pytest.fixture(scope="module")
def road_frame():
    img = cv2.imread("app/static/samples/traffic2.jpg")
    assert img is not None, "sample image missing"
    return img


def test_streams_do_not_share_tracker_state(detector, road_frame):
    """Regression: two sessions sharing one model leaked frame_id and open
    tracks, so the second lost its first frame and inherited stale ids."""
    first = video.TrackedStream(detector, conf=0.3)
    for _ in range(4):
        first.process(road_frame)
    assert first.seen_ids, "first stream detected nothing"

    second = video.TrackedStream(detector, conf=0.3)
    tracks = second.process(road_frame)

    assert tracks, "second stream lost its first frame to leaked tracker state"
    # A fresh tracker numbers from 1 again.
    assert min(t.track_id for t in tracks) == 1


def test_stream_has_its_own_model(detector):
    a = video.TrackedStream(detector)
    b = video.TrackedStream(detector)
    assert a.model is not b.model
    assert a.model is not detector.model


def test_ids_persist_across_frames(detector, road_frame):
    stream = video.TrackedStream(detector, conf=0.3)
    first = {t.track_id for t in stream.process(road_frame)}
    for _ in range(3):
        later = {t.track_id for t in stream.process(road_frame)}
    assert first & later, "no id survived a static scene"


# --------------------------------------------------------------------------
# Video encoding
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip(tmp_path_factory, road_frame):
    """A short panning clip, so tracking has genuine motion to follow."""
    path = tmp_path_factory.mktemp("clips") / "pan.mp4"
    h, w = road_frame.shape[:2]
    ch, cw = (int(h * 0.6) & ~1), (int(w * 0.6) & ~1)

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{cw}x{ch}", "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frames = 30
    for i in range(frames):
        x = int(i / (frames - 1) * (w - cw))
        proc.stdin.write(np.ascontiguousarray(road_frame[0:ch, x:x + cw]).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, proc.stderr.read().decode()[-400:]
    return path


def test_probe_reads_metadata(clip):
    width, height, fps, frames = video.probe(clip)
    assert width > 0 and height > 0
    assert fps == pytest.approx(20, abs=1)
    assert frames == 30


def test_probe_rejects_non_video(tmp_path):
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"definitely not a video")
    with pytest.raises(ValueError):
        video.probe(bogus)


def test_iter_frames_respects_stride(clip):
    every = list(video.iter_frames(clip))
    third = list(video.iter_frames(clip, stride=3))
    assert len(every) == 30
    assert len(third) == 10


def test_iter_frames_respects_max(clip):
    assert len(list(video.iter_frames(clip, max_frames=7))) == 7


def test_process_video_writes_playable_h264(detector, clip, tmp_path):
    dst = tmp_path / "out.mp4"
    seen = []
    stats = video.process_video(
        detector, clip, dst, conf=0.35,
        on_progress=lambda done, total: seen.append(done),
    )

    assert dst.exists() and dst.stat().st_size > 1000
    assert stats.frames == 30
    assert seen and seen[-1] == 30, "progress never reported completion"

    # The output must be decodable and H.264, or browsers will not play it.
    width, height, fps, frames = video.probe(dst)
    assert width == stats.width and height == stats.height
    info = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(dst)],
        capture_output=True, text=True).stderr
    assert "h264" in info, f"expected H.264, got: {info[-300:]}"


def test_process_video_downscales_oversized_input(detector, clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video.config, "VIDEO_MAX_WIDTH", 320)
    stats = video.process_video(detector, clip, tmp_path / "small.mp4", max_frames=5)
    assert stats.width <= 320


def test_ffmpeg_writer_forces_even_dimensions(tmp_path):
    """libx264 rejects odd dimensions; the writer must round them down."""
    writer = video.FfmpegWriter(tmp_path / "odd.mp4", 101, 77, 10)
    assert writer.width == 100 and writer.height == 76
    for _ in range(4):
        writer.write(np.zeros((77, 101, 3), np.uint8))
    writer.close()
    assert (tmp_path / "odd.mp4").stat().st_size > 0


def test_draw_overlay_marks_the_frame(detector):
    stream = video.TrackedStream(detector)
    frame = np.zeros((400, 600, 3), np.uint8)
    tracks = [Track(1, 2, "car", 0.9, 100, 100, 200, 200)]
    stream.trails[1].extend([(150, 150), (155, 158)])
    out = video.draw_overlay(frame, tracks, stream)
    assert out.shape == frame.shape
    assert out.any(), "overlay drew nothing onto the frame"
    assert not np.array_equal(out, frame)
