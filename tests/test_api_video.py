"""API tests for the video job endpoints and the live websocket."""

import json
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def road_frame():
    img = cv2.imread("app/static/samples/traffic2.jpg")
    assert img is not None
    return img


@pytest.fixture(scope="module")
def clip_bytes(road_frame):
    """A tiny panning clip as raw bytes, ready to upload."""
    import tempfile
    from pathlib import Path

    h, w = road_frame.shape[:2]
    ch, cw = (int(h * 0.6) & ~1), (int(w * 0.6) & ~1)
    path = Path(tempfile.mkdtemp()) / "clip.mp4"

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{cw}x{ch}", "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(20):
        x = int(i / 19 * (w - cw))
        proc.stdin.write(np.ascontiguousarray(road_frame[0:ch, x:x + cw]).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path.read_bytes()


def wait_for_job(client, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/video/{job_id}").json()
        if body["state"] in {"done", "failed"}:
            return body
        time.sleep(0.4)
    raise AssertionError("job did not finish in time")


# --------------------------------------------------------------------------
# video endpoints
# --------------------------------------------------------------------------

def test_video_round_trip(client, clip_bytes):
    res = client.post(
        "/api/video",
        files={"file": ("clip.mp4", clip_bytes, "video/mp4")},
        data={"conf": "0.35", "stride": "1", "line_y": "0.5"},
    )
    assert res.status_code == 200
    submitted = res.json()
    assert submitted["state"] in {"queued", "running"}
    assert submitted["source"]["frames"] == 20

    job = wait_for_job(client, submitted["job_id"])
    assert job["state"] == "done", job.get("error")
    assert job["progress"] == 1.0
    assert job["stats"]["frames"] == 20
    assert "result_url" in job

    result = client.get(job["result_url"])
    assert result.status_code == 200
    assert result.headers["content-type"] == "video/mp4"
    assert len(result.content) > 1000


def test_video_stride_reduces_frames(client, clip_bytes):
    res = client.post(
        "/api/video",
        files={"file": ("clip.mp4", clip_bytes, "video/mp4")},
        data={"stride": "4"},
    )
    job = wait_for_job(client, res.json()["job_id"])
    assert job["state"] == "done"
    assert job["stats"]["frames"] == 5


def test_video_rejects_unsupported_extension(client, clip_bytes):
    res = client.post(
        "/api/video",
        files={"file": ("clip.gif", clip_bytes, "image/gif")},
    )
    assert res.status_code == 400
    assert "format" in res.json()["detail"].lower()


def test_video_rejects_non_video_payload(client):
    res = client.post(
        "/api/video",
        files={"file": ("clip.mp4", b"this is not a video", "video/mp4")},
    )
    assert res.status_code == 400


def test_video_rejects_empty_upload(client):
    res = client.post("/api/video", files={"file": ("clip.mp4", b"", "video/mp4")})
    assert res.status_code == 400


def test_video_rejects_oversized_upload(client, clip_bytes, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "MAX_VIDEO_BYTES", 128)
    res = client.post("/api/video", files={"file": ("clip.mp4", clip_bytes, "video/mp4")})
    assert res.status_code == 413


def test_unknown_job_is_404(client):
    assert client.get("/api/video/does-not-exist").status_code == 404
    assert client.get("/api/video/does-not-exist/result").status_code == 404


def test_result_before_completion_is_409(client, clip_bytes):
    res = client.post("/api/video", files={"file": ("clip.mp4", clip_bytes, "video/mp4")})
    job_id = res.json()["job_id"]
    early = client.get(f"/api/video/{job_id}/result")
    # Either still working (409) or already finished on a fast machine (200).
    assert early.status_code in {409, 200}
    wait_for_job(client, job_id)


def test_health_reports_capacity(client):
    body = client.get("/api/health").json()
    assert body["live_sessions"]["max"] >= 1
    assert set(body["jobs"]) == {"total", "running", "queued"}


# --------------------------------------------------------------------------
# live websocket
# --------------------------------------------------------------------------

def jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    return buf.tobytes()


def test_live_detects_on_the_first_frame(client, road_frame):
    """Regression: leaked tracker state used to swallow every session's frame 1."""
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"conf": 0.3, "classes": [2, 5, 7]}))
        assert ws.receive_json() == {"ack": True}

        ws.send_bytes(jpeg(road_frame))
        msg = ws.receive_json()

    assert msg["frame"] == 1
    assert msg["tracks"], "no detections on the first frame"
    assert msg["unique_total"] == len(msg["tracks"])
    assert msg["inference_ms"] > 0

    box = msg["tracks"][0]
    # Coordinates are normalised so the client can scale them to any size.
    assert 0.0 <= box["x"] <= 1.0 and 0.0 <= box["y"] <= 1.0
    assert 0.0 < box["w"] <= 1.0 and 0.0 < box["h"] <= 1.0


def test_live_sessions_are_independent(client, road_frame):
    payload = jpeg(road_frame)
    with client.websocket_connect("/ws/live") as first:
        first.send_bytes(payload)
        first_msg = first.receive_json()

        with client.websocket_connect("/ws/live") as second:
            second.send_bytes(payload)
            second_msg = second.receive_json()

    assert first_msg["tracks"] and second_msg["tracks"]
    # Each session numbers its own tracks from 1.
    assert min(t["id"] for t in second_msg["tracks"]) == 1


def test_live_reset_clears_counters(client, road_frame):
    payload = jpeg(road_frame)
    with client.websocket_connect("/ws/live") as ws:
        for _ in range(3):
            ws.send_bytes(payload)
            before = ws.receive_json()
        assert before["unique_total"] > 0

        ws.send_text(json.dumps({"reset": True}))
        assert ws.receive_json() == {"ack": True}

        ws.send_bytes(payload)
        after = ws.receive_json()

    assert after["frame"] == 4, "frame counter should keep running"
    assert after["line"]["total_in"] == 0 and after["line"]["total_out"] == 0


def test_live_rejects_undecodable_frame(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"absolutely not a jpeg")
        assert "error" in ws.receive_json()


def test_live_rejects_malformed_config(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text("{not json")
        assert "error" in ws.receive_json()


def test_live_class_filter_is_applied(client, road_frame):
    with client.websocket_connect("/ws/live") as ws:
        # 11 is 'stop sign' - a highway scene has none.
        ws.send_text(json.dumps({"classes": [11], "conf": 0.3}))
        ws.receive_json()
        ws.send_bytes(jpeg(road_frame))
        msg = ws.receive_json()
    assert msg["tracks"] == []


def test_live_survives_a_bad_frame_and_keeps_going(client, road_frame):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"garbage")
        assert "error" in ws.receive_json()
        ws.send_bytes(jpeg(road_frame))
        good = ws.receive_json()
    assert good["tracks"], "connection did not recover after a bad frame"
