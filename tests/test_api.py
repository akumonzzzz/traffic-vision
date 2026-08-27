"""End-to-end API tests. These load the real model, so the first run is slow."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    # The context manager triggers lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


def make_image(size=(640, 480), color=(120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["classes"] > 0


def test_classes_endpoint_lists_traffic_classes(client):
    classes = client.get("/api/classes").json()["classes"]
    names = {c["name"] for c in classes}
    assert "car" in names
    assert all(len(c["color"]) == 3 for c in classes)


def test_detect_returns_expected_schema(client):
    res = client.post(
        "/api/detect",
        files={"file": ("scene.jpg", make_image(), "image/jpeg")},
        data={"conf": "0.35", "annotate": "true"},
    )
    assert res.status_code == 200
    body = res.json()

    assert set(body) >= {"count", "counts_by_class", "detections", "image", "inference_ms"}
    assert body["count"] == len(body["detections"])
    assert body["image"] == {"width": 640, "height": 480}
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")


def test_detect_image_returns_jpeg(client):
    res = client.post(
        "/api/detect/image",
        files={"file": ("scene.jpg", make_image(), "image/jpeg")},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert "X-Detection-Count" in res.headers
    Image.open(io.BytesIO(res.content)).verify()  # decodes as a real JPEG


def test_non_image_upload_is_rejected(client):
    res = client.post(
        "/api/detect",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert res.status_code == 400


def test_empty_upload_is_rejected(client):
    res = client.post("/api/detect", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert res.status_code == 400


def test_bad_class_filter_is_rejected(client):
    res = client.post(
        "/api/detect",
        files={"file": ("scene.jpg", make_image(), "image/jpeg")},
        data={"classes": "not-a-number"},
    )
    assert res.status_code == 400


def test_unknown_class_id_is_rejected(client):
    res = client.post(
        "/api/detect",
        files={"file": ("scene.jpg", make_image(), "image/jpeg")},
        data={"classes": "9999"},
    )
    assert res.status_code == 400


def test_oversized_upload_is_rejected(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    res = client.post(
        "/api/detect",
        files={"file": ("big.jpg", make_image(), "image/jpeg")},
    )
    assert res.status_code == 413
