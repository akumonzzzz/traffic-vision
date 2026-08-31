"""Low-light enhancement tests.

The important one is `test_gamma_brightens_not_darkens`. The exponent convention
is easy to invert, and inverting it silently costs detections instead of
recovering them -- the bug this module shipped with first time round.
"""

import cv2
import numpy as np
import pytest

from app import enhance


@pytest.fixture
def road():
    img = cv2.imread("app/static/samples/traffic2.jpg")
    assert img is not None
    return img


def darken(img, gain=0.3, gamma=2.0):
    f = (img.astype(np.float32) / 255.0) ** gamma * gain
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def test_brightness_ranks_scenes_correctly(road):
    assert enhance.brightness(road) > enhance.brightness(darken(road))


def test_brightness_bounds():
    assert enhance.brightness(np.zeros((80, 80, 3), np.uint8)) == pytest.approx(0, abs=1)
    white = np.full((80, 80, 3), 255, np.uint8)
    assert enhance.brightness(white) == pytest.approx(255, abs=1)


def test_low_light_detection(road):
    assert not enhance.is_low_light(road)
    assert enhance.is_low_light(darken(road))


def test_gamma_brightens_not_darkens(road):
    """Regression: gamma < 1 must brighten. The inverted form darkens instead."""
    dark = darken(road)
    out = enhance.enhance(dark, clip_limit=0.0, gamma=0.6)
    assert enhance.brightness(out) > enhance.brightness(dark)


def test_gamma_of_one_is_a_no_op_without_clahe(road):
    out = enhance.enhance(road, clip_limit=0.0, gamma=1.0)
    np.testing.assert_array_equal(out, road)


def test_clahe_is_off_by_default(road):
    """Measured decision: CLAHE cost detections on dark frames, so default is off."""
    dark = darken(road)
    default = enhance.enhance(dark)
    explicit = enhance.enhance(dark, clip_limit=0.0, gamma=0.6)
    np.testing.assert_array_equal(default, explicit)


def test_enhance_preserves_shape_and_dtype(road):
    out = enhance.enhance(darken(road))
    assert out.shape == road.shape
    assert out.dtype == np.uint8


def test_enhance_does_not_wreck_hue(road):
    """CLAHE on RGB channels shifts colour; on LAB's L channel it should not."""
    dark = darken(road)
    out = enhance.enhance(dark, clip_limit=2.5, gamma=1.0)
    a_before = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)[:, :, 1].mean()
    a_after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 1].mean()
    assert abs(a_before - a_after) < 3


@pytest.mark.parametrize("mode,expect", [("off", False), ("on", True)])
def test_maybe_enhance_explicit_modes(road, mode, expect):
    _, applied = enhance.maybe_enhance(road, mode=mode)
    assert applied is expect


def test_maybe_enhance_off_returns_original_object(road):
    out, applied = enhance.maybe_enhance(road, mode="off")
    assert out is road and applied is False


def test_maybe_enhance_auto_skips_bright_frames(road):
    _, applied = enhance.maybe_enhance(road, mode="auto")
    assert applied is False


def test_maybe_enhance_auto_fires_on_dark_frames(road):
    out, applied = enhance.maybe_enhance(darken(road), mode="auto")
    assert applied is True
    assert enhance.brightness(out) > enhance.brightness(darken(road))


def test_enhancement_recovers_detections_on_a_dark_frame(road):
    """The claim the feature rests on: a boosted dark frame yields at least as
    many detections as the raw one."""
    from app.detector import TrafficDetector

    det = TrafficDetector()
    dark = darken(road, gain=0.4)

    def n(frame):
        r = det.model.predict(frame, conf=0.25, classes=det.traffic_class_ids,
                              imgsz=640, device="cpu", verbose=False)[0]
        return len(r.boxes)

    assert n(enhance.enhance(dark)) >= n(dark)
