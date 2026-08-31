"""Low-light frame enhancement.

COCO is overwhelmingly a daytime dataset, so a detector trained on it loses
recall badly after dark: vehicles become low-contrast blobs, headlights blow out
the sensor, and confidence scores collapse below any sane threshold.

Fine-tuning on night footage is the real fix. This module is the cheap one --
raise local contrast before inference so the existing weights have something to
work with. It costs a few milliseconds and needs no retraining.
"""

from __future__ import annotations

import cv2
import numpy as np


def brightness(frame_bgr: np.ndarray) -> float:
    """Mean perceived luminance, 0-255.

    Sampled on a downscaled copy: full-resolution statistics cost real time on a
    video loop and buy no extra accuracy for a single scalar.
    """
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def is_low_light(frame_bgr: np.ndarray, threshold: float = 75.0) -> bool:
    """Whether a frame is dark enough to be worth enhancing.

    75 sits below dusk but above a well-lit night road, so street-lit scenes with
    working detections are left alone and genuinely dark ones are lifted.
    """
    return brightness(frame_bgr) < threshold


def enhance(frame_bgr: np.ndarray, clip_limit: float = 0.0,
            gamma: float = 0.6) -> np.ndarray:
    """Lift a dark frame with a gamma curve, and optionally CLAHE.

    **CLAHE is off by default, and that is a measured decision, not an oversight.**
    It is the textbook recommendation for low light, but on a simulated night
    sweep it consistently *cost* detections (8 -> 6 at one exposure) where plain
    gamma gained them (8 -> 10). On a dark, noisy frame its local contrast
    amplification boosts sensor noise as hard as signal, and the detector's
    features drown in it. Raise clip_limit only if your footage is genuinely
    low-noise and low-contrast rather than dark.

    Both operate on L in LAB, not on RGB channels: equalising channels
    independently shifts hue, and the class colours would drift with it.

    Caveat worth keeping in mind: the sweep counted *detections*, not correct
    ones. Gamma below ~0.4 produced more boxes than the same scene in daylight,
    which means false positives rather than recovered vehicles. 0.6 was the most
    aggressive setting that never overshot the daylight count. Measuring this
    properly needs mAP against a labelled night set.
    """
    # With both stages disabled there is nothing to do, and the BGR->LAB->BGR
    # round trip is not free: 8-bit quantisation shifts values by up to 3 levels,
    # so "no enhancement" would still alter every pixel.
    if clip_limit <= 0 and gamma == 1.0:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    if clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)

    merged = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

    if gamma != 1.0:
        # out = in ** gamma, so gamma < 1 brightens. Note this is the opposite
        # convention to the 1/gamma form used elsewhere -- getting it backwards
        # darkens the midtones and costs detections rather than recovering them.
        # Built once as a 256-entry LUT; per-pixel pow() on a 1 MP frame is
        # orders of magnitude slower.
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                         dtype=np.uint8)
        merged = cv2.LUT(merged, table)

    return merged


def maybe_enhance(frame_bgr: np.ndarray, mode: str = "auto",
                  threshold: float = 75.0) -> tuple[np.ndarray, bool]:
    """Apply enhancement per policy. Returns (frame, was_enhanced).

    mode: "auto" enhances only dark frames, "on" always, "off" never.
    """
    if mode == "off":
        return frame_bgr, False
    if mode == "on":
        return enhance(frame_bgr), True
    if is_low_light(frame_bgr, threshold):
        return enhance(frame_bgr), True
    return frame_bgr, False
