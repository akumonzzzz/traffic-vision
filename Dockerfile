# syntax=docker/dockerfile:1
FROM python:3.12-slim

# HF Spaces serves on 7860; keep it the default everywhere for parity.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    MODEL_PATH=yolo11s.pt \
    DEVICE=cpu \
    # Ultralytics writes a settings file at import time; point it somewhere writable.
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# libglib2 is still needed by opencv-headless; libgl is not.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as uid 1000. Match it so the writable dirs line up.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# CPU-only torch first, then the rest. Separate layer = cached across code edits.
#
# The opencv dance below is load-bearing, and both halves are needed.
#
# Ultralytics declares `opencv-python` (the GUI build) as a dependency, so pip
# installs it alongside the `opencv-python-headless` we asked for. The GUI build
# links libxcb/libGL, which a slim image does not carry, so `import cv2` dies
# with "libxcb.so.1: cannot open shared object file".
#
# Uninstalling it is not enough on its own: both wheels unpack into the *same*
# cv2/ directory, so removing one deletes files the other still needs and leaves
# a hollow module ("cv2 has no attribute __version__"). Reinstalling headless
# afterwards restores that directory with headless as its sole owner.
#
# The import check makes a regression here fail the build instead of the
# container dying at runtime.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install -r requirements.txt \
 && pip uninstall -y opencv-python opencv-contrib-python \
 && pip install --force-reinstall --no-deps opencv-python-headless==4.12.0.88 \
 && python -c "import cv2; print('cv2', cv2.__version__, 'headless OK')"

COPY --chown=appuser:appuser app ./app

# Bake the weights into the image so the first request is not a 6 MB download.
RUN mkdir -p "$YOLO_CONFIG_DIR" /home/appuser/.cache \
 && chown -R appuser:appuser /home/appuser /app
USER appuser
RUN python -c "from ultralytics import YOLO; YOLO('${MODEL_PATH}')"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
