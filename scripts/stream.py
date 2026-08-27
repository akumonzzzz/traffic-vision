"""Run detection continuously against a real camera.

This is the headless counterpart to the web demo: point it at an RTSP traffic
camera, a USB webcam, or a video file, and it tracks vehicles and reports counts
as they cross a virtual line. It is what you would actually deploy on a box
sitting next to a camera.

    # A public/ONVIF traffic camera
    python scripts/stream.py --source "rtsp://user:pass@10.0.0.20:554/stream1"

    # Local webcam, write an annotated recording
    python scripts/stream.py --source 0 --save out.mp4

    # Log a CSV of crossings for later analysis
    python scripts/stream.py --source feed.mp4 --csv counts.csv --headless

Press q in the preview window to stop, or Ctrl-C when headless.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_stopping = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _stopping
    _stopping = True
    print("\nStopping...", file=sys.stderr)


def parse_source(raw: str):
    """A bare integer means a local camera index; anything else is a path/URL."""
    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="RTSP/HTTP URL, video file path, or camera index (0)")
    parser.add_argument("--model", default=None,
                        help="Weights to use (defaults to MODEL_PATH / yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--line-y", type=float, default=0.5,
                        help="Counting line height, 0-1 of frame height")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class ids (default: traffic classes)")
    parser.add_argument("--width", type=int, default=960,
                        help="Downscale frames to this width before inference")
    parser.add_argument("--save", type=Path, default=None,
                        help="Write an annotated H.264 MP4 here")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Append a row per line-crossing event")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a preview window")
    parser.add_argument("--report-every", type=float, default=5.0,
                        help="Seconds between console stat lines")
    parser.add_argument("--reconnect", type=int, default=5,
                        help="Reconnect attempts if the stream drops (0 to disable)")
    args = parser.parse_args()

    import cv2

    from app import config, video
    from app.detector import TrafficDetector

    signal.signal(signal.SIGINT, _handle_signal)

    detector = TrafficDetector(args.model or config.MODEL_PATH)
    classes = ([int(c) for c in args.classes.split(",")] if args.classes else None)
    stream = video.TrackedStream(detector, classes=classes, conf=args.conf, iou=args.iou)
    stream.counter = video.LineCounter((0.0, args.line_y), (1.0, args.line_y))

    csv_writer = csv_file = None
    if args.csv:
        new_file = not args.csv.exists()
        csv_file = args.csv.open("a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(["timestamp", "track_id", "class", "direction",
                                 "total_in", "total_out"])

    source = parse_source(args.source)
    writer = None
    attempts = 0
    frames = 0
    started = time.perf_counter()
    last_report = started
    # Remember counts so we can tell which ids crossed since the last frame.
    prev_in = prev_out = 0

    try:
        while not _stopping:
            cap = cv2.VideoCapture(source)
            # Keep latency low on live sources: a big buffer means you are always
            # looking at the past.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                attempts += 1
                if args.reconnect and attempts <= args.reconnect:
                    print(f"Cannot open source; retry {attempts}/{args.reconnect} in 3s",
                          file=sys.stderr)
                    time.sleep(3)
                    continue
                print(f"Could not open source: {args.source}", file=sys.stderr)
                return 1

            attempts = 0
            print(f"Connected to {args.source}", file=sys.stderr)

            while not _stopping:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame.shape[1] > args.width:
                    ratio = args.width / frame.shape[1]
                    frame = cv2.resize(frame, (args.width, int(frame.shape[0] * ratio)),
                                       interpolation=cv2.INTER_AREA)

                tracks = stream.process(frame)
                frames += 1
                annotated = video.draw_overlay(frame, tracks, stream)

                counter = stream.counter
                if csv_writer and (counter.total_in != prev_in or counter.total_out != prev_out):
                    direction = "in" if counter.total_in != prev_in else "out"
                    csv_writer.writerow([
                        datetime.now(UTC).isoformat(timespec="seconds"),
                        "", "", direction, counter.total_in, counter.total_out,
                    ])
                    csv_file.flush()
                prev_in, prev_out = counter.total_in, counter.total_out

                if args.save:
                    if writer is None:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                        writer = video.FfmpegWriter(
                            args.save, annotated.shape[1], annotated.shape[0], fps)
                    writer.write(annotated)

                if not args.headless:
                    cv2.imshow("Traffic Vision - press q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        _handle_signal(None, None)

                now = time.perf_counter()
                if now - last_report >= args.report_every:
                    elapsed = now - started
                    print(
                        f"[{frames:>6} frames | {frames / elapsed:5.1f} fps] "
                        f"in-frame={len(tracks):<3} unique={len(stream.seen_ids):<4} "
                        f"IN={counter.total_in:<4} OUT={counter.total_out:<4} "
                        f"{stream.unique_totals()}",
                        file=sys.stderr,
                    )
                    last_report = now

            cap.release()
            if _stopping or not args.reconnect:
                break
            print("Stream ended; reconnecting...", file=sys.stderr)
            time.sleep(2)

    finally:
        if writer:
            writer.close()
            print(f"Saved annotated video to {args.save}", file=sys.stderr)
        if csv_file:
            csv_file.close()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-6)
    counter = stream.counter
    print("\n=== Session summary ===")
    print(f"frames processed : {frames} ({frames / elapsed:.1f} fps)")
    print(f"unique vehicles  : {len(stream.seen_ids)}")
    print(f"by class         : {stream.unique_totals()}")
    print(f"crossed inbound  : {counter.total_in}")
    print(f"crossed outbound : {counter.total_out}")
    print(f"per class        : {dict(counter.counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
