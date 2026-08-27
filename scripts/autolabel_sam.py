"""Turn click-points into YOLO bounding-box labels using SAM 2.1.

Fixes three bugs from the original prototype:

1. Each point is now its own object prompt. Passing ``points=[[p1, p2, ...]]``
   tells SAM that every point belongs to ONE object, so 11 clicked vehicles
   collapsed into a single box spanning the whole image. We pass one prompt per
   object instead and keep the highest-scoring mask for each.
2. Labels are written to ``dataset/labels/<split>/``, not next to the image.
   Ultralytics resolves labels by swapping ``/images/`` for ``/labels/`` in the
   path, so a .txt sitting beside the .jpg is silently ignored.
3. A prompt that produces no mask is reported instead of leaving an empty file
   that looks like a legitimate "no objects here" negative sample.

Usage:
    python scripts/autolabel_sam.py --points points.json --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "dataset"


def load_points(path: Path) -> dict[str, list]:
    """Read {"image.jpg": [[x, y], ...]} written by your labelling tool."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Points file must be an object mapping filename -> [[x, y], ...]")
    return data


def label_image(model, image_path: Path, points: list, class_id: int) -> list[str]:
    """One SAM call per clicked point; returns YOLO-format label lines."""
    lines: list[str] = []

    for point in points:
        # points=[[[x, y]]] -> a single prompt, for a single object.
        # labels=[[1]] marks it foreground (0 would mean "exclude this region").
        results = model(str(image_path), points=[[point]], labels=[[1]], verbose=False)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"  no mask for point {point}", file=sys.stderr)
            continue

        # SAM can return several candidate masks; keep the most confident.
        best = 0
        if boxes.conf is not None and len(boxes.conf) > 1:
            best = int(boxes.conf.argmax().item())

        x, y, w, h = (float(v) for v in boxes.xywhn[best].tolist())
        if w <= 0 or h <= 0:
            print(f"  degenerate box for point {point}", file=sys.stderr)
            continue
        # A box covering nearly the whole frame is the classic sign of a merged
        # multi-object mask -- worth flagging rather than writing silently.
        if w > 0.97 and h > 0.97:
            print(f"  WARNING: near-full-frame box for point {point}", file=sys.stderr)

        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, required=True,
                        help="JSON file: {'train1.jpg': [[x, y], ...], ...}")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--model", default="sam2.1_b.pt")
    parser.add_argument("--class-id", type=int, default=0,
                        help="Class index to assign to every box")
    args = parser.parse_args()

    from ultralytics import SAM  # imported late so --help stays instant

    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    label_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}", file=sys.stderr)
        return 1

    image_points = load_points(args.points)
    model = SAM(args.model)

    labelled = skipped = total_boxes = 0

    for filename, points in image_points.items():
        image_path = image_dir / filename
        if not image_path.exists():
            print(f"SKIP (not found): {image_path}", file=sys.stderr)
            skipped += 1
            continue
        if not points:
            print(f"SKIP (no points): {filename}", file=sys.stderr)
            skipped += 1
            continue

        print(f"{filename}: {len(points)} point(s)")
        lines = label_image(model, image_path, points, args.class_id)

        if not lines:
            print(f"SKIP (no boxes produced): {filename}", file=sys.stderr)
            skipped += 1
            continue

        out_path = label_dir / f"{image_path.stem}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        labelled += 1
        total_boxes += len(lines)
        print(f"  -> {out_path.relative_to(REPO_ROOT)} ({len(lines)} boxes)")

    print(f"\nDone: {labelled} image(s) labelled, {total_boxes} boxes, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
