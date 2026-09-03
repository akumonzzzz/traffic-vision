"""Measure detection accuracy of the model this service actually serves.

`train.py` reports mAP at the end of a fine-tune, but that only covers a model
you trained yourself. The weights being served by default are stock COCO
weights, and their accuracy on road scenes has never been measured -- the
README quotes latency and says nothing about whether the boxes are right.

This script closes that gap. It evaluates whatever MODEL_PATH points at,
against a dataset in the layout described in docs/DATASET.md, and prints a
Markdown table ready to paste into the README.

    python scripts/evaluate.py                          # serve-time defaults
    python scripts/evaluate.py --model yolo11s.pt       # compare a bigger model
    python scripts/evaluate.py --data path/to/data.yaml

Reading the numbers:

  mAP50      Average precision at IoU 0.5 -- a box counts as correct if it
             overlaps the true box by half. The number usually quoted.
  mAP50-95   Averaged over IoU thresholds 0.5 to 0.95. Much harsher: it also
             rewards boxes that are tightly placed, not merely overlapping.
             Always lower than mAP50; a large gap means loose boxes.
  precision  Of the objects the model reported, how many were real.
  recall     Of the objects that were there, how many it found.

Precision and recall move against each other as --conf changes, so a single
pair of numbers only describes one operating point. Report the confidence
threshold alongside them or the numbers cannot be compared with anyone else's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"


def build_parser() -> argparse.ArgumentParser:
    from app import config

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Ultralytics dataset config (default: dataset/data.yaml)")
    parser.add_argument("--model", default=config.MODEL_PATH,
                        help="Weights to evaluate (default: whatever MODEL_PATH serves)")
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence floor for evaluation. Keep this low: mAP is "
                             "computed across the whole precision-recall curve, and a "
                             "serve-time threshold like 0.35 truncates the curve and "
                             "understates the score.")
    parser.add_argument("--iou", type=float, default=config.DEFAULT_IOU,
                        help="NMS IoU threshold")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--split", default="val", choices=("val", "train", "test"))
    parser.add_argument("--all-classes", action="store_true",
                        help="Score every class the model knows. By default only the "
                             "classes this service actually serves (TRAFFIC_CLASS_NAMES) "
                             "are scored, since a road-scene mAP that averages in teddy "
                             "bears and toothbrushes describes nothing useful.")
    return parser


def traffic_class_ids(model_names: dict[int, str]) -> list[int]:
    """Class ids the served model shares with TRAFFIC_CLASS_NAMES.

    A custom fine-tune may not use COCO names at all; in that case nothing
    matches and we fall back to scoring everything, mirroring how
    detector.resolve_class_filter behaves at serve time.
    """
    from app import config

    wanted = {name.lower() for name in config.TRAFFIC_CLASS_NAMES}
    return sorted(i for i, name in model_names.items() if str(name).lower() in wanted)


def resolve_split_dir(data_yaml: Path, split: str) -> Path | None:
    """Work out where the images for a split actually live."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    rel = cfg.get(split)
    if rel is None:
        return None
    base = (data_yaml.parent / cfg.get("path", ".")).resolve()
    return (base / rel).resolve()


def is_builtin_dataset(data: Path) -> bool:
    """True for names Ultralytics resolves and downloads itself, e.g. coco128.yaml.

    A bare filename with no directory part is never a path we should look for on
    disk -- Ultralytics ships those configs and fetches the images on first use.
    """
    return not data.exists() and data.parent == Path(".")


def preflight(data_yaml: Path, split: str) -> str | None:
    """Return an error message if the dataset is not usable, else None.

    Ultralytics fails deep inside its loader with a message that does not say
    which of these went wrong, so check the likely causes here where we can name
    them.
    """
    if not data_yaml.exists():
        return (
            f"Dataset config not found: {data_yaml}\n\n"
            "The dataset is deliberately not committed -- see docs/DATASET.md for the\n"
            "layout the code expects and where to get a labelled traffic set.\n"
            "To check the pipeline against a public set instead:\n"
            "    python scripts/evaluate.py --data coco128.yaml --split train"
        )

    images_dir = resolve_split_dir(data_yaml, split)
    if images_dir is None:
        return f"'{split}' is not defined in {data_yaml}"
    if not images_dir.exists():
        return (
            f"Split directory not found: {images_dir}\n\n"
            "Populate it as described in docs/DATASET.md, or point --data at another\n"
            "dataset config."
        )

    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        return f"No images found under {images_dir}"

    labels_dir = Path(str(images_dir).replace("images", "labels", 1))
    if not labels_dir.exists():
        return (
            f"Found {len(images)} images but no label directory at {labels_dir}\n\n"
            "Ultralytics locates labels by swapping '/images/' for '/labels/' in the\n"
            "image path. A .txt sitting beside its .jpg is silently ignored."
        )

    labelled = sum(1 for img in images if (labels_dir / f"{img.stem}.txt").exists())
    if labelled == 0:
        return f"None of the {len(images)} images has a matching .txt in {labels_dir}"
    if labelled < len(images):
        print(f"  warning: {len(images) - labelled} of {len(images)} images have no label file")

    print(f"  {len(images)} images, {labelled} with labels")
    return None


def markdown_table(metrics, names: dict[int, str]) -> str:
    """Per-class results as a Markdown table, ready for the README."""
    box = metrics.box
    lines = [
        "| Class | Images | Instances | P | R | mAP50 | mAP50-95 |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]

    # ap_class_index tells us which class each row of the per-class arrays is for;
    # classes absent from the split are omitted by Ultralytics, so never assume
    # row order matches the ids in data.yaml.
    for row, class_id in enumerate(box.ap_class_index):
        name = names.get(int(class_id), str(class_id))
        p, r, ap50, ap = box.class_result(row)
        cid = int(class_id)
        instances = int(metrics.nt_per_class[cid]) if hasattr(metrics, "nt_per_class") else 0
        images = int(metrics.nt_per_image[cid]) if hasattr(metrics, "nt_per_image") else 0
        lines.append(
            f"| {name} | {images} | {instances} | "
            f"{p:.3f} | {r:.3f} | {ap50:.3f} | {ap:.3f} |"
        )

    lines.append(
        f"| **All** | | | **{box.mp:.3f}** | **{box.mr:.3f}** | "
        f"**{box.map50:.3f}** | **{box.map:.3f}** |"
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()

    print(f"Model   : {args.model}")
    print(f"Dataset : {args.data}  (split: {args.split})")
    print(f"Settings: imgsz={args.imgsz} conf={args.conf} iou={args.iou} device={args.device}")
    print()

    if is_builtin_dataset(args.data):
        print(f"  '{args.data}' is an Ultralytics built-in; it will download on first use.")
    else:
        problem = preflight(args.data, args.split)
        if problem:
            print(problem, file=sys.stderr)
            return 1

    from ultralytics import YOLO

    model = YOLO(args.model)

    classes = None
    if not args.all_classes:
        classes = traffic_class_ids(model.names)
        if classes:
            print(f"  scoring {len(classes)} traffic classes "
                  f"({', '.join(model.names[i] for i in classes)})")
        else:
            print("  no class name matches TRAFFIC_CLASS_NAMES -- scoring every class")

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=classes,
        verbose=False,
    )

    box = metrics.box
    print()
    print("=" * 62)
    print(f"  mAP50     {box.map50:.4f}")
    print(f"  mAP50-95  {box.map:.4f}")
    print(f"  precision {box.mp:.4f}")
    print(f"  recall    {box.mr:.4f}")
    print("=" * 62)
    print()
    print("Per-class table for the README:")
    print()
    print(markdown_table(metrics, model.names))
    print()
    print(f"_Đo bằng `scripts/evaluate.py` trên `{args.model}`, "
          f"imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, device={args.device}._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
