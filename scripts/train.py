"""Fine-tune YOLO on the traffic dataset.

Your machine reports CPU-only torch, so a real training run belongs on a GPU
(Colab, Kaggle, or a rented instance). Copy the resulting best.pt into
weights/ and set MODEL_PATH to serve it -- no other code changes needed.

    python scripts/train.py --model yolo11s.pt --epochs 100 --batch 16
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"
WEIGHTS_DIR = REPO_ROOT / "weights"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11s.pt", help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    parser.add_argument("--name", default="traffic-yolo")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early-stop after N epochs without improvement")
    args = parser.parse_args()

    # A bare name like "coco128.yaml" is an Ultralytics built-in that it resolves
    # and downloads itself, so only a value that looks like a local path gets an
    # existence check. Rejecting the built-in name blocks the obvious smoke test
    # -- and evaluate.py already accepts it, so the two scripts disagreed.
    looks_local = args.data.parent != pathlib.Path(".") or args.data.exists()
    if looks_local and not args.data.exists():
        raise SystemExit(
            f"Dataset config not found: {args.data}. "
            f"For a public smoke test try: --data coco128.yaml"
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=args.patience,
        project=str(REPO_ROOT / "runs"),
        pretrained=True,
        # Augmentation tuned for dashcam/CCTV road scenes: horizontal flips are
        # valid, vertical flips are not (traffic is never upside down).
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        scale=0.5,
        mosaic=1.0,
    )

    metrics = model.val(data=str(args.data), device=args.device)
    print("\n=== Validation ===")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall   : {metrics.box.mr:.4f}")

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_DIR.mkdir(exist_ok=True)
        target = WEIGHTS_DIR / f"{args.name}.pt"
        shutil.copy2(best, target)
        print(f"\nBest weights copied to {target}")
        print(f"Serve them with:  MODEL_PATH=weights/{args.name}.pt uvicorn app.main:app")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
