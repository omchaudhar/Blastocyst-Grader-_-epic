#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blastocyst_grader.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train blastocyst grading model.")
    parser.add_argument("--annotations", required=True, help="CSV with labels and image paths")
    parser.add_argument("--image-root", required=True, help="Root directory for image_path values")
    parser.add_argument("--output-dir", default="models", help="Directory for checkpoints and history")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ImageNet weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = train_model(
        annotations_csv=args.annotations,
        image_root=args.image_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )
    print(f"Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
