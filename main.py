#!/usr/bin/env python3
"""CLI entry point for pitch prediction preprocessing and training."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing import DATA_DIR, OUTPUT_DIR, preprocess_phase1
from logistic_regression import (
    ARTIFACTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    MIN_CLASS_COUNT,
    run_logistic_regression,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pitch prediction preprocessing and baseline training.")
    subparsers = parser.add_subparsers(dest="command")

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Build cleaned Phase 1 datasets from raw Statcast CSVs.",
    )
    preprocess_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    preprocess_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    train_parser = subparsers.add_parser(
        "train-logreg",
        help="Train per-pitcher multiclass logistic-regression baselines.",
    )
    train_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    train_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    train_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--min-class-count", type=int, default=MIN_CLASS_COUNT)
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "preprocess"):
        preprocess_phase1(
            data_dir=getattr(args, "data_dir", DATA_DIR),
            output_dir=getattr(args, "output_dir", OUTPUT_DIR),
        )
        return

    if args.command == "train-logreg":
        run_logistic_regression(args)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
