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
import lstm_model
import xgboost_model
import knn_model
import hmm_model


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

    lstm_parser = subparsers.add_parser(
        "train-lstm",
        help="Train per-pitcher LSTM sequence baselines.",
    )
    lstm_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    lstm_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    lstm_parser.add_argument("--artifacts-dir", type=Path, default=lstm_model.ARTIFACTS_DIR)
    lstm_parser.add_argument("--seed", type=int, default=lstm_model.DEFAULT_SEED)
    lstm_parser.add_argument("--min-class-count", type=int, default=lstm_model.MIN_CLASS_COUNT)
    lstm_parser.add_argument("--epochs", type=int, default=lstm_model.DEFAULT_EPOCHS)
    lstm_parser.add_argument("--batch-size", type=int, default=lstm_model.DEFAULT_BATCH_SIZE)
    lstm_parser.add_argument("--learning-rate", type=float, default=lstm_model.DEFAULT_LR)
    lstm_parser.add_argument("--optimizer", choices=["sgd", "adam"], default=lstm_model.DEFAULT_OPTIMIZER)
    lstm_parser.add_argument("--hidden-size", type=int, default=lstm_model.DEFAULT_HIDDEN_SIZE)
    lstm_parser.add_argument("--num-layers", type=int, default=lstm_model.DEFAULT_NUM_LAYERS)
    lstm_parser.add_argument("--dropout", type=float, default=lstm_model.DEFAULT_DROPOUT)
    lstm_parser.add_argument("--sequence-length", type=int, default=lstm_model.DEFAULT_SEQUENCE_LENGTH)
    lstm_parser.add_argument("--grad-clip", type=float, default=lstm_model.DEFAULT_GRAD_CLIP)
    lstm_parser.add_argument("--max-pitchers", type=int, default=None)
    lstm_parser.add_argument("--include-location", action="store_true")
    lstm_parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    lstm_parser.add_argument("--progress-every", type=int, default=10)

    xgb_parser = subparsers.add_parser(
        "train-xgboost",
        help="Train per-pitcher XGBoost baselines.",
    )
    xgb_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    xgb_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    xgb_parser.add_argument("--artifacts-dir", type=Path, default=xgboost_model.ARTIFACTS_DIR)
    xgb_parser.add_argument("--seed", type=int, default=xgboost_model.DEFAULT_SEED)
    xgb_parser.add_argument("--min-class-count", type=int, default=xgboost_model.MIN_CLASS_COUNT)
    xgb_parser.add_argument("--sequence-length", type=int, default=xgboost_model.DEFAULT_SEQUENCE_LENGTH)
    xgb_parser.add_argument("--num-boost-round", type=int, default=xgboost_model.DEFAULT_NUM_BOOST_ROUND)
    xgb_parser.add_argument("--learning-rate", type=float, default=xgboost_model.DEFAULT_LEARNING_RATE)
    xgb_parser.add_argument("--max-depth", type=int, default=xgboost_model.DEFAULT_MAX_DEPTH)
    xgb_parser.add_argument("--subsample", type=float, default=xgboost_model.DEFAULT_SUBSAMPLE)
    xgb_parser.add_argument("--colsample-bytree", type=float, default=xgboost_model.DEFAULT_COLSAMPLE_BYTREE)
    xgb_parser.add_argument("--max-pitchers", type=int, default=None)
    xgb_parser.add_argument("--include-location", action="store_true")

    knn_parser = subparsers.add_parser(
        "train-knn",
        help="Train per-pitcher KNN baselines.",
    )
    knn_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    knn_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    knn_parser.add_argument("--artifacts-dir", type=Path, default=knn_model.ARTIFACTS_DIR)
    knn_parser.add_argument("--seed", type=int, default=knn_model.DEFAULT_SEED)
    knn_parser.add_argument("--min-class-count", type=int, default=knn_model.MIN_CLASS_COUNT)
    knn_parser.add_argument("--sequence-length", type=int, default=knn_model.DEFAULT_SEQUENCE_LENGTH)
    knn_parser.add_argument("--n-neighbors", type=int, default=knn_model.DEFAULT_N_NEIGHBORS)
    knn_parser.add_argument("--weights", choices=["uniform", "distance"], default=knn_model.DEFAULT_WEIGHTS)
    knn_parser.add_argument("--metric", default=knn_model.DEFAULT_METRIC)
    knn_parser.add_argument("--max-pitchers", type=int, default=None)

    hmm_parser = subparsers.add_parser(
        "train-hmm",
        help="Train per-pitcher supervised Markov/HMM-style baselines.",
    )
    hmm_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    hmm_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    hmm_parser.add_argument("--artifacts-dir", type=Path, default=hmm_model.ARTIFACTS_DIR)
    hmm_parser.add_argument("--seed", type=int, default=hmm_model.DEFAULT_SEED)
    hmm_parser.add_argument("--min-class-count", type=int, default=hmm_model.MIN_CLASS_COUNT)
    hmm_parser.add_argument("--smoothing", type=float, default=hmm_model.DEFAULT_SMOOTHING)
    hmm_parser.add_argument("--max-pitchers", type=int, default=None)

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

    if args.command == "train-lstm":
        lstm_model.run_lstm(args)
        return

    if args.command == "train-xgboost":
        xgboost_model.run_xgboost(args)
        return

    if args.command == "train-knn":
        knn_model.run_knn(args)
        return

    if args.command == "train-hmm":
        hmm_model.run_hmm(args)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
