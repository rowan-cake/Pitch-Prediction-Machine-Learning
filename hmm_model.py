"""Supervised HMM-style Markov baseline for pitch-type prediction."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from data_preprocessing import DATA_DIR, FEATURE_COLUMNS, OUTPUT_DIR, preprocess_phase1, slugify, write_summary
from logistic_regression import evaluate_predictions
from lstm_model import TRAIN_FRACTION, chronological_game_split


ARTIFACTS_DIR = Path("artifacts/hmm")
DEFAULT_SEED = 440
MIN_CLASS_COUNT = 20
DEFAULT_SMOOTHING = 1.0


def ensure_hmm_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing HMM dependencies. Install them with `python3 -m pip install -r requirements.txt` "
            "before running `train-hmm`."
        ) from exc
    return np, pd


def ensure_processed_dataset(processed_csv: Path, data_dir: Path, output_dir: Path) -> None:
    if processed_csv.exists():
        return
    print("Processed dataset missing. Running preprocessing first.")
    preprocess_phase1(data_dir, output_dir)


def build_label_mapping(labels: list[str]) -> tuple[list[str], dict[str, int]]:
    label_names = sorted(str(label) for label in labels)
    return label_names, {label: idx for idx, label in enumerate(label_names)}


def context_key(row: Any) -> str:
    return "|".join(
        [
            f"balls={row['balls']}",
            f"strikes={row['strikes']}",
            f"stand={row['stand']}",
            f"p_throws={row['p_throws']}",
            f"r2={row['runners_on_2b']}",
            f"times={row['n_thruorder_pitcher']}",
        ]
    )


def normalize_counts(counts: Counter[str], labels: list[str], smoothing: float) -> dict[str, float]:
    total = sum(counts.values()) + smoothing * len(labels)
    if total == 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: (counts.get(label, 0) + smoothing) / total for label in labels}


def train_hmm_task(
    pitcher_name: str,
    pitcher_df: Any,
    train_game_ids: set[str],
    test_game_ids: set[str],
    args: Any,
) -> dict[str, Any]:
    target_column = "target_pitch_type"
    target_counts = pitcher_df[target_column].value_counts()
    retained_labels = sorted(str(label) for label, count in target_counts.items() if int(count) >= args.min_class_count)
    dropped_labels = {str(label): int(count) for label, count in target_counts.items() if int(count) < args.min_class_count}
    task_df = pitcher_df[pitcher_df[target_column].astype(str).isin(retained_labels)].copy()

    if len(retained_labels) < 2:
        return {
            "task_name": "pitch_type",
            "status": "skipped",
            "skip_reason": "fewer_than_two_classes_after_filtering",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df))},
        }

    train_df = task_df[task_df["game_pk"].astype(str).isin(train_game_ids)].copy()
    test_df = task_df[task_df["game_pk"].astype(str).isin(test_game_ids)].copy()
    train_labels = sorted(str(value) for value in train_df[target_column].astype(str).unique().tolist())
    test_labels = set(str(value) for value in test_df[target_column].astype(str).unique().tolist())

    if len(train_labels) < 2 or test_df.empty:
        return {
            "task_name": "pitch_type",
            "status": "skipped",
            "skip_reason": "insufficient_class_coverage_after_split",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_df)), "test": int(len(test_df))},
        }
    if not test_labels.issubset(set(train_labels)):
        return {
            "task_name": "pitch_type",
            "status": "skipped",
            "skip_reason": "test_contains_unseen_class_after_split",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_df)), "test": int(len(test_df))},
        }

    label_names, label_to_index = build_label_mapping(train_labels)
    global_counts: Counter[str] = Counter()
    start_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_counts: dict[str, Counter[str]] = defaultdict(Counter)
    transition_context_counts: dict[str, Counter[str]] = defaultdict(Counter)

    sorted_train = train_df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number", "row_id"])
    for (_, _), at_bat_df in sorted_train.groupby(["game_pk", "at_bat_number"], sort=False):
        previous_pitch = None
        for _, row in at_bat_df.iterrows():
            pitch = str(row[target_column])
            key = context_key(row)
            global_counts[pitch] += 1
            context_counts[key][pitch] += 1
            if previous_pitch is None:
                start_counts[pitch] += 1
            else:
                transition_counts[previous_pitch][pitch] += 1
                transition_context_counts[f"{previous_pitch}|{key}"][pitch] += 1
            previous_pitch = pitch

    global_probs = normalize_counts(global_counts, label_names, args.smoothing)
    start_probs = normalize_counts(start_counts, label_names, args.smoothing)

    train_truth: list[int] = []
    train_pred: list[int] = []
    test_truth: list[int] = []
    test_pred: list[int] = []

    def predict_row(row: Any, previous_pitch: str | None) -> str:
        key = context_key(row)
        if previous_pitch is not None:
            combined_key = f"{previous_pitch}|{key}"
            if combined_key in transition_context_counts:
                counts = transition_context_counts[combined_key]
                return max(label_names, key=lambda label: counts.get(label, 0) + args.smoothing * global_probs[label])
            if previous_pitch in transition_counts:
                probs = normalize_counts(transition_counts[previous_pitch], label_names, args.smoothing)
                return max(label_names, key=lambda label: probs[label])
        if key in context_counts:
            probs = normalize_counts(context_counts[key], label_names, args.smoothing)
            return max(label_names, key=lambda label: probs[label])
        if previous_pitch is None:
            return max(label_names, key=lambda label: start_probs[label])
        return max(label_names, key=lambda label: global_probs[label])

    def evaluate_split(split_df: Any) -> tuple[list[int], list[int], list[int]]:
        truth: list[int] = []
        pred: list[int] = []
        row_ids: list[int] = []
        sorted_split = split_df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number", "row_id"])
        for (_, _), at_bat_df in sorted_split.groupby(["game_pk", "at_bat_number"], sort=False):
            previous_pitch = None
            for _, row in at_bat_df.iterrows():
                pitch = str(row[target_column])
                if pitch not in label_to_index:
                    previous_pitch = pitch
                    continue
                predicted = predict_row(row, previous_pitch)
                truth.append(label_to_index[pitch])
                pred.append(label_to_index[predicted])
                row_ids.append(int(row["row_id"]))
                previous_pitch = pitch
        return truth, pred, row_ids

    train_truth, train_pred, train_row_ids = evaluate_split(train_df)
    test_truth, test_pred, test_row_ids = evaluate_split(test_df)
    train_metrics = evaluate_predictions(train_truth, train_pred, label_names)
    test_metrics = evaluate_predictions(test_truth, test_pred, label_names)

    artifact_dir = args.artifacts_dir / slugify(pitcher_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "pitch_type_model.json"
    confusion_path = artifact_dir / "confusion_matrix_pitch_type.csv"

    write_summary(
        model_path,
        {
            "model": "supervised_markov_hmm_style",
            "pitcher_name": pitcher_name,
            "target_column": target_column,
            "label_names": label_names,
            "label_to_index": label_to_index,
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "smoothing": args.smoothing,
            "global_probs": global_probs,
            "start_probs": start_probs,
            "transition_counts": {key: dict(value) for key, value in transition_counts.items()},
            "context_counts": {key: dict(value) for key, value in context_counts.items()},
            "transition_context_counts": {key: dict(value) for key, value in transition_context_counts.items()},
        },
    )
    write_confusion_matrix(confusion_path, label_names, test_metrics["confusion_matrix"])

    return {
        "task_name": "pitch_type",
        "status": "trained",
        "target_column": target_column,
        "retained_labels": retained_labels,
        "dropped_labels": dropped_labels,
        "row_counts": {
            "total": int(len(task_df)),
            "train": int(len(train_truth)),
            "test": int(len(test_truth)),
        },
        "split": {
            "train_games": len(train_game_ids),
            "test_games": len(test_game_ids),
        },
        "sample_row_ids": {
            "train": sorted(train_row_ids),
            "test": sorted(test_row_ids),
        },
        "training": {
            "model": "supervised_markov_hmm_style",
            "smoothing": args.smoothing,
            "class_weighting": "disabled",
            "context_features": ["balls", "strikes", "stand", "p_throws", "runners_on_2b", "n_thruorder_pitcher"],
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model_path": str(model_path),
            "confusion_matrix_path": str(confusion_path),
        },
    }


def write_confusion_matrix(path: Path, label_names: list[str], confusion: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *label_names])
        for label_name, row in zip(label_names, confusion):
            writer.writerow([label_name, *row])


def skipped_location() -> dict[str, Any]:
    return {
        "task_name": "location",
        "status": "skipped",
        "skip_reason": "not_requested_pitch_type_focus",
        "retained_labels": [],
        "dropped_labels": {},
        "row_counts": {"total": 0, "train": 0, "test": 0},
    }


def build_pitcher_summary(
    pitcher_name: str,
    pitch_type_metrics: dict[str, Any],
    train_game_ids: set[str],
    test_game_ids: set[str],
) -> dict[str, Any]:
    return {
        "pitcher_name": pitcher_name,
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "train_games": len(train_game_ids),
            "test_games": len(test_game_ids),
            "overlap_games": len(train_game_ids.intersection(test_game_ids)),
        },
        "tasks": {
            "pitch_type": pitch_type_metrics,
            "location": skipped_location(),
        },
    }


def summarize_hmm_results(pitcher_summaries: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    accuracies = []
    macro_f1s = []
    weighted_f1s = []
    skipped = []
    for pitcher_summary in pitcher_summaries:
        task = pitcher_summary["tasks"]["pitch_type"]
        if task["status"] != "trained":
            skipped.append({"pitcher_name": pitcher_summary["pitcher_name"], "reason": task["skip_reason"]})
            continue
        accuracies.append(task["test_metrics"]["accuracy"])
        macro_f1s.append(task["test_metrics"]["macro_avg"]["f1"])
        weighted_f1s.append(task["test_metrics"]["weighted_avg"]["f1"])

    return {
        "model": "supervised_markov_hmm_style",
        "seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        "min_class_count": args.min_class_count,
        "smoothing": args.smoothing,
        "feature_columns": FEATURE_COLUMNS,
        "pitchers_processed": len(pitcher_summaries),
        "pitchers": pitcher_summaries,
        "aggregates": {
            "pitch_type": {
                "trained_pitchers": len(accuracies),
                "skipped_pitchers": skipped,
                "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
                "mean_macro_f1": sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0,
                "mean_weighted_f1": sum(weighted_f1s) / len(weighted_f1s) if weighted_f1s else 0.0,
            },
            "location": {
                "trained_pitchers": 0,
                "skipped_pitchers": [
                    {"pitcher_name": summary["pitcher_name"], "reason": "not_requested_pitch_type_focus"}
                    for summary in pitcher_summaries
                ],
                "mean_accuracy": 0.0,
                "mean_macro_f1": 0.0,
                "mean_weighted_f1": 0.0,
            },
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_hmm(args: Any) -> dict[str, Any]:
    np, pd = ensure_hmm_dependencies()

    processed_csv = args.output_dir / "phase1_cleaned_all_pitchers.csv"
    ensure_processed_dataset(processed_csv, args.data_dir, args.output_dir)

    df = pd.read_csv(processed_csv)
    df["row_id"] = np.arange(len(df))
    df["game_pk"] = df["game_pk"].astype(str)

    pitcher_groups = list(df.groupby("pitcher_name", sort=True))
    if args.max_pitchers is not None:
        pitcher_groups = pitcher_groups[: args.max_pitchers]

    pitcher_summaries = []
    for pitcher_number, (pitcher_name, pitcher_df) in enumerate(pitcher_groups, start=1):
        print(f"[{pitcher_number}/{len(pitcher_groups)}] Starting {pitcher_name}", flush=True)
        pitcher_df = pitcher_df.copy()
        train_game_ids, test_game_ids = chronological_game_split(pitcher_df, TRAIN_FRACTION)

        if not test_game_ids:
            pitch_type_metrics = {
                "task_name": "pitch_type",
                "status": "skipped",
                "skip_reason": "not_enough_games_for_chronological_split",
                "retained_labels": [],
                "dropped_labels": {},
                "row_counts": {"total": int(len(pitcher_df)), "train": 0, "test": 0},
            }
        else:
            pitch_type_metrics = train_hmm_task(
                pitcher_name=pitcher_name,
                pitcher_df=pitcher_df,
                train_game_ids=train_game_ids,
                test_game_ids=test_game_ids,
                args=args,
            )

        pitcher_summary = build_pitcher_summary(pitcher_name, pitch_type_metrics, train_game_ids, test_game_ids)
        write_summary(args.artifacts_dir / slugify(pitcher_name) / "metrics.json", pitcher_summary)
        pitcher_summaries.append(pitcher_summary)

        pitch_acc = pitch_type_metrics["test_metrics"]["accuracy"] if pitch_type_metrics["status"] == "trained" else 0.0
        print(f"Trained {pitcher_name}: pitch_type_acc={pitch_acc:.3f}", flush=True)

    summary = summarize_hmm_results(pitcher_summaries, args)
    write_summary(args.artifacts_dir / "summary.json", summary)
    print(f"HMM artifacts written to {args.artifacts_dir}")
    return summary
