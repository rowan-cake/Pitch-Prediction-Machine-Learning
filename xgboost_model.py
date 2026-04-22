"""Per-pitcher XGBoost baselines with flattened pitch-history features."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from data_preprocessing import (
    CATEGORICAL_FEATURE_COLUMNS,
    DATA_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    OUTPUT_DIR,
    preprocess_phase1,
    slugify,
    write_summary,
)
from logistic_regression import evaluate_predictions
from lstm_model import TRAIN_FRACTION, chronological_game_split


ARTIFACTS_DIR = Path("artifacts/xgboost")
DEFAULT_SEED = 440
MIN_CLASS_COUNT = 20
DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_NUM_BOOST_ROUND = 500
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_MAX_DEPTH = 6
DEFAULT_SUBSAMPLE = 0.9
DEFAULT_COLSAMPLE_BYTREE = 0.9


def ensure_xgboost_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing XGBoost dependencies. Install them with `python3 -m pip install -r requirements.txt` "
            "before running `train-xgboost`."
        ) from exc
    return np, pd, xgb


def ensure_processed_dataset(processed_csv: Path, data_dir: Path, output_dir: Path) -> None:
    if processed_csv.exists():
        return
    print("Processed dataset missing. Running preprocessing first.")
    preprocess_phase1(data_dir, output_dir)


def build_context_preprocessor(train_df: Any) -> dict[str, Any]:
    categorical_vocabularies = {
        column: sorted(str(value) for value in train_df[column].astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURE_COLUMNS
    }
    context_feature_names = list(NUMERIC_FEATURE_COLUMNS)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        context_feature_names.extend(f"{column}={value}" for value in categorical_vocabularies[column])

    return {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_FEATURE_COLUMNS,
        "categorical_vocabularies": categorical_vocabularies,
        "context_feature_names": context_feature_names,
        "context_size": len(context_feature_names),
    }


def transform_context_rows(df: Any, preprocessor: dict[str, Any], np: Any) -> dict[int, Any]:
    parts = [df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=float).astype(np.float32)]
    for column in CATEGORICAL_FEATURE_COLUMNS:
        vocab = preprocessor["categorical_vocabularies"][column]
        index = {value: idx for idx, value in enumerate(vocab)}
        encoded = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for row_idx, value in enumerate(df[column].astype(str).tolist()):
            idx = index.get(value)
            if idx is not None:
                encoded[row_idx, idx] = 1.0
        parts.append(encoded)

    matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return {int(row_id): matrix[index] for index, row_id in enumerate(df["row_id"].tolist())}


def one_hot(value: str, vocab_index: dict[str, int], np: Any) -> Any:
    encoded = np.zeros(len(vocab_index), dtype=np.float32)
    idx = vocab_index.get(str(value))
    if idx is not None:
        encoded[idx] = 1.0
    return encoded


def build_label_mapping(labels: list[str]) -> tuple[list[str], dict[str, int]]:
    label_names = sorted(str(label) for label in labels)
    return label_names, {label: idx for idx, label in enumerate(label_names)}


def build_feature_names(preprocessor: dict[str, Any], sequence_length: int) -> list[str]:
    names = [f"current_{name}" for name in preprocessor["context_feature_names"]]
    history_base_names = (
        preprocessor["context_feature_names"]
        + [f"prior_pitch_type={label}" for label in preprocessor["prior_pitch_type_vocab"]]
        + [f"prior_location={label}" for label in preprocessor["prior_location_vocab"]]
    )
    for lag in range(sequence_length, 0, -1):
        names.extend(f"lag_{lag}_{name}" for name in history_base_names)
    return names


def build_flattened_samples(
    task_df: Any,
    all_pitcher_df: Any,
    context_by_row_id: dict[int, Any],
    label_to_index: dict[str, int],
    target_column: str,
    train_game_ids: set[str],
    test_game_ids: set[str],
    preprocessor: dict[str, Any],
    args: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pitch_type_index = {label: idx for idx, label in enumerate(preprocessor["prior_pitch_type_vocab"])}
    location_index = {label: idx for idx, label in enumerate(preprocessor["prior_location_vocab"])}
    history_feature_size = preprocessor["history_feature_size"]
    zero_history = np.zeros((args.sequence_length, history_feature_size), dtype=np.float32)

    samples = {
        "train": {"features": [], "labels": [], "row_ids": []},
        "test": {"features": [], "labels": [], "row_ids": []},
    }
    allowed_row_ids = set(int(row_id) for row_id in task_df["row_id"].tolist())

    sorted_df = all_pitcher_df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number", "row_id"])
    for (_, _), at_bat_df in sorted_df.groupby(["game_pk", "at_bat_number"], sort=False):
        previous_rows: list[Any] = []
        for _, row in at_bat_df.iterrows():
            row_id = int(row["row_id"])
            game_id = str(row["game_pk"])

            if row_id in allowed_row_ids:
                split = "train" if game_id in train_game_ids else "test" if game_id in test_game_ids else None
                target_value = str(row[target_column])
                if split is not None and target_value in label_to_index:
                    history_rows = previous_rows[-args.sequence_length :]
                    history = zero_history.copy()
                    start = args.sequence_length - len(history_rows)
                    for offset, prior_row in enumerate(history_rows):
                        prior_row_id = int(prior_row["row_id"])
                        history[start + offset] = np.concatenate(
                            [
                                context_by_row_id[prior_row_id],
                                one_hot(prior_row["target_pitch_type"], pitch_type_index, np),
                                one_hot(prior_row["target_location_bucket"], location_index, np),
                            ]
                        ).astype(np.float32)

                    features = np.concatenate([context_by_row_id[row_id], history.reshape(-1)]).astype(np.float32)
                    samples[split]["features"].append(features)
                    samples[split]["labels"].append(label_to_index[target_value])
                    samples[split]["row_ids"].append(row_id)

            previous_rows.append(row)

    result = {}
    feature_count = preprocessor["feature_count"]
    for split, split_samples in samples.items():
        result[split] = {
            "features": np.stack(split_samples["features"]).astype(np.float32)
            if split_samples["features"]
            else np.zeros((0, feature_count), dtype=np.float32),
            "labels": np.array(split_samples["labels"], dtype=np.int64),
            "row_ids": split_samples["row_ids"],
        }
    return result["train"], result["test"]


def write_confusion_matrix(path: Path, label_names: list[str], confusion: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *label_names])
        for label_name, row in zip(label_names, confusion):
            writer.writerow([label_name, *row])


def train_xgboost_task(
    task_name: str,
    pitcher_name: str,
    pitcher_df: Any,
    train_game_ids: set[str],
    test_game_ids: set[str],
    target_column: str,
    args: Any,
    np: Any,
    xgb: Any,
) -> dict[str, Any]:
    target_counts = pitcher_df[target_column].value_counts()
    retained_labels = sorted(str(label) for label, count in target_counts.items() if int(count) >= args.min_class_count)
    dropped_labels = {str(label): int(count) for label, count in target_counts.items() if int(count) < args.min_class_count}
    task_df = pitcher_df[pitcher_df[target_column].astype(str).isin(retained_labels)].copy()

    if len(retained_labels) < 2:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "fewer_than_two_classes_after_filtering",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df))},
        }

    train_source_df = pitcher_df[pitcher_df["game_pk"].astype(str).isin(train_game_ids)].copy()
    train_task_df = task_df[task_df["game_pk"].astype(str).isin(train_game_ids)].copy()
    test_task_df = task_df[task_df["game_pk"].astype(str).isin(test_game_ids)].copy()
    train_labels = sorted(str(value) for value in train_task_df[target_column].astype(str).unique().tolist())
    test_labels = set(str(value) for value in test_task_df[target_column].astype(str).unique().tolist())

    if len(train_labels) < 2 or test_task_df.empty:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "insufficient_class_coverage_after_split",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_task_df)), "test": int(len(test_task_df))},
        }
    if not test_labels.issubset(set(train_labels)):
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "test_contains_unseen_class_after_split",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_task_df)), "test": int(len(test_task_df))},
        }

    label_names, label_to_index = build_label_mapping(train_labels)
    preprocessor = build_context_preprocessor(train_source_df)
    preprocessor["prior_pitch_type_vocab"] = sorted(str(value) for value in train_source_df["target_pitch_type"].astype(str).unique().tolist())
    preprocessor["prior_location_vocab"] = sorted(str(value) for value in train_source_df["target_location_bucket"].astype(str).unique().tolist())
    preprocessor["history_feature_size"] = (
        preprocessor["context_size"]
        + len(preprocessor["prior_pitch_type_vocab"])
        + len(preprocessor["prior_location_vocab"])
    )
    preprocessor["sequence_length"] = args.sequence_length
    preprocessor["feature_names"] = build_feature_names(preprocessor, args.sequence_length)
    preprocessor["feature_count"] = len(preprocessor["feature_names"])

    context_by_row_id = transform_context_rows(pitcher_df, preprocessor, np)
    train_samples, test_samples = build_flattened_samples(
        task_df=task_df,
        all_pitcher_df=pitcher_df,
        context_by_row_id=context_by_row_id,
        label_to_index=label_to_index,
        target_column=target_column,
        train_game_ids=train_game_ids,
        test_game_ids=test_game_ids,
        preprocessor=preprocessor,
        args=args,
        np=np,
    )

    if len(train_samples["labels"]) == 0 or len(test_samples["labels"]) == 0:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "empty_flattened_samples",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_samples["labels"])), "test": int(len(test_samples["labels"]))},
        }

    params = {
        "objective": "multi:softprob",
        "num_class": len(label_names),
        "eval_metric": "mlogloss",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": "hist",
        "seed": args.seed,
    }
    train_matrix = xgb.DMatrix(
        train_samples["features"],
        label=train_samples["labels"],
        feature_names=preprocessor["feature_names"],
    )
    test_matrix = xgb.DMatrix(
        test_samples["features"],
        label=test_samples["labels"],
        feature_names=preprocessor["feature_names"],
    )
    model = xgb.train(params=params, dtrain=train_matrix, num_boost_round=args.num_boost_round, verbose_eval=False)

    train_predictions = model.predict(train_matrix).argmax(axis=1).tolist()
    test_predictions = model.predict(test_matrix).argmax(axis=1).tolist()
    train_metrics = evaluate_predictions(train_samples["labels"].tolist(), train_predictions, label_names)
    test_metrics = evaluate_predictions(test_samples["labels"].tolist(), test_predictions, label_names)

    artifact_dir = args.artifacts_dir / slugify(pitcher_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{task_name}_model.json"
    preprocessor_path = artifact_dir / f"{task_name}_preprocessor.json"
    confusion_path = artifact_dir / f"confusion_matrix_{task_name}.csv"

    model.save_model(model_path)
    write_summary(
        preprocessor_path,
        {
            "pitcher_name": pitcher_name,
            "task_name": task_name,
            "target_column": target_column,
            "label_names": label_names,
            "label_to_index": label_to_index,
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            **preprocessor,
        },
    )
    write_confusion_matrix(confusion_path, label_names, test_metrics["confusion_matrix"])

    return {
        "task_name": task_name,
        "status": "trained",
        "target_column": target_column,
        "retained_labels": retained_labels,
        "dropped_labels": dropped_labels,
        "row_counts": {
            "total": int(len(task_df)),
            "train": int(len(train_samples["labels"])),
            "test": int(len(test_samples["labels"])),
        },
        "split": {
            "train_games": len(train_game_ids),
            "test_games": len(test_game_ids),
        },
        "sample_row_ids": {
            "train": sorted(int(row_id) for row_id in train_samples["row_ids"]),
            "test": sorted(int(row_id) for row_id in test_samples["row_ids"]),
        },
        "training": {
            "model": "xgboost",
            "num_boost_round": args.num_boost_round,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "sequence_length": args.sequence_length,
            "class_weighting": "disabled",
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "confusion_matrix_path": str(confusion_path),
        },
    }


def build_pitcher_summary(
    pitcher_name: str,
    pitch_type_metrics: dict[str, Any],
    location_metrics: dict[str, Any],
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
            "location": location_metrics,
        },
    }


def summarize_xgboost_results(pitcher_summaries: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    summary = {
        "model": "xgboost",
        "seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        "min_class_count": args.min_class_count,
        "sequence_length": args.sequence_length,
        "num_boost_round": args.num_boost_round,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "include_location": args.include_location,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "pitchers_processed": len(pitcher_summaries),
        "pitchers": pitcher_summaries,
        "aggregates": {},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    for task_key in ("pitch_type", "location"):
        accuracies = []
        macro_f1s = []
        weighted_f1s = []
        skipped = []
        for pitcher_summary in pitcher_summaries:
            task = pitcher_summary["tasks"][task_key]
            if task["status"] != "trained":
                skipped.append({"pitcher_name": pitcher_summary["pitcher_name"], "reason": task["skip_reason"]})
                continue
            accuracies.append(task["test_metrics"]["accuracy"])
            macro_f1s.append(task["test_metrics"]["macro_avg"]["f1"])
            weighted_f1s.append(task["test_metrics"]["weighted_avg"]["f1"])

        summary["aggregates"][task_key] = {
            "trained_pitchers": len(accuracies),
            "skipped_pitchers": skipped,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "mean_macro_f1": sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0,
            "mean_weighted_f1": sum(weighted_f1s) / len(weighted_f1s) if weighted_f1s else 0.0,
        }
    return summary


def run_xgboost(args: Any) -> dict[str, Any]:
    np, pd, xgb = ensure_xgboost_dependencies()

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
            skipped = {
                "status": "skipped",
                "skip_reason": "not_enough_games_for_chronological_split",
                "retained_labels": [],
                "dropped_labels": {},
                "row_counts": {"total": int(len(pitcher_df)), "train": 0, "test": 0},
            }
            pitcher_summary = build_pitcher_summary(
                pitcher_name,
                {"task_name": "pitch_type", **skipped},
                {"task_name": "location", **skipped},
                train_game_ids,
                test_game_ids,
            )
            pitcher_summaries.append(pitcher_summary)
            continue

        pitch_type_metrics = train_xgboost_task(
            task_name="pitch_type",
            pitcher_name=pitcher_name,
            pitcher_df=pitcher_df,
            train_game_ids=train_game_ids,
            test_game_ids=test_game_ids,
            target_column="target_pitch_type",
            args=args,
            np=np,
            xgb=xgb,
        )
        if args.include_location:
            location_metrics = train_xgboost_task(
                task_name="location",
                pitcher_name=pitcher_name,
                pitcher_df=pitcher_df,
                train_game_ids=train_game_ids,
                test_game_ids=test_game_ids,
                target_column="target_location_bucket",
                args=args,
                np=np,
                xgb=xgb,
            )
        else:
            location_metrics = {
                "task_name": "location",
                "status": "skipped",
                "skip_reason": "not_requested_pitch_type_focus",
                "retained_labels": [],
                "dropped_labels": {},
                "row_counts": {"total": int(len(pitcher_df)), "train": 0, "test": 0},
            }

        pitcher_summary = build_pitcher_summary(
            pitcher_name,
            pitch_type_metrics,
            location_metrics,
            train_game_ids,
            test_game_ids,
        )
        write_summary(args.artifacts_dir / slugify(pitcher_name) / "metrics.json", pitcher_summary)
        pitcher_summaries.append(pitcher_summary)

        pitch_acc = pitch_type_metrics["test_metrics"]["accuracy"] if pitch_type_metrics["status"] == "trained" else 0.0
        location_acc = location_metrics["test_metrics"]["accuracy"] if location_metrics["status"] == "trained" else 0.0
        print(f"Trained {pitcher_name}: pitch_type_acc={pitch_acc:.3f} location_acc={location_acc:.3f}", flush=True)

    summary = summarize_xgboost_results(pitcher_summaries, args)
    write_summary(args.artifacts_dir / "summary.json", summary)
    print(f"XGBoost artifacts written to {args.artifacts_dir}")
    return summary
