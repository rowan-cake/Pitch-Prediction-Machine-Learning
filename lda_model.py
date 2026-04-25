"""Per-pitcher LDA baseline with flattened pitch-history features."""

from __future__ import annotations

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
from xgboost_model import (
    DEFAULT_SEQUENCE_LENGTH,
    MIN_CLASS_COUNT,
    build_context_preprocessor,
    build_feature_names,
    build_flattened_samples,
    build_label_mapping,
    transform_context_rows,
    write_confusion_matrix,
)


ARTIFACTS_DIR = Path("artifacts/lda")
DEFAULT_SEED = 440
DEFAULT_SOLVER = "lsqr"
DEFAULT_SHRINKAGE = "auto"


def ensure_lda_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing LDA dependencies. Install them with `python3 -m pip install -r requirements.txt` "
            "before running `train-lda`."
        ) from exc
    return np, pd, joblib, (LinearDiscriminantAnalysis, StandardScaler)


def ensure_processed_dataset(processed_csv: Path, data_dir: Path, output_dir: Path) -> None:
    if processed_csv.exists():
        return
    print("Processed dataset missing. Running preprocessing first.")
    preprocess_phase1(data_dir, output_dir)


def skipped_location_task(reason: str, total_rows: int = 0) -> dict[str, Any]:
    return {
        "task_name": "location",
        "status": "skipped",
        "skip_reason": reason,
        "retained_labels": [],
        "dropped_labels": {},
        "row_counts": {"total": total_rows, "train": 0, "test": 0},
    }


def normalize_shrinkage(value: str | float | None, solver: str) -> str | float | None:
    if solver == "svd":
        return None
    if value is None:
        return None
    if isinstance(value, float):
        return value

    lowered = value.strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    if lowered == "auto":
        return "auto"
    try:
        shrinkage = float(lowered)
    except ValueError as exc:
        raise SystemExit("--shrinkage must be 'auto', 'none', or a float between 0 and 1.") from exc
    if not 0.0 <= shrinkage <= 1.0:
        raise SystemExit("--shrinkage float must be between 0 and 1.")
    return shrinkage


def train_lda_task(
    pitcher_name: str,
    pitcher_df: Any,
    train_game_ids: set[str],
    test_game_ids: set[str],
    args: Any,
    np: Any,
    joblib: Any,
    LinearDiscriminantAnalysis: Any,
    StandardScaler: Any,
) -> dict[str, Any]:
    task_name = "pitch_type"
    target_column = "target_pitch_type"
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
            "row_counts": {
                "total": int(len(task_df)),
                "train": int(len(train_samples["labels"])),
                "test": int(len(test_samples["labels"])),
            },
        }

    shrinkage = normalize_shrinkage(args.shrinkage, args.solver)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_samples["features"])
    x_test = scaler.transform(test_samples["features"])
    model = LinearDiscriminantAnalysis(solver=args.solver, shrinkage=shrinkage)
    model.fit(x_train, train_samples["labels"])

    train_predictions = model.predict(x_train).tolist()
    test_predictions = model.predict(x_test).tolist()
    train_metrics = evaluate_predictions(train_samples["labels"].tolist(), train_predictions, label_names)
    test_metrics = evaluate_predictions(test_samples["labels"].tolist(), test_predictions, label_names)

    artifact_dir = args.artifacts_dir / slugify(pitcher_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "pitch_type_model.joblib"
    preprocessor_path = artifact_dir / "pitch_type_preprocessor.json"
    confusion_path = artifact_dir / "confusion_matrix_pitch_type.csv"

    joblib.dump({"model": model, "scaler": scaler}, model_path)
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
            "scaler": "StandardScaler",
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
            "model": "linear_discriminant_analysis",
            "solver": args.solver,
            "shrinkage": shrinkage,
            "sequence_length": args.sequence_length,
            "scaler": "StandardScaler",
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
            "location": skipped_location_task("not_requested_pitch_type_focus"),
        },
    }


def summarize_lda_results(pitcher_summaries: list[dict[str, Any]], args: Any) -> dict[str, Any]:
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
        "model": "linear_discriminant_analysis",
        "seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        "min_class_count": args.min_class_count,
        "sequence_length": args.sequence_length,
        "solver": args.solver,
        "shrinkage": normalize_shrinkage(args.shrinkage, args.solver),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
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


def run_lda(args: Any) -> dict[str, Any]:
    np, pd, joblib, sklearn_objects = ensure_lda_dependencies()
    LinearDiscriminantAnalysis, StandardScaler = sklearn_objects

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
            pitch_type_metrics = train_lda_task(
                pitcher_name=pitcher_name,
                pitcher_df=pitcher_df,
                train_game_ids=train_game_ids,
                test_game_ids=test_game_ids,
                args=args,
                np=np,
                joblib=joblib,
                LinearDiscriminantAnalysis=LinearDiscriminantAnalysis,
                StandardScaler=StandardScaler,
            )

        pitcher_summary = build_pitcher_summary(pitcher_name, pitch_type_metrics, train_game_ids, test_game_ids)
        write_summary(args.artifacts_dir / slugify(pitcher_name) / "metrics.json", pitcher_summary)
        pitcher_summaries.append(pitcher_summary)

        pitch_acc = pitch_type_metrics["test_metrics"]["accuracy"] if pitch_type_metrics["status"] == "trained" else 0.0
        print(f"Trained {pitcher_name}: pitch_type_acc={pitch_acc:.3f}", flush=True)

    summary = summarize_lda_results(pitcher_summaries, args)
    write_summary(args.artifacts_dir / "summary.json", summary)
    print(f"LDA artifacts written to {args.artifacts_dir}")
    return summary
