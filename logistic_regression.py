"""Per-pitcher multiclass logistic-regression training pipeline."""

from __future__ import annotations

import csv
import random
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


ARTIFACTS_DIR = Path("artifacts/logreg")
DEFAULT_SEED = 440
MIN_CLASS_COUNT = 20
TRAIN_FRACTION = 0.8
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 0.05


def ensure_ml_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing ML dependencies. Install them with `python3 -m pip install -r requirements.txt` "
            "before running `train-logreg`."
        ) from exc
    return np, pd, torch, nn


def set_global_seed(seed: int, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_processed_dataset(processed_csv: Path, data_dir: Path, output_dir: Path) -> None:
    if processed_csv.exists():
        return
    print("Processed dataset missing. Running preprocessing first.")
    preprocess_phase1(data_dir, output_dir)


def stratified_split_indices(
    labels: list[str],
    seed: int,
    train_fraction: float,
    np: Any,
) -> tuple[list[int], list[int]]:
    grouped_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped_indices.setdefault(label, []).append(index)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label, indices in sorted(grouped_indices.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(f"Label {label!r} does not have enough rows for a split.")
        proposed_train = int(round(len(shuffled) * train_fraction))
        train_count = min(max(proposed_train, 1), len(shuffled) - 1)
        train_indices.extend(shuffled[:train_count])
        test_indices.extend(shuffled[train_count:])

    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


def build_preprocessor(
    train_df: Any,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        values = train_df[column].to_numpy(dtype=float)
        mean_value = float(values.mean()) if len(values) else 0.0
        std_value = float(values.std()) if len(values) else 1.0
        if std_value == 0.0:
            std_value = 1.0
        stats[column] = {"mean": mean_value, "std": std_value}

    vocabularies: dict[str, list[str]] = {}
    for column in categorical_columns:
        categories = sorted(str(value) for value in train_df[column].astype(str).unique().tolist())
        vocabularies[column] = categories

    feature_size = len(numeric_columns) + sum(len(vocabularies[column]) for column in categorical_columns)
    return {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_stats": stats,
        "categorical_vocabularies": vocabularies,
        "feature_size": feature_size,
    }


def transform_features(df: Any, preprocessor: dict[str, Any], np: Any) -> Any:
    numeric_parts = []
    for column in preprocessor["numeric_columns"]:
        stats = preprocessor["numeric_stats"][column]
        values = df[column].to_numpy(dtype=float)
        normalized = (values - stats["mean"]) / stats["std"]
        numeric_parts.append(normalized.reshape(-1, 1))

    categorical_parts = []
    for column in preprocessor["categorical_columns"]:
        categories = preprocessor["categorical_vocabularies"][column]
        category_index = {category: idx for idx, category in enumerate(categories)}
        encoded = np.zeros((len(df), len(categories)), dtype=np.float32)
        values = df[column].astype(str).tolist()
        for row_idx, value in enumerate(values):
            idx = category_index.get(value)
            if idx is not None:
                encoded[row_idx, idx] = 1.0
        categorical_parts.append(encoded)

    parts = [part.astype(np.float32) for part in numeric_parts]
    parts.extend(categorical_parts)
    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def encode_labels(series: Any) -> tuple[Any, list[str], dict[str, int]]:
    labels = sorted(str(value) for value in series.astype(str).unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    encoded = series.astype(str).map(label_to_index).to_numpy(dtype=int)
    return encoded, labels, label_to_index


def build_class_weights(encoded_labels: Any, num_classes: int, np: Any, torch: Any) -> Any:
    counts = np.bincount(encoded_labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = len(encoded_labels) / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


def make_batches(num_rows: int, batch_size: int, seed: int, epoch: int, np: Any) -> list[Any]:
    rng = np.random.default_rng(seed + epoch)
    indices = np.arange(num_rows)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, num_rows, batch_size)]


def evaluate_predictions(
    true_labels: list[int],
    pred_labels: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    total = len(true_labels)
    correct = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == pred)
    accuracy = correct / total if total else 0.0

    confusion = [[0 for _ in label_names] for _ in label_names]
    for truth, pred in zip(true_labels, pred_labels):
        confusion[truth][pred] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for index, label in enumerate(label_names):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(label_names)) if row != index)
        fn = sum(confusion[index][col] for col in range(len(label_names)) if col != index)
        support = sum(confusion[index])

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    macro = {
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
    weighted = {
        "precision": weighted_precision / total if total else 0.0,
        "recall": weighted_recall / total if total else 0.0,
        "f1": weighted_f1 / total if total else 0.0,
    }

    return {
        "accuracy": accuracy,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def train_task_model(
    task_name: str,
    pitcher_name: str,
    df: Any,
    train_row_ids: set[int],
    test_row_ids: set[int],
    target_column: str,
    args: Any,
    np: Any,
    torch: Any,
    nn: Any,
) -> dict[str, Any]:
    target_counts = df[target_column].value_counts()
    kept_labels = sorted(label for label, count in target_counts.items() if int(count) >= args.min_class_count)
    dropped_labels = {str(label): int(count) for label, count in target_counts.items() if int(count) < args.min_class_count}
    task_df = df[df[target_column].isin(kept_labels)].copy()

    if len(kept_labels) < 2:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "fewer_than_two_classes_after_filtering",
            "retained_labels": kept_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df))},
        }

    task_df["split"] = task_df["row_id"].map(lambda row_id: "train" if row_id in train_row_ids else "test")
    train_df = task_df[task_df["split"] == "train"].copy()
    test_df = task_df[task_df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "empty_train_or_test_after_split",
            "retained_labels": kept_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_df)), "test": int(len(test_df))},
        }

    train_labels = set(str(value) for value in train_df[target_column].astype(str).tolist())
    test_labels = set(str(value) for value in test_df[target_column].astype(str).tolist())

    if train_df[target_column].nunique() < 2 or test_df[target_column].nunique() < 1:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "insufficient_class_coverage_after_split",
            "retained_labels": kept_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_df)), "test": int(len(test_df))},
        }
    if not test_labels.issubset(train_labels):
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "test_contains_unseen_class_after_split",
            "retained_labels": kept_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_df)), "test": int(len(test_df))},
        }

    preprocessor = build_preprocessor(
        train_df=train_df,
        feature_columns=FEATURE_COLUMNS,
        numeric_columns=NUMERIC_FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_FEATURE_COLUMNS,
    )
    x_train = transform_features(train_df, preprocessor, np)
    x_test = transform_features(test_df, preprocessor, np)
    y_train, label_names, label_to_index = encode_labels(train_df[target_column])
    y_test = test_df[target_column].astype(str).map(label_to_index).to_numpy(dtype=int)

    class_weights = build_class_weights(y_train, len(label_names), np, torch)
    model = nn.Linear(preprocessor["feature_size"], len(label_names))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    epoch_losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for batch_indices in make_batches(len(train_df), args.batch_size, args.seed, epoch, np):
            inputs = x_train_tensor[batch_indices]
            targets = y_train_tensor[batch_indices]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    model.eval()
    with torch.no_grad():
        train_logits = model(x_train_tensor)
        test_logits = model(x_test_tensor)
        train_loss = float(loss_fn(train_logits, y_train_tensor).item())
        test_loss = float(loss_fn(test_logits, y_test_tensor).item())
        train_predictions = train_logits.argmax(dim=1).cpu().numpy().tolist()
        test_predictions = test_logits.argmax(dim=1).cpu().numpy().tolist()

    train_metrics = evaluate_predictions(y_train.tolist(), train_predictions, label_names)
    test_metrics = evaluate_predictions(y_test.tolist(), test_predictions, label_names)

    artifact_dir = args.artifacts_dir / slugify(pitcher_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / f"{task_name}_model.pt"
    preprocessor_path = artifact_dir / f"{task_name}_preprocessor.json"
    confusion_path = artifact_dir / f"confusion_matrix_{task_name}.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": preprocessor["feature_size"],
            "output_dim": len(label_names),
            "label_names": label_names,
            "target_column": target_column,
            "feature_columns": FEATURE_COLUMNS,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        model_path,
    )

    write_summary(
        preprocessor_path,
        {
            "pitcher_name": pitcher_name,
            "task_name": task_name,
            "target_column": target_column,
            "feature_columns": preprocessor["feature_columns"],
            "numeric_columns": preprocessor["numeric_columns"],
            "categorical_columns": preprocessor["categorical_columns"],
            "numeric_stats": preprocessor["numeric_stats"],
            "categorical_vocabularies": preprocessor["categorical_vocabularies"],
            "feature_size": preprocessor["feature_size"],
            "label_names": label_names,
            "label_to_index": label_to_index,
            "retained_labels": kept_labels,
            "dropped_labels": dropped_labels,
        },
    )

    with confusion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *label_names])
        for label_name, row in zip(label_names, test_metrics["confusion_matrix"]):
            writer.writerow([label_name, *row])

    return {
        "task_name": task_name,
        "status": "trained",
        "target_column": target_column,
        "retained_labels": kept_labels,
        "dropped_labels": dropped_labels,
        "row_counts": {
            "total": int(len(task_df)),
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "split_row_ids": {
            "train": sorted(int(row_id) for row_id in train_df["row_id"].tolist()),
            "test": sorted(int(row_id) for row_id in test_df["row_id"].tolist()),
        },
        "training": {
            "optimizer": "SGD",
            "loss": "CrossEntropyLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "class_weights": {label: float(class_weights[idx].item()) for idx, label in enumerate(label_names)},
            "final_epoch_loss": epoch_losses[-1],
            "train_loss": train_loss,
            "test_loss": test_loss,
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
    train_size: int,
    test_size: int,
) -> dict[str, Any]:
    return {
        "pitcher_name": pitcher_name,
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "train_rows": train_size,
            "test_rows": test_size,
        },
        "tasks": {
            "pitch_type": pitch_type_metrics,
            "location": location_metrics,
        },
    }


def summarize_logreg_results(
    pitcher_summaries: list[dict[str, Any]],
    args: Any,
) -> dict[str, Any]:
    aggregate = {
        "seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        "min_class_count": args.min_class_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
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
                skipped.append(
                    {
                        "pitcher_name": pitcher_summary["pitcher_name"],
                        "reason": task["skip_reason"],
                    }
                )
                continue
            accuracies.append(task["test_metrics"]["accuracy"])
            macro_f1s.append(task["test_metrics"]["macro_avg"]["f1"])
            weighted_f1s.append(task["test_metrics"]["weighted_avg"]["f1"])
        aggregate["aggregates"][task_key] = {
            "trained_pitchers": len(accuracies),
            "skipped_pitchers": skipped,
            "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "mean_macro_f1": sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0,
            "mean_weighted_f1": sum(weighted_f1s) / len(weighted_f1s) if weighted_f1s else 0.0,
        }

    return aggregate


def run_logistic_regression(args: Any) -> dict[str, Any]:
    np, pd, torch, nn = ensure_ml_dependencies()
    set_global_seed(args.seed, np, torch)

    processed_csv = args.output_dir / "phase1_cleaned_all_pitchers.csv"
    ensure_processed_dataset(processed_csv, args.data_dir, args.output_dir)

    df = pd.read_csv(processed_csv)
    df["row_id"] = np.arange(len(df))

    pitcher_summaries: list[dict[str, Any]] = []
    for pitcher_name, pitcher_df in df.groupby("pitcher_name", sort=True):
        pitch_type_counts = pitcher_df["target_pitch_type"].value_counts()
        split_eligible_labels = sorted(
            label for label, count in pitch_type_counts.items() if int(count) >= args.min_class_count
        )
        split_df = pitcher_df[pitcher_df["target_pitch_type"].isin(split_eligible_labels)].copy()

        if split_df["target_pitch_type"].nunique() < 2:
            skipped_task = {
                "task_name": "pitch_type",
                "status": "skipped",
                "skip_reason": "fewer_than_two_pitch_type_classes_for_split",
                "retained_labels": split_eligible_labels,
                "dropped_labels": {
                    str(label): int(count)
                    for label, count in pitch_type_counts.items()
                    if int(count) < args.min_class_count
                },
                "row_counts": {"total": int(len(split_df)), "train": 0, "test": 0},
            }
            pitcher_summaries.append(
                build_pitcher_summary(
                    pitcher_name=pitcher_name,
                    pitch_type_metrics=skipped_task,
                    location_metrics={
                        "task_name": "location",
                        "status": "skipped",
                        "skip_reason": "pitch_type_split_not_available",
                        "retained_labels": [],
                        "dropped_labels": {},
                        "row_counts": {"total": 0, "train": 0, "test": 0},
                    },
                    train_size=0,
                    test_size=0,
                )
            )
            continue

        train_split_indices, test_split_indices = stratified_split_indices(
            labels=split_df["target_pitch_type"].astype(str).tolist(),
            seed=args.seed,
            train_fraction=TRAIN_FRACTION,
            np=np,
        )
        split_train_row_ids = set(int(row_id) for row_id in split_df.iloc[train_split_indices]["row_id"].tolist())
        split_test_row_ids = set(int(row_id) for row_id in split_df.iloc[test_split_indices]["row_id"].tolist())

        pitch_type_metrics = train_task_model(
            task_name="pitch_type",
            pitcher_name=pitcher_name,
            df=split_df,
            train_row_ids=split_train_row_ids,
            test_row_ids=split_test_row_ids,
            target_column="target_pitch_type",
            args=args,
            np=np,
            torch=torch,
            nn=nn,
        )
        location_metrics = train_task_model(
            task_name="location",
            pitcher_name=pitcher_name,
            df=split_df,
            train_row_ids=split_train_row_ids,
            test_row_ids=split_test_row_ids,
            target_column="target_location_bucket",
            args=args,
            np=np,
            torch=torch,
            nn=nn,
        )

        pitcher_summary = build_pitcher_summary(
            pitcher_name=pitcher_name,
            pitch_type_metrics=pitch_type_metrics,
            location_metrics=location_metrics,
            train_size=len(split_train_row_ids),
            test_size=len(split_test_row_ids),
        )
        write_summary(args.artifacts_dir / slugify(pitcher_name) / "metrics.json", pitcher_summary)
        pitcher_summaries.append(pitcher_summary)

        pitch_acc = pitch_type_metrics["test_metrics"]["accuracy"] if pitch_type_metrics["status"] == "trained" else 0.0
        location_acc = location_metrics["test_metrics"]["accuracy"] if location_metrics["status"] == "trained" else 0.0
        print(f"Trained {pitcher_name}: pitch_type_acc={pitch_acc:.3f} location_acc={location_acc:.3f}")

    summary = summarize_logreg_results(pitcher_summaries, args)
    write_summary(args.artifacts_dir / "summary.json", summary)
    print(f"Logistic-regression artifacts written to {args.artifacts_dir}")
    return summary
