"""Per-pitcher LSTM sequence baseline for pitch prediction."""

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
from logistic_regression import ensure_ml_dependencies, evaluate_predictions, set_global_seed


ARTIFACTS_DIR = Path("artifacts/lstm")
DEFAULT_SEED = 440
MIN_CLASS_COUNT = 20
TRAIN_FRACTION = 0.8
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.001
DEFAULT_OPTIMIZER = "adam"
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.3
DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_GRAD_CLIP = 1.0


def ensure_processed_dataset(processed_csv: Path, data_dir: Path, output_dir: Path) -> None:
    if processed_csv.exists():
        return
    print("Processed dataset missing. Running preprocessing first.")
    preprocess_phase1(data_dir, output_dir)


def chronological_game_split(pitcher_df: Any, train_fraction: float) -> tuple[set[str], set[str]]:
    game_order = (
        pitcher_df[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk"])
        ["game_pk"]
        .astype(str)
        .tolist()
    )
    if len(game_order) < 2:
        return set(game_order), set()

    train_count = int(round(len(game_order) * train_fraction))
    train_count = min(max(train_count, 1), len(game_order) - 1)
    return set(game_order[:train_count]), set(game_order[train_count:])


def build_context_preprocessor(train_df: Any) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    for column in NUMERIC_FEATURE_COLUMNS:
        values = train_df[column].to_numpy(dtype=float)
        mean_value = float(values.mean()) if len(values) else 0.0
        std_value = float(values.std()) if len(values) else 1.0
        if std_value == 0.0:
            std_value = 1.0
        stats[column] = {"mean": mean_value, "std": std_value}

    vocabularies: dict[str, list[str]] = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        vocabularies[column] = sorted(str(value) for value in train_df[column].astype(str).unique().tolist())

    context_size = len(NUMERIC_FEATURE_COLUMNS) + sum(
        len(vocabularies[column]) for column in CATEGORICAL_FEATURE_COLUMNS
    )
    return {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_FEATURE_COLUMNS,
        "numeric_stats": stats,
        "categorical_vocabularies": vocabularies,
        "context_size": context_size,
    }


def transform_context_rows(df: Any, preprocessor: dict[str, Any], np: Any) -> dict[int, Any]:
    numeric_parts = []
    for column in preprocessor["numeric_columns"]:
        stats = preprocessor["numeric_stats"][column]
        values = df[column].to_numpy(dtype=float)
        numeric_parts.append(((values - stats["mean"]) / stats["std"]).reshape(-1, 1))

    categorical_parts = []
    for column in preprocessor["categorical_columns"]:
        categories = preprocessor["categorical_vocabularies"][column]
        category_index = {category: idx for idx, category in enumerate(categories)}
        encoded = np.zeros((len(df), len(categories)), dtype=np.float32)
        for row_idx, value in enumerate(df[column].astype(str).tolist()):
            idx = category_index.get(value)
            if idx is not None:
                encoded[row_idx, idx] = 1.0
        categorical_parts.append(encoded)

    parts = [part.astype(np.float32) for part in numeric_parts]
    parts.extend(categorical_parts)
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


def build_sequence_samples(
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
        "train": {"history": [], "static": [], "lengths": [], "labels": [], "row_ids": []},
        "test": {"history": [], "static": [], "lengths": [], "labels": [], "row_ids": []},
    }
    allowed_row_ids = set(int(row_id) for row_id in task_df["row_id"].tolist())

    sorted_df = all_pitcher_df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number", "row_id"])
    grouped = sorted_df.groupby(["game_pk", "at_bat_number"], sort=False)
    for (_, _), at_bat_df in grouped:
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

                    samples[split]["history"].append(history)
                    samples[split]["static"].append(context_by_row_id[row_id])
                    samples[split]["lengths"].append(max(1, len(history_rows)))
                    samples[split]["labels"].append(label_to_index[target_value])
                    samples[split]["row_ids"].append(row_id)

            previous_rows.append(row)

    tensors = {}
    for split, split_samples in samples.items():
        if split_samples["history"]:
            tensors[split] = {
                "history": np.stack(split_samples["history"]).astype(np.float32),
                "static": np.stack(split_samples["static"]).astype(np.float32),
                "lengths": np.array(split_samples["lengths"], dtype=np.int64),
                "labels": np.array(split_samples["labels"], dtype=np.int64),
                "row_ids": split_samples["row_ids"],
            }
        else:
            tensors[split] = {
                "history": np.zeros((0, args.sequence_length, history_feature_size), dtype=np.float32),
                "static": np.zeros((0, preprocessor["context_size"]), dtype=np.float32),
                "lengths": np.array([], dtype=np.int64),
                "labels": np.array([], dtype=np.int64),
                "row_ids": [],
            }
    return tensors["train"], tensors["test"]


def make_batches(num_rows: int, batch_size: int, seed: int, epoch: int, np: Any) -> list[Any]:
    rng = np.random.default_rng(seed + epoch)
    indices = np.arange(num_rows)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, num_rows, batch_size)]


def write_confusion_matrix(path: Path, label_names: list[str], confusion: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *label_names])
        for label_name, row in zip(label_names, confusion):
            writer.writerow([label_name, *row])


def build_lstm_classifier(nn: Any, torch: Any) -> type:
    class LSTMClassifier(nn.Module):
        def __init__(
            self,
            history_input_size: int,
            static_input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            output_size: int,
        ) -> None:
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=history_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.classifier = nn.Linear(hidden_size + static_input_size, output_size)

        def forward(self, history: Any, static: Any, lengths: Any) -> Any:
            packed = nn.utils.rnn.pack_padded_sequence(
                history,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            final_hidden = hidden[-1]
            return self.classifier(torch.cat([final_hidden, static], dim=1))

    return LSTMClassifier


def build_optimizer(model: Any, args: Any, torch: Any) -> Any:
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_tensor_dataset(samples: dict[str, Any], torch: Any) -> Any:
    return torch.utils.data.TensorDataset(
        torch.tensor(samples["history"], dtype=torch.float32),
        torch.tensor(samples["static"], dtype=torch.float32),
        torch.tensor(samples["lengths"], dtype=torch.long),
        torch.tensor(samples["labels"], dtype=torch.long),
    )


def build_data_loader(dataset: Any, batch_size: int, seed: int, torch: Any, shuffle: bool) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def predict_dataset(model: Any, data_loader: Any, torch: Any) -> tuple[list[int], list[int]]:
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for history, static, lengths, targets in data_loader:
            logits = model(history, static, lengths)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(targets.cpu().numpy().tolist())
    return labels, predictions


def dataset_loss(model: Any, data_loader: Any, loss_fn: Any, torch: Any) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for history, static, lengths, targets in data_loader:
            losses.append(float(loss_fn(model(history, static, lengths), targets).item()))
    return sum(losses) / len(losses) if losses else 0.0


def train_lstm_task(
    task_name: str,
    pitcher_name: str,
    pitcher_df: Any,
    train_game_ids: set[str],
    test_game_ids: set[str],
    target_column: str,
    args: Any,
    np: Any,
    torch: Any,
    nn: Any,
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
    if train_source_df.empty:
        return {
            "task_name": task_name,
            "status": "skipped",
            "skip_reason": "empty_train_games",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": 0, "test": 0},
        }

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

    context_by_row_id = transform_context_rows(pitcher_df, preprocessor, np)
    train_samples, test_samples = build_sequence_samples(
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
            "skip_reason": "empty_sequence_samples",
            "retained_labels": retained_labels,
            "dropped_labels": dropped_labels,
            "row_counts": {"total": int(len(task_df)), "train": int(len(train_samples["labels"])), "test": int(len(test_samples["labels"]))},
        }

    LSTMClassifier = build_lstm_classifier(nn, torch)
    model = LSTMClassifier(
        history_input_size=preprocessor["history_feature_size"],
        static_input_size=preprocessor["context_size"],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=len(label_names),
    )
    optimizer = build_optimizer(model, args, torch)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = build_tensor_dataset(train_samples, torch)
    test_dataset = build_tensor_dataset(test_samples, torch)
    train_loader = build_data_loader(train_dataset, args.batch_size, args.seed, torch, shuffle=True)
    eval_train_loader = build_data_loader(train_dataset, args.batch_size, args.seed, torch, shuffle=False)
    test_loader = build_data_loader(test_dataset, args.batch_size, args.seed, torch, shuffle=False)

    epoch_losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for history, static, lengths, targets in train_loader:
            optimizer.zero_grad()
            logits = model(history, static, lengths)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)
        if args.progress and (epoch == 0 or (epoch + 1) % args.progress_every == 0 or epoch + 1 == args.epochs):
            print(
                f"  {pitcher_name} {task_name}: "
                f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.4f}",
                flush=True,
            )

    train_loss = dataset_loss(model, eval_train_loader, loss_fn, torch)
    test_loss = dataset_loss(model, test_loader, loss_fn, torch)
    train_truth, train_predictions = predict_dataset(model, eval_train_loader, torch)
    test_truth, test_predictions = predict_dataset(model, test_loader, torch)

    train_metrics = evaluate_predictions(train_truth, train_predictions, label_names)
    test_metrics = evaluate_predictions(test_truth, test_predictions, label_names)

    artifact_dir = args.artifacts_dir / slugify(pitcher_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{task_name}_model.pt"
    preprocessor_path = artifact_dir / f"{task_name}_preprocessor.json"
    confusion_path = artifact_dir / f"confusion_matrix_{task_name}.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history_input_size": preprocessor["history_feature_size"],
            "static_input_size": preprocessor["context_size"],
            "output_dim": len(label_names),
            "label_names": label_names,
            "target_column": target_column,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length,
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
            "optimizer": args.optimizer,
            "loss": "CrossEntropyLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length,
            "grad_clip": args.grad_clip,
            "class_weights": "disabled",
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


def summarize_lstm_results(pitcher_summaries: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    summary = {
        "seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        "min_class_count": args.min_class_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "sequence_length": args.sequence_length,
        "grad_clip": args.grad_clip,
        "include_location": args.include_location,
        "max_pitchers": args.max_pitchers,
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


def run_lstm(args: Any) -> dict[str, Any]:
    np, pd, torch, nn = ensure_ml_dependencies()
    set_global_seed(args.seed, np, torch)

    processed_csv = args.output_dir / "phase1_cleaned_all_pitchers.csv"
    ensure_processed_dataset(processed_csv, args.data_dir, args.output_dir)

    df = pd.read_csv(processed_csv)
    df["row_id"] = np.arange(len(df))
    df["game_pk"] = df["game_pk"].astype(str)

    pitcher_summaries: list[dict[str, Any]] = []
    pitcher_groups = list(df.groupby("pitcher_name", sort=True))
    if args.max_pitchers is not None:
        pitcher_groups = pitcher_groups[: args.max_pitchers]

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

        pitch_type_metrics = train_lstm_task(
            task_name="pitch_type",
            pitcher_name=pitcher_name,
            pitcher_df=pitcher_df,
            train_game_ids=train_game_ids,
            test_game_ids=test_game_ids,
            target_column="target_pitch_type",
            args=args,
            np=np,
            torch=torch,
            nn=nn,
        )
        if args.include_location:
            location_metrics = train_lstm_task(
                task_name="location",
                pitcher_name=pitcher_name,
                pitcher_df=pitcher_df,
                train_game_ids=train_game_ids,
                test_game_ids=test_game_ids,
                target_column="target_location_bucket",
                args=args,
                np=np,
                torch=torch,
                nn=nn,
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
        print(f"Trained {pitcher_name}: pitch_type_acc={pitch_acc:.3f} location_acc={location_acc:.3f}")

    summary = summarize_lstm_results(pitcher_summaries, args)
    write_summary(args.artifacts_dir / "summary.json", summary)
    print(f"LSTM artifacts written to {args.artifacts_dir}")
    return summary
