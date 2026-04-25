"""Per-pitcher 1D CNN sequence baseline for pitch prediction."""

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
from logistic_regression import ensure_ml_dependencies, evaluate_predictions, set_global_seed
from lstm_model import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR,
    DEFAULT_OPTIMIZER,
    DEFAULT_SEED,
    DEFAULT_SEQUENCE_LENGTH,
    MIN_CLASS_COUNT,
    TRAIN_FRACTION,
    build_context_preprocessor,
    build_data_loader,
    build_label_mapping,
    build_sequence_samples,
    build_tensor_dataset,
    chronological_game_split,
    dataset_loss,
    ensure_processed_dataset,
    predict_dataset,
    transform_context_rows,
    write_confusion_matrix,
)


ARTIFACTS_DIR = Path("artifacts/cnn1d")
DEFAULT_CHANNELS = 256
DEFAULT_NUM_BLOCKS = 3
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT = 0.3
DEFAULT_STATIC_HIDDEN_SIZE = 128


def build_cnn1d_classifier(nn: Any, torch: Any) -> type:
    class CNN1DClassifier(nn.Module):
        def __init__(
            self,
            history_input_size: int,
            static_input_size: int,
            channels: int,
            num_blocks: int,
            kernel_size: int,
            dropout: float,
            static_hidden_size: int,
            output_size: int,
        ) -> None:
            super().__init__()
            padding = kernel_size // 2
            blocks = []
            in_channels = history_input_size
            for _ in range(num_blocks):
                blocks.extend(
                    [
                        nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=padding),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_channels = channels
            self.history_encoder = nn.Sequential(*blocks)
            self.static_encoder = nn.Sequential(
                nn.Linear(static_input_size, static_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(channels * 2 + static_hidden_size, channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, output_size),
            )

        def forward(self, history: Any, static: Any, lengths: Any) -> Any:
            del lengths
            encoded_history = self.history_encoder(history.transpose(1, 2))
            pooled_max = encoded_history.amax(dim=2)
            pooled_avg = encoded_history.mean(dim=2)
            static_features = self.static_encoder(static)
            features = torch.cat([pooled_max, pooled_avg, static_features], dim=1)
            return self.classifier(features)

    return CNN1DClassifier


def build_optimizer(model: Any, args: Any, torch: Any) -> Any:
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def train_cnn1d_task(
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
            "row_counts": {
                "total": int(len(task_df)),
                "train": int(len(train_samples["labels"])),
                "test": int(len(test_samples["labels"])),
            },
        }

    CNN1DClassifier = build_cnn1d_classifier(nn, torch)
    model = CNN1DClassifier(
        history_input_size=preprocessor["history_feature_size"],
        static_input_size=preprocessor["context_size"],
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        static_hidden_size=args.static_hidden_size,
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
            "channels": args.channels,
            "num_blocks": args.num_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "static_hidden_size": args.static_hidden_size,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
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
            "model": "cnn1d",
            "optimizer": args.optimizer,
            "loss": "CrossEntropyLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "channels": args.channels,
            "num_blocks": args.num_blocks,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "static_hidden_size": args.static_hidden_size,
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


def summarize_cnn1d_results(pitcher_summaries: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    summary = {
        "model": "cnn1d",
        "seed": args.seed,
        "train_fraction": TRAIN_FRACTION,
        "min_class_count": args.min_class_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "channels": args.channels,
        "num_blocks": args.num_blocks,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "static_hidden_size": args.static_hidden_size,
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


def run_cnn1d(args: Any) -> dict[str, Any]:
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

        pitch_type_metrics = train_cnn1d_task(
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
            location_metrics = train_cnn1d_task(
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
        print(f"Trained {pitcher_name}: pitch_type_acc={pitch_acc:.3f} location_acc={location_acc:.3f}", flush=True)

    summary = summarize_cnn1d_results(pitcher_summaries, args)
    write_summary(args.artifacts_dir / "summary.json", summary)
    print(f"CNN1D artifacts written to {args.artifacts_dir}")
    return summary
