"""Generate candidate figures for the course project report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm_model import TRAIN_FRACTION, chronological_game_split


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
PROCESSED_CSV = ROOT / "processed" / "phase1_cleaned_all_pitchers.csv"
SUMMARY_PATHS = {
    "Logistic Regression": ROOT / "artifacts" / "logreg" / "summary.json",
    "KNN": ROOT / "artifacts" / "knn" / "summary.json",
    "HMM-style": ROOT / "artifacts" / "hmm" / "summary.json",
    "LDA": ROOT / "artifacts" / "lda" / "summary.json",
    "GDA": ROOT / "artifacts" / "gda" / "summary.json",
    "LSTM": ROOT / "artifacts" / "lstm_unweighted_big" / "summary.json",
    "1D CNN": ROOT / "artifacts" / "cnn1d" / "summary.json",
    "XGBoost": ROOT / "artifacts" / "xgboost_unweighted_big" / "summary.json",
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def build_model_summary_frame() -> pd.DataFrame:
    rows = []
    for model_name, path in SUMMARY_PATHS.items():
        summary = load_summary(path)
        metrics = summary["aggregates"]["pitch_type"]
        rows.append(
            {
                "model": model_name,
                "trained_pitchers": metrics["trained_pitchers"],
                "accuracy": metrics["mean_accuracy"],
                "macro_f1": metrics["mean_macro_f1"],
                "weighted_f1": metrics["mean_weighted_f1"],
            }
        )
    return pd.DataFrame(rows)


def build_mode_baseline_frame() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_CSV)
    df["game_pk"] = df["game_pk"].astype(str)

    rows = []
    for pitcher_name, pitcher_df in df.groupby("pitcher_name", sort=True):
        train_games, test_games = chronological_game_split(pitcher_df, TRAIN_FRACTION)
        train_df = pitcher_df[pitcher_df["game_pk"].isin(train_games)].copy()
        test_df = pitcher_df[pitcher_df["game_pk"].isin(test_games)].copy()
        if train_df.empty or test_df.empty:
            continue
        mode_pitch = train_df["target_pitch_type"].astype(str).value_counts().idxmax()
        mode_accuracy = float((test_df["target_pitch_type"].astype(str) == mode_pitch).mean())
        rows.append(
            {
                "pitcher_name": pitcher_name,
                "mode_pitch_type": mode_pitch,
                "mode_accuracy": mode_accuracy,
            }
        )
    return pd.DataFrame(rows)


def build_per_pitcher_accuracy_frame() -> pd.DataFrame:
    rows = []
    for model_name, path in SUMMARY_PATHS.items():
        summary = load_summary(path)
        for pitcher_summary in summary["pitchers"]:
            task = pitcher_summary["tasks"]["pitch_type"]
            rows.append(
                {
                    "model": model_name,
                    "pitcher_name": pitcher_summary["pitcher_name"],
                    "status": task["status"],
                    "accuracy": task["test_metrics"]["accuracy"] if task["status"] == "trained" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_figure(path_stem: str) -> tuple[Path, Path]:
    png_path = FIGURES_DIR / f"{path_stem}.png"
    pdf_path = FIGURES_DIR / f"{path_stem}.pdf"
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def plot_model_leaderboard(model_df: pd.DataFrame, mode_mean_accuracy: float) -> None:
    plot_df = (
        pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "model": "Naive Mode Baseline",
                            "trained_pitchers": 30,
                            "accuracy": mode_mean_accuracy,
                            "macro_f1": np.nan,
                            "weighted_f1": np.nan,
                        }
                    ]
                ),
                model_df,
            ],
            ignore_index=True,
        )
        .sort_values("accuracy", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#9AA5B1" if model == "Naive Mode Baseline" else "#1f77b4" for model in plot_df["model"]]
    bars = ax.barh(plot_df["model"], plot_df["accuracy"], color=colors, edgecolor="#23303B", linewidth=0.6)
    ax.set_xlabel("Mean per-pitcher test accuracy")
    ax.set_ylabel("")
    ax.set_title("Model Bake-Off on 2025 Statcast Pitch-Type Prediction")
    ax.set_xlim(0.0, max(plot_df["accuracy"]) + 0.06)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, trained_pitchers in zip(bars, plot_df["trained_pitchers"]):
        width = bar.get_width()
        label = f"{width:.3f}"
        if trained_pitchers < 30:
            label += f" ({trained_pitchers}/30)"
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)

    save_figure("model_leaderboard")


def plot_xgboost_vs_mode(per_pitcher_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    xgb_df = per_pitcher_df[per_pitcher_df["model"] == "XGBoost"][["pitcher_name", "accuracy"]].rename(
        columns={"accuracy": "xgboost_accuracy"}
    )
    merged = baseline_df.merge(xgb_df, on="pitcher_name", how="inner")
    merged["improvement"] = merged["xgboost_accuracy"] - merged["mode_accuracy"]
    merged = merged.sort_values("improvement", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    y_positions = np.arange(len(merged))
    ax.hlines(y_positions, merged["mode_accuracy"], merged["xgboost_accuracy"], color="#B8C4CE", linewidth=2.0)
    ax.scatter(merged["mode_accuracy"], y_positions, color="#7A8793", label="Naive mode baseline", s=34, zorder=3)
    ax.scatter(merged["xgboost_accuracy"], y_positions, color="#0E7490", label="XGBoost", s=34, zorder=3)
    ax.axvline(merged["mode_accuracy"].mean(), color="#7A8793", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(merged["xgboost_accuracy"].mean(), color="#0E7490", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(merged["pitcher_name"], fontsize=8)
    ax.set_xlabel("Test accuracy")
    ax.set_ylabel("")
    ax.set_title("XGBoost Improves on the Naive Baseline for Most Pitchers")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    ax.set_axisbelow(True)

    save_figure("xgboost_vs_mode")


def plot_pitcher_heatmap(per_pitcher_df: pd.DataFrame) -> None:
    selected_models = ["HMM-style", "LDA", "LSTM", "1D CNN", "XGBoost"]
    heat_df = (
        per_pitcher_df[per_pitcher_df["model"].isin(selected_models)]
        .pivot(index="pitcher_name", columns="model", values="accuracy")
        .loc[:, selected_models]
    )
    heat_df["sort_key"] = heat_df["XGBoost"]
    heat_df = heat_df.sort_values("sort_key", ascending=False).drop(columns="sort_key")

    fig, ax = plt.subplots(figsize=(7.8, 8.4))
    matrix = heat_df.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.22, vmax=0.58)
    ax.set_xticks(np.arange(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=8)
    ax.set_title("Per-Pitcher Accuracy Varies Substantially Across Models")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="#102A43")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy", rotation=270, labelpad=12)

    save_figure("pitcher_model_heatmap")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_df = build_model_summary_frame()
    baseline_df = build_mode_baseline_frame()
    per_pitcher_df = build_per_pitcher_accuracy_frame()

    plot_model_leaderboard(model_df, baseline_df["mode_accuracy"].mean())
    plot_xgboost_vs_mode(per_pitcher_df, baseline_df)
    plot_pitcher_heatmap(per_pitcher_df)

    summary_csv = FIGURES_DIR / "report_figure_data.csv"
    model_df.sort_values("accuracy", ascending=False).to_csv(summary_csv, index=False)
    print("Wrote figures to", FIGURES_DIR)
    print("Wrote summary data to", summary_csv)


if __name__ == "__main__":
    main()
