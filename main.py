#!/usr/bin/env python3
"""Phase 1 preprocessing pipeline for pitch prediction.

This script turns the raw Statcast exports in ``data/`` into cleaned, model-ready
CSV files. The Phase 1 goals from the README are implemented here by:

1. Restricting the dataset to 2022-2025 regular-season pitches.
2. Keeping only pre-pitch game-state features so we do not leak outcome data.
3. Removing redundant/collinear representations by writing a curated feature set.
4. Producing targets for both pitch type and pitch location.
5. Writing per-pitcher cleaned datasets plus a combined dataset and summary file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed")
VALID_GAME_YEARS = {2022, 2023, 2024, 2025}
VALID_GAME_TYPE = "R"

METADATA_COLUMNS = [
    "pitcher_name",
    "source_file",
    "game_date",
    "game_pk",
]

FEATURE_COLUMNS = [
    "game_year",
    "home_team",
    "away_team",
    "stand",
    "p_throws",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "balls",
    "strikes",
    "at_bat_number",
    "pitch_number",
    "runners_on_1b",
    "runners_on_2b",
    "runners_on_3b",
    "bat_score_diff",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "age_pit",
    "age_bat",
    "if_fielding_alignment",
    "of_fielding_alignment",
]

NUMERIC_FEATURE_COLUMNS = [
    "game_year",
    "inning",
    "outs_when_up",
    "balls",
    "strikes",
    "at_bat_number",
    "pitch_number",
    "runners_on_1b",
    "runners_on_2b",
    "runners_on_3b",
    "bat_score_diff",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "age_pit",
    "age_bat",
]

CATEGORICAL_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURE_COLUMNS
]

TARGET_COLUMNS = [
    "target_pitch_type",
    "target_pitch_name",
    "target_plate_x",
    "target_plate_z",
    "target_location_bucket",
]

OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def normalize_header(header: str) -> str:
    """Normalize Statcast headers by removing BOMs and wrapping quotes."""
    return header.replace("\ufeff", "").strip().strip('"')


def normalize_pitcher_name(filename: str, raw_player_name: str | None) -> str:
    """Prefer the Statcast player name, but fall back to a cleaned filename."""
    if raw_player_name:
        return raw_player_name.strip()

    stem = Path(filename).stem
    pieces = []
    current = ""
    for char in stem:
        if char.isupper() and current:
            pieces.append(current)
            current = char
        else:
            current += char
    if current:
        pieces.append(current)
    return " ".join(piece.capitalize() for piece in pieces)


def safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return int(float(value))


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def runner_on_base(value: str | None) -> int:
    return 0 if value is None or value.strip() == "" else 1


def compute_bat_score_diff(row: dict[str, str]) -> int | None:
    raw_diff = safe_int(row.get("bat_score_diff"))
    if raw_diff is not None:
        return raw_diff

    bat_score = safe_int(row.get("bat_score"))
    fld_score = safe_int(row.get("fld_score"))
    if bat_score is None or fld_score is None:
        return None
    return bat_score - fld_score


def make_location_bucket(
    plate_x: float | None,
    plate_z: float | None,
    sz_top: float | None,
    sz_bot: float | None,
) -> str | None:
    """Create a coarse 3x3 location label relative to the batter's strike zone."""
    if plate_x is None or plate_z is None:
        return None

    strike_zone_left = -0.83
    strike_zone_right = 0.83

    if sz_top is None or sz_bot is None or sz_top <= sz_bot:
        relative_height = None
    else:
        relative_height = (plate_z - sz_bot) / (sz_top - sz_bot)

    if plate_x < strike_zone_left:
        horizontal = "outside"
    elif plate_x > strike_zone_right:
        horizontal = "inside"
    elif plate_x < -0.28:
        horizontal = "left"
    elif plate_x > 0.28:
        horizontal = "right"
    else:
        horizontal = "center"

    if relative_height is None:
        if plate_z < 1.5:
            vertical = "low"
        elif plate_z > 3.5:
            vertical = "high"
        else:
            vertical = "middle"
    elif relative_height < 0.0:
        vertical = "below"
    elif relative_height > 1.0:
        vertical = "above"
    elif relative_height < 0.33:
        vertical = "low"
    elif relative_height > 0.66:
        vertical = "high"
    else:
        vertical = "middle"

    return f"{vertical}_{horizontal}"


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def format_numeric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def parse_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    game_date = row["game_date"] or ""
    return (
        game_date,
        safe_int(str(row["game_pk"])) if row["game_pk"] not in ("", None) else -1,
        safe_int(str(row["at_bat_number"])) if row["at_bat_number"] not in ("", None) else -1,
        safe_int(str(row["pitch_number"])) if row["pitch_number"] not in ("", None) else -1,
    )


def load_rows(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, filter, and engineer rows from all pitcher CSVs."""
    rows: list[dict[str, Any]] = []
    skipped_rows = Counter()
    pitch_types_by_pitcher: dict[str, set[str]] = defaultdict(set)
    season_counts = Counter()
    raw_columns = set()

    for path in sorted(data_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue

            reader.fieldnames = [normalize_header(header) for header in reader.fieldnames]

            for raw_row in reader:
                row = {normalize_header(key): (value or "").strip() for key, value in raw_row.items()}
                raw_columns.update(row.keys())

                game_year = safe_int(row.get("game_year"))
                if game_year not in VALID_GAME_YEARS:
                    skipped_rows["outside_year_window"] += 1
                    continue

                if row.get("game_type") != VALID_GAME_TYPE:
                    skipped_rows["non_regular_season"] += 1
                    continue

                pitch_type = row.get("pitch_type", "")
                pitch_name = row.get("pitch_name", "")
                plate_x = safe_float(row.get("plate_x"))
                plate_z = safe_float(row.get("plate_z"))
                if pitch_type == "":
                    skipped_rows["missing_pitch_type"] += 1
                    continue
                if pitch_type == "UN" or pitch_name == "Unknown":
                    skipped_rows["unknown_pitch_type"] += 1
                    continue
                if plate_x is None or plate_z is None:
                    skipped_rows["missing_pitch_location"] += 1
                    continue

                pitcher_name = normalize_pitcher_name(path.name, row.get("player_name"))
                season_counts[game_year] += 1
                pitch_types_by_pitcher[pitcher_name].add(pitch_type)

                cleaned = {
                    "pitcher_name": pitcher_name,
                    "source_file": path.name,
                    "game_date": row.get("game_date", ""),
                    "game_pk": safe_int(row.get("game_pk")),
                    "game_year": game_year,
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "stand": row.get("stand", ""),
                    "p_throws": row.get("p_throws", ""),
                    "inning": safe_int(row.get("inning")),
                    "inning_topbot": row.get("inning_topbot", ""),
                    "outs_when_up": safe_int(row.get("outs_when_up")),
                    "balls": safe_int(row.get("balls")),
                    "strikes": safe_int(row.get("strikes")),
                    "at_bat_number": safe_int(row.get("at_bat_number")),
                    "pitch_number": safe_int(row.get("pitch_number")),
                    "runners_on_1b": runner_on_base(row.get("on_1b")),
                    "runners_on_2b": runner_on_base(row.get("on_2b")),
                    "runners_on_3b": runner_on_base(row.get("on_3b")),
                    "bat_score_diff": compute_bat_score_diff(row),
                    "n_thruorder_pitcher": safe_int(row.get("n_thruorder_pitcher")),
                    "n_priorpa_thisgame_player_at_bat": safe_int(row.get("n_priorpa_thisgame_player_at_bat")),
                    "pitcher_days_since_prev_game": safe_int(row.get("pitcher_days_since_prev_game")),
                    "batter_days_since_prev_game": safe_int(row.get("batter_days_since_prev_game")),
                    "age_pit": safe_int(row.get("age_pit")),
                    "age_bat": safe_int(row.get("age_bat")),
                    "if_fielding_alignment": row.get("if_fielding_alignment", ""),
                    "of_fielding_alignment": row.get("of_fielding_alignment", ""),
                    "target_pitch_type": pitch_type,
                    "target_pitch_name": pitch_name,
                    "target_plate_x": plate_x,
                    "target_plate_z": plate_z,
                    "target_location_bucket": make_location_bucket(
                        plate_x=plate_x,
                        plate_z=plate_z,
                        sz_top=safe_float(row.get("sz_top")),
                        sz_bot=safe_float(row.get("sz_bot")),
                    ),
                }

                rows.append(cleaned)

    rows.sort(key=parse_sort_key)

    summary = {
        "raw_columns_seen": sorted(raw_columns),
        "season_counts": dict(sorted(season_counts.items())),
        "skipped_rows": dict(skipped_rows),
        "pitch_types_by_pitcher": {
            pitcher: sorted(pitch_types)
            for pitcher, pitch_types in sorted(pitch_types_by_pitcher.items())
        },
    }
    return rows, summary


def impute_missing_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fill remaining holes so the cleaned output is model-ready."""
    numeric_medians: dict[str, float] = {}
    categorical_modes: dict[str, str] = {}

    for column in NUMERIC_FEATURE_COLUMNS:
        values = [row[column] for row in rows if row[column] is not None]
        numeric_medians[column] = median([float(value) for value in values]) if values else 0.0

    for column in CATEGORICAL_FEATURE_COLUMNS:
        values = [row[column] for row in rows if row[column] not in ("", None)]
        categorical_modes[column] = Counter(values).most_common(1)[0][0] if values else "UNKNOWN"

    missing_before = Counter()
    for row in rows:
        for column in NUMERIC_FEATURE_COLUMNS:
            if row[column] is None:
                missing_before[column] += 1
                median_value = numeric_medians[column]
                row[column] = int(median_value) if median_value.is_integer() else median_value
        for column in CATEGORICAL_FEATURE_COLUMNS:
            if row[column] in ("", None):
                missing_before[column] += 1
                row[column] = categorical_modes[column]
        if row["target_pitch_name"] in ("", None):
            row["target_pitch_name"] = row["target_pitch_type"]
        if row["target_location_bucket"] in ("", None):
            row["target_location_bucket"] = "unknown"

    return {
        "numeric_feature_medians": numeric_medians,
        "categorical_feature_modes": categorical_modes,
        "imputed_values_count": dict(missing_before),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_numeric(row.get(column)) for column in OUTPUT_COLUMNS})


def build_phase1_summary(rows: list[dict[str, Any]], load_summary: dict[str, Any], impute_summary: dict[str, Any]) -> dict[str, Any]:
    pitch_types_by_pitcher = load_summary["pitch_types_by_pitcher"]
    unique_pitch_type_counts = {
        pitcher: len(pitch_types) for pitcher, pitch_types in pitch_types_by_pitcher.items()
    }

    rows_by_pitcher = Counter(row["pitcher_name"] for row in rows)
    target_pitch_type_counts = Counter(row["target_pitch_type"] for row in rows)
    location_bucket_counts = Counter(row["target_location_bucket"] for row in rows)

    retained_columns = set(OUTPUT_COLUMNS)
    dropped_columns = [
        column
        for column in load_summary["raw_columns_seen"]
        if column not in retained_columns and column not in {"on_1b", "on_2b", "on_3b", "pitch_type", "pitch_name", "plate_x", "plate_z"}
    ]

    average_pitch_types = (
        round(statistics.mean(unique_pitch_type_counts.values()), 2)
        if unique_pitch_type_counts
        else 0.0
    )

    return {
        "phase": "Phase 1",
        "filters": {
            "game_years": sorted(VALID_GAME_YEARS),
            "game_type": VALID_GAME_TYPE,
        },
        "row_counts": {
            "clean_rows": len(rows),
            "pitchers": len(rows_by_pitcher),
            "rows_by_pitcher": dict(sorted(rows_by_pitcher.items())),
            "season_counts": load_summary["season_counts"],
            "skipped_rows": load_summary["skipped_rows"],
        },
        "targets": {
            "pitch_type_classes": dict(target_pitch_type_counts.most_common()),
            "location_bucket_classes": dict(location_bucket_counts.most_common()),
        },
        "pitch_mix_summary": {
            "average_unique_pitch_types_per_pitcher": average_pitch_types,
            "unique_pitch_types_per_pitcher": dict(sorted(unique_pitch_type_counts.items())),
            "pitch_types_by_pitcher": pitch_types_by_pitcher,
        },
        "retained_feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "dropped_raw_columns": dropped_columns,
        "imputation": impute_summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_report(summary: dict[str, Any], output_dir: Path) -> None:
    print("Phase 1 preprocessing complete.")
    print(f"Cleaned rows: {summary['row_counts']['clean_rows']}")
    print(f"Pitchers: {summary['row_counts']['pitchers']}")
    print(f"Average unique pitch types per pitcher: {summary['pitch_mix_summary']['average_unique_pitch_types_per_pitcher']}")
    print(f"Combined dataset: {output_dir / 'phase1_cleaned_all_pitchers.csv'}")
    print(f"Summary file: {output_dir / 'phase1_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned Phase 1 pitch-prediction datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing raw per-pitcher Statcast CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where cleaned Phase 1 outputs will be written.",
    )
    args = parser.parse_args()

    rows, load_summary = load_rows(args.data_dir)
    if not rows:
        raise SystemExit("No valid rows were found in the input data directory.")

    impute_summary = impute_missing_values(rows)

    per_pitcher_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_pitcher_rows[row["pitcher_name"]].append(row)

    write_csv(args.output_dir / "phase1_cleaned_all_pitchers.csv", rows)

    per_pitcher_dir = args.output_dir / "pitchers"
    for pitcher_name, pitcher_rows in sorted(per_pitcher_rows.items()):
        filename = pitcher_name.lower().replace(",", "").replace(" ", "_") + "_phase1.csv"
        write_csv(per_pitcher_dir / filename, pitcher_rows)

    summary = build_phase1_summary(rows, load_summary, impute_summary)
    write_summary(args.output_dir / "phase1_summary.json", summary)
    print_report(summary, args.output_dir)


if __name__ == "__main__":
    main()
