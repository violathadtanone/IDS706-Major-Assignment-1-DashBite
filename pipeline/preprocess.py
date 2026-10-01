import argparse
import csv
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import load_config
from pipeline.paths import resolve_data_dir

OUTPUT_COLUMNS = [
    "order_id",
    "timestamp",
    "distance_km",
    "prep_minutes",
    "order_value",
    "was_late",
    "hour",
    "is_peak",
]
RAW_COLUMNS = ("order_id", "timestamp", "distance_km", "prep_minutes", "order_value", "was_late")
FEATURE_SUFFIX = "_features.csv"


def _parse_iso_timestamp(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _parse_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _parse_non_negative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _validate_order_row(
    raw_row: dict[str, object],
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    order_id = str(raw_row.get("order_id", "")).strip()
    invalid_fields = []
    if not order_id:
        invalid_fields.append("order_id")

    timestamp_value = str(raw_row.get("timestamp", "")).strip()
    timestamp_dt = _parse_iso_timestamp(timestamp_value)
    if timestamp_dt is None:
        invalid_fields.append("timestamp")

    distance_value = _parse_positive_float(raw_row.get("distance_km"))
    if distance_value is None:
        invalid_fields.append("distance_km")

    prep_minutes_value = _parse_positive_float(raw_row.get("prep_minutes"))
    if prep_minutes_value is None:
        invalid_fields.append("prep_minutes")

    order_value = _parse_non_negative_float(raw_row.get("order_value"))
    if order_value is None:
        invalid_fields.append("order_value")

    was_late = _parse_bool(raw_row.get("was_late"))
    if was_late is None:
        invalid_fields.append("was_late")

    if invalid_fields:
        return None, tuple(invalid_fields)

    hour = timestamp_dt.astimezone(timezone.utc).hour
    is_peak = (11 <= hour <= 13) or (17 <= hour <= 20)
    normalized_prep_minutes: int | float = (
        int(prep_minutes_value) if prep_minutes_value.is_integer() else prep_minutes_value
    )

    return (
        {
            "order_id": order_id,
            "timestamp": timestamp_value,
            "distance_km": float(distance_value),
            "prep_minutes": normalized_prep_minutes,
            "order_value": float(order_value),
            "was_late": was_late,
            "hour": hour,
            "is_peak": is_peak,
        },
        (),
    )


def normalize_order_row(raw_row: dict[str, object]) -> dict[str, object] | None:
    """Validate and normalize one raw order row into feature-ready output."""
    normalized, _ = _validate_order_row(raw_row)
    return normalized


def _feature_output_path(raw_path: Path, features_dir: Path) -> Path:
    feature_name = raw_path.name.replace(".csv", "_features.csv")
    return features_dir / feature_name


def _quality_output_path(feature_path: Path, quality_dir: Path) -> Path:
    if not feature_path.name.endswith(FEATURE_SUFFIX):
        raise ValueError(f"Feature batch name must end with {FEATURE_SUFFIX!r}.")
    quality_name = feature_path.name[: -len(FEATURE_SUFFIX)] + "_quality.json"
    return quality_dir / quality_name


def _read_raw_batch(raw_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    normalized_rows = []
    input_rows = 0
    rejected_rows = 0
    field_failures = {field: 0 for field in RAW_COLUMNS}
    with raw_path.open("r", newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        for raw_row in reader:
            input_rows += 1
            normalized, invalid_fields = _validate_order_row(raw_row)
            if invalid_fields:
                rejected_rows += 1
                for field in invalid_fields:
                    field_failures[field] += 1
            elif normalized is not None:
                normalized_rows.append(normalized)

    accepted_rows = len(normalized_rows)
    quality = {
        "source_file": raw_path.name,
        "input_rows": input_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "drop_rate": rejected_rows / input_rows if input_rows else None,
        "field_failures": field_failures,
    }
    return normalized_rows, quality


def _write_quality_sidecar(
    feature_path: Path, quality_dir: Path, quality: dict[str, object]
) -> Path:
    quality_dir.mkdir(parents=True, exist_ok=True)
    quality_path = _quality_output_path(feature_path, quality_dir)
    payload = {**quality, "feature_file": feature_path.name}
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=quality_dir,
            prefix=".quality_",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, quality_path)
    except BaseException:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return quality_path


def _backfill_quality_sidecars(raw_dir: Path, features_dir: Path, quality_dir: Path) -> int:
    backfilled = 0
    for feature_path in sorted(features_dir.glob(f"*{FEATURE_SUFFIX}")):
        quality_path = _quality_output_path(feature_path, quality_dir)
        if quality_path.exists():
            continue
        raw_name = feature_path.name[: -len(FEATURE_SUFFIX)] + ".csv"
        raw_path = raw_dir / raw_name
        if not raw_path.is_file():
            print(
                f"Quality history unavailable for {feature_path.name}: "
                f"raw source {raw_name} is missing; cannot backfill."
            )
            continue
        try:
            _, quality = _read_raw_batch(raw_path)
            _write_quality_sidecar(feature_path, quality_dir, quality)
        except Exception as exc:
            print(f"Could not backfill quality for {feature_path.name}: {exc}")
            continue
        print(f"Backfilled quality sidecar for {feature_path.name}.")
        backfilled += 1
    return backfilled


def _candidate_is_stable(path: Path, *, delay_seconds: float = 0.1) -> bool:
    try:
        first = path.stat()
    except FileNotFoundError:
        return False
    time.sleep(delay_seconds)
    try:
        second = path.stat()
    except FileNotFoundError:
        return False
    return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns


def _iter_raw_batches(raw_dir: Path, features_dir: Path):
    if not raw_dir.exists():
        return []
    candidates = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if not name.startswith("orders_") or not name.endswith(".csv"):
            continue
        if name.startswith("."):
            continue
        if name.endswith(".tmp"):
            continue
        feature_path = _feature_output_path(path, features_dir)
        if feature_path.exists():
            continue
        if _candidate_is_stable(path):
            candidates.append(path)
    return candidates


def process_raw_csv(
    raw_path: Path, features_dir: Path, quality_dir: Path | None = None
) -> Path:
    """Read a finalized raw CSV batch and write a cleaned feature CSV batch."""
    features_dir.mkdir(parents=True, exist_ok=True)
    quality_dir = Path(quality_dir) if quality_dir is not None else features_dir.parent / "quality"
    feature_path = _feature_output_path(raw_path, features_dir)
    temp_path: str | None = None

    try:
        rows, quality = _read_raw_batch(raw_path)

        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=features_dir,
            prefix=".orders_",
            suffix=".features.csv.tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        os.replace(temp_path, feature_path)
        _write_quality_sidecar(feature_path, quality_dir, quality)
    except BaseException:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

    print(
        f"Processed {raw_path.name}: accepted={quality['accepted_rows']} "
        f"rejected={quality['rejected_rows']} "
        f"feature={feature_path.name}"
    )
    return feature_path


def preprocess_once(
    *,
    raw_dir: str | Path | None = None,
    features_dir: str | Path | None = None,
    quality_dir: str | Path | None = None,
) -> int:
    """Process any finalized raw CSV batches that have not yet been turned into feature files."""
    raw_dir = Path(raw_dir) if raw_dir is not None else resolve_data_dir("raw")
    custom_features_dir = features_dir is not None
    features_dir = Path(features_dir) if custom_features_dir else resolve_data_dir("features")
    quality_dir = (
        Path(quality_dir)
        if quality_dir is not None
        else features_dir.parent / "quality"
        if custom_features_dir
        else resolve_data_dir("quality")
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    _backfill_quality_sidecars(raw_dir, features_dir, quality_dir)

    processed = 0
    for raw_path in sorted(_iter_raw_batches(raw_dir, features_dir)):
        feature_output = _feature_output_path(raw_path, features_dir)
        if feature_output.exists():
            continue
        try:
            process_raw_csv(raw_path, features_dir, quality_dir)
        except Exception as exc:
            print(f"Skipping batch {raw_path.name} due to error: {exc}")
            continue
        processed += 1
    return processed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw order batches into feature CSVs.")
    parser.add_argument("--once", action="store_true", help="process one pass and exit")
    parser.add_argument("--raw-dir", type=Path, help="override the raw data directory")
    parser.add_argument("--features-dir", type=Path, help="override the feature directory")
    parser.add_argument("--poll-interval", type=float, help="seconds between polls")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config()
    raw_dir = args.raw_dir if args.raw_dir is not None else resolve_data_dir("raw")
    features_dir = args.features_dir if args.features_dir is not None else resolve_data_dir("features")
    poll_interval = args.poll_interval if args.poll_interval is not None else config["poll_interval_seconds"]

    try:
        while True:
            processed = preprocess_once(raw_dir=raw_dir, features_dir=features_dir)
            if processed:
                print(f"Completed {processed} processed batch(es).")
            if args.once:
                return 0
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Preprocess stopped cleanly.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
