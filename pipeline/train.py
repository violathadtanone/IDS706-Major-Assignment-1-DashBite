import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline.config import load_config
from pipeline.paths import resolve_data_dir

FEATURE_COLUMNS = ("distance_km", "prep_minutes")
TARGET_COLUMN = "was_late"
RANDOM_STATE = 42
TEST_FRACTION = 0.2
_CHECKPOINT_RE = re.compile(r"late_order_v(\d{4,})\.joblib$")
_SIDECAR_RE = re.compile(r"late_order_v(\d{4,})\.metrics\.json$")


class InsufficientTrainingData(ValueError):
    """Raised when a stratified split cannot represent both labels in both sets."""


def _parse_label(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _read_feature_batch(path: Path) -> tuple[list[dict[str, float | bool]], int]:
    records: list[dict[str, float | bool]] = []
    rejected_rows = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
        ):
            raise ValueError("missing one or more required training columns")

        for row in reader:
            try:
                distance_km = float(row["distance_km"] or "")
                prep_minutes = float(row["prep_minutes"] or "")
            except (TypeError, ValueError):
                rejected_rows += 1
                continue
            was_late = _parse_label(row.get(TARGET_COLUMN))
            if (
                not math.isfinite(distance_km)
                or distance_km <= 0
                or not math.isfinite(prep_minutes)
                or prep_minutes <= 0
                or was_late is None
            ):
                rejected_rows += 1
                continue
            records.append(
                {
                    "distance_km": distance_km,
                    "prep_minutes": prep_minutes,
                    "was_late": was_late,
                }
            )
    return records, rejected_rows


def _load_feature_batches(features_dir: Path) -> tuple[list[dict[str, float | bool]], dict[str, int]]:
    records: list[dict[str, float | bool]] = []
    batch_row_counts: dict[str, int] = {}
    if not features_dir.is_dir():
        return records, batch_row_counts

    for path in sorted(features_dir.glob("*_features.csv")):
        try:
            batch_records, rejected_rows = _read_feature_batch(path)
        except (OSError, csv.Error, ValueError) as exc:
            print(f"Skipping feature batch {path.name} due to error: {exc}")
            continue
        if rejected_rows:
            print(f"Rejected {rejected_rows} invalid training row(s) from {path.name}.")
        batch_row_counts[path.name] = len(batch_records)
        records.extend(batch_records)
    return records, batch_row_counts


def _stratified_split(labels: list[bool]) -> tuple[list[int], list[int]]:
    class_counts = {label: labels.count(label) for label in (False, True)}
    test_count = math.ceil(len(labels) * TEST_FRACTION)
    train_count = len(labels) - test_count
    if min(class_counts.values()) < 2 or test_count < 2 or train_count < 2:
        raise InsufficientTrainingData(
            "need both label classes represented in the training and evaluation sets"
        )
    train_indices, test_indices = train_test_split(
        list(range(len(labels))),
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return list(train_indices), list(test_indices)


def _evaluate_and_fit(
    records: list[dict[str, float | bool]],
) -> tuple[LogisticRegression, dict[str, float], dict[str, int]]:
    labels = [bool(record[TARGET_COLUMN]) for record in records]
    train_indices, test_indices = _stratified_split(labels)
    inputs = [
        [float(record["distance_km"]), float(record["prep_minutes"])]
        for record in records
    ]
    train_inputs = [inputs[index] for index in train_indices]
    test_inputs = [inputs[index] for index in test_indices]
    train_labels = [labels[index] for index in train_indices]
    test_labels = [labels[index] for index in test_indices]

    evaluation_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    evaluation_model.fit(train_inputs, train_labels)
    predictions = evaluation_model.predict(test_inputs)
    positive_index = list(evaluation_model.classes_).index(True)
    positive_probabilities = evaluation_model.predict_proba(test_inputs)[:, positive_index]
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision_score(test_labels, predictions, pos_label=True, zero_division=0)),
        "recall": float(recall_score(test_labels, predictions, pos_label=True, zero_division=0)),
        "f1": float(f1_score(test_labels, predictions, pos_label=True, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_labels, positive_probabilities)),
    }

    published_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    published_model.fit(inputs, labels)
    setattr(published_model, "dashbite_feature_names_", FEATURE_COLUMNS)
    split_sizes = {"train_size": len(train_indices), "test_size": len(test_indices)}
    return published_model, metrics, split_sizes


def _artifact_version(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _read_successful_sidecar(models_dir: Path, version: int) -> dict[str, object] | None:
    checkpoint_path = models_dir / f"late_order_v{version:04d}.joblib"
    sidecar_path = models_dir / f"late_order_v{version:04d}.metrics.json"
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        with sidecar_path.open(encoding="utf-8") as handle:
            sidecar = json.load(handle)
        if (
            sidecar.get("version") != version
            or sidecar.get("model_version") != f"v{version:04d}"
            or sidecar.get("feature_names") != list(FEATURE_COLUMNS)
            or not isinstance(sidecar.get("feature_batch_ids"), list)
            or not isinstance(sidecar.get("pending_event_count"), int)
        ):
            return None
        return sidecar
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _latest_successful_sidecar(models_dir: Path) -> dict[str, object] | None:
    versions = []
    for path in models_dir.glob("late_order_v*.metrics.json"):
        version = _artifact_version(path, _SIDECAR_RE)
        if version is not None:
            versions.append(version)

    for version in sorted(versions, reverse=True):
        sidecar = _read_successful_sidecar(models_dir, version)
        if sidecar is not None:
            return sidecar
    return None


def _remove_orphan_checkpoints(models_dir: Path) -> None:
    for checkpoint_path in models_dir.glob("late_order_v*.joblib"):
        version = _artifact_version(checkpoint_path, _CHECKPOINT_RE)
        if version is not None and _read_successful_sidecar(models_dir, version) is None:
            checkpoint_path.unlink()


def _next_version(models_dir: Path) -> int:
    versions = []
    for path in models_dir.iterdir():
        version = _artifact_version(path, _CHECKPOINT_RE) or _artifact_version(path, _SIDECAR_RE)
        if version is not None:
            versions.append(version)
    return max(versions, default=0) + 1


def _publish_artifacts(
    model: LogisticRegression,
    sidecar: dict[str, object],
    models_dir: Path,
) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(models_dir)
    model_version = f"v{version:04d}"
    checkpoint_path = models_dir / f"late_order_{model_version}.joblib"
    sidecar_path = models_dir / f"late_order_{model_version}.metrics.json"
    published_sidecar = {**sidecar, "version": version, "model_version": model_version}
    checkpoint_temp: str | None = None
    sidecar_temp: str | None = None
    checkpoint_published = False
    sidecar_published = False

    try:
        with tempfile.NamedTemporaryFile(
            dir=models_dir, prefix=".late_order_", suffix=".joblib.tmp", delete=False
        ) as handle:
            checkpoint_temp = handle.name
        joblib.dump(model, checkpoint_temp)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=models_dir,
            prefix=".late_order_",
            suffix=".metrics.json.tmp",
            delete=False,
        ) as handle:
            sidecar_temp = handle.name
            json.dump(published_sidecar, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

        os.link(checkpoint_temp, checkpoint_path)
        checkpoint_published = True
        os.unlink(checkpoint_temp)
        checkpoint_temp = None
        os.link(sidecar_temp, sidecar_path)
        sidecar_published = True
        os.unlink(sidecar_temp)
        sidecar_temp = None
    except BaseException:
        if checkpoint_published and checkpoint_path.exists():
            checkpoint_path.unlink()
        if sidecar_published and sidecar_path.exists():
            sidecar_path.unlink()
        for temp_path in (checkpoint_temp, sidecar_temp):
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)
        raise

    return checkpoint_path, sidecar_path


def train_once(
    *,
    features_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    threshold: int | None = None,
) -> int | None:
    """Publish one model when enough unseen labeled events are available."""
    config = load_config()
    threshold = int(config["train_every_n_events"]) if threshold is None else threshold
    if threshold <= 0:
        raise ValueError("threshold must be a positive integer")
    features_dir = Path(features_dir) if features_dir is not None else resolve_data_dir("features")
    models_dir = Path(models_dir) if models_dir is not None else resolve_data_dir("models")

    if models_dir.is_dir():
        _remove_orphan_checkpoints(models_dir)
    records, batch_row_counts = _load_feature_batches(features_dir)
    latest_sidecar = _latest_successful_sidecar(models_dir) if models_dir.is_dir() else None
    previously_included = set(latest_sidecar.get("feature_batch_ids", [])) if latest_sidecar else set()
    pending_before = int(latest_sidecar.get("pending_event_count", 0)) if latest_sidecar else 0
    unseen_event_count = sum(
        row_count for batch_id, row_count in batch_row_counts.items() if batch_id not in previously_included
    )
    events_toward_threshold = pending_before + unseen_event_count

    if events_toward_threshold < threshold:
        print(
            f"Training waiting: {events_toward_threshold}/{threshold} new labeled event(s) "
            "since the latest successful model."
        )
        return None

    print(f"Training threshold reached: {events_toward_threshold} new labeled event(s).")
    try:
        model, metrics, split_sizes = _evaluate_and_fit(records)
    except InsufficientTrainingData as exc:
        print(f"Training waiting for more data: {exc}.")
        return None

    positive_count = sum(bool(record[TARGET_COLUMN]) for record in records)
    total_event_count = len(records)
    sidecar = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_COLUMNS),
        "total_event_count": total_event_count,
        "target_class_counts": {"false": total_event_count - positive_count, "true": positive_count},
        "feature_batch_ids": sorted(batch_row_counts),
        "feature_batch_row_counts": dict(sorted(batch_row_counts.items())),
        "split": {
            **split_sizes,
            "test_fraction": TEST_FRACTION,
            "random_state": RANDOM_STATE,
        },
        "metrics": metrics,
        "pending_event_count": events_toward_threshold % threshold,
    }
    try:
        checkpoint_path, sidecar_path = _publish_artifacts(model, sidecar, models_dir)
    except Exception as exc:
        print(f"Training publication failed: {exc}")
        return None

    print(
        "Training published an artifact to disk: "
        f"checkpoint={checkpoint_path.name} sidecar={sidecar_path.name}"
    )
    return int(_CHECKPOINT_RE.fullmatch(checkpoint_path.name).group(1))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and publish a late-order classifier.")
    parser.add_argument("--once", action="store_true", help="train once and exit")
    parser.add_argument("--features-dir", type=Path, help="override the feature directory")
    parser.add_argument("--models-dir", type=Path, help="override the model directory")
    parser.add_argument("--poll-interval", type=float, help="seconds between polls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config()
    features_dir = args.features_dir if args.features_dir is not None else resolve_data_dir("features")
    models_dir = args.models_dir if args.models_dir is not None else resolve_data_dir("models")
    poll_interval = args.poll_interval if args.poll_interval is not None else config["poll_interval_seconds"]

    try:
        while True:
            train_once(features_dir=features_dir, models_dir=models_dir)
            if args.once:
                return 0
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Training stopped cleanly.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())