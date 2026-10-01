import argparse
import csv
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import joblib

from pipeline.config import load_config
from pipeline.paths import resolve_data_dir

FEATURE_COLUMNS = ("distance_km", "prep_minutes")
PREDICTION_COLUMNS = ("order_id", "late_probability", "predicted_late", "checkpoint_id")
_CHECKPOINT_RE = re.compile(r"late_order_v(\d{4,})\.joblib$")
_SIDECAR_RE = re.compile(r"late_order_v(\d{4,})\.metrics\.json$")
_FEATURE_SUFFIX = "_features.csv"


def _artifact_version(path: Path, pattern: re.Pattern[str]) -> int | None:
    match = pattern.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _candidate_versions(models_dir: Path) -> list[int]:
    versions = set()
    for pattern, glob_pattern in (
        (_CHECKPOINT_RE, "late_order_v*.joblib"),
        (_SIDECAR_RE, "late_order_v*.metrics.json"),
    ):
        for path in models_dir.glob(glob_pattern):
            version = _artifact_version(path, pattern)
            if version is not None:
                versions.add(version)
    return sorted(versions, reverse=True)


def _sidecar_is_valid(sidecar: object, version: int) -> bool:
    if not isinstance(sidecar, dict):
        return False
    total_events = sidecar.get("total_event_count")
    class_counts = sidecar.get("target_class_counts")
    batch_ids = sidecar.get("feature_batch_ids")
    batch_counts = sidecar.get("feature_batch_row_counts")
    split = sidecar.get("split")
    metrics = sidecar.get("metrics")
    if (
        sidecar.get("version") != version
        or sidecar.get("model_version") != f"v{version:04d}"
        or sidecar.get("feature_names") != list(FEATURE_COLUMNS)
        or not isinstance(sidecar.get("training_timestamp"), str)
        or not sidecar["training_timestamp"]
        or not isinstance(total_events, int)
        or total_events <= 0
        or not isinstance(class_counts, dict)
        or not isinstance(batch_ids, list)
        or not all(isinstance(batch_id, str) for batch_id in batch_ids)
        or len(set(batch_ids)) != len(batch_ids)
        or not isinstance(batch_counts, dict)
        or set(batch_counts) != set(batch_ids)
        or not all(isinstance(count, int) and count >= 0 for count in batch_counts.values())
        or sum(batch_counts.values()) != total_events
        or not isinstance(split, dict)
        or split.get("random_state") != 42
        or split.get("test_fraction") != 0.2
        or split.get("train_size", 0) + split.get("test_size", 0) != total_events
        or not isinstance(metrics, dict)
        or set(metrics) != {"accuracy", "precision", "recall", "f1", "roc_auc"}
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
    ):
        return False
    if (
        not isinstance(class_counts.get("false"), int)
        or not isinstance(class_counts.get("true"), int)
        or class_counts["false"] <= 0
        or class_counts["true"] <= 0
        or class_counts["false"] + class_counts["true"] != total_events
    ):
        return False
    return True


def _load_usable_checkpoint(models_dir: Path):
    for version in _candidate_versions(models_dir):
        checkpoint_path = models_dir / f"late_order_v{version:04d}.joblib"
        sidecar_path = models_dir / f"late_order_v{version:04d}.metrics.json"
        if not checkpoint_path.is_file() or not sidecar_path.is_file():
            print(f"Skipping checkpoint v{version:04d}: incomplete checkpoint/sidecar pair.")
            continue

        try:
            with sidecar_path.open(encoding="utf-8") as handle:
                sidecar = json.load(handle)
            feature_names = sidecar.get("feature_names")
            if not _sidecar_is_valid(sidecar, version):
                raise ValueError("sidecar is incomplete or inconsistent")

            model = joblib.load(checkpoint_path)
            classes = list(model.classes_)
            true_indices = [index for index, label in enumerate(classes) if bool(label)]
            false_indices = [index for index, label in enumerate(classes) if not bool(label)]
            if (
                not callable(getattr(model, "predict_proba", None))
                or getattr(model, "n_features_in_", None) != len(feature_names)
                or len(classes) != 2
                or len(true_indices) != 1
                or len(false_indices) != 1
            ):
                raise ValueError("checkpoint does not expose the expected binary scoring interface")
            model_feature_names = getattr(model, "feature_names_in_", None)
            if model_feature_names is None:
                model_feature_names = getattr(model, "dashbite_feature_names_", None)
            if model_feature_names is None or list(model_feature_names) != feature_names:
                raise ValueError("checkpoint feature order does not match its sidecar")
        except Exception as exc:
            print(f"Skipping checkpoint v{version:04d}: {exc}")
            continue

        model_version = f"v{version:04d}"
        print(f"Loaded checkpoint {model_version} from {checkpoint_path.name}.")
        return model, model_version, feature_names
    return None


def _prediction_path(feature_path: Path, predictions_dir: Path) -> Path:
    if not feature_path.name.endswith(_FEATURE_SUFFIX):
        raise ValueError(f"Feature batch name must end with {_FEATURE_SUFFIX!r}.")
    prediction_name = feature_path.name[: -len(_FEATURE_SUFFIX)] + "_predictions.csv"
    return predictions_dir / prediction_name


def _read_feature_rows(feature_path: Path, feature_names: list[str]) -> list[dict[str, object]]:
    rows = []
    with feature_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in ("order_id", *feature_names)
        ):
            raise ValueError("feature batch is missing order_id or required model feature columns")
        for row in reader:
            order_id = str(row.get("order_id", "")).strip()
            if not order_id:
                raise ValueError("feature batch contains a blank order_id")
            features = []
            for feature_name in feature_names:
                try:
                    value = float(row[feature_name] or "")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {feature_name} value for {order_id}") from exc
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {feature_name} value for {order_id}")
                features.append(value)
            rows.append({"order_id": order_id, "features": features})
    return rows


def _write_predictions(
    feature_path: Path,
    predictions_dir: Path,
    model,
    model_version: str,
    feature_names: list[str],
) -> Path:
    rows = _read_feature_rows(feature_path, feature_names)
    inputs = [row["features"] for row in rows]
    probabilities = model.predict_proba(inputs) if inputs else []
    classes = list(model.classes_)
    positive_index = next(index for index, label in enumerate(classes) if bool(label))
    if len(probabilities) != len(rows):
        raise ValueError("checkpoint returned an unexpected number of predictions")

    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_path = _prediction_path(feature_path, predictions_dir)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=predictions_dir,
            prefix=".predictions_",
            suffix=".csv.tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for row, row_probabilities in zip(rows, probabilities):
                probability = float(row_probabilities[positive_index])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("checkpoint returned an invalid probability")
                writer.writerow(
                    {
                        "order_id": row["order_id"],
                        "late_probability": probability,
                        "predicted_late": probability >= 0.5,
                        "checkpoint_id": model_version,
                    }
                )
        os.replace(temp_path, output_path)
    except BaseException:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return output_path


def infer_once(
    *,
    features_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    predictions_dir: str | Path | None = None,
) -> int:
    """Score unprocessed feature batches with the newest usable published model."""
    features_dir = Path(features_dir) if features_dir is not None else resolve_data_dir("features")
    models_dir = Path(models_dir) if models_dir is not None else resolve_data_dir("models")
    predictions_dir = (
        Path(predictions_dir) if predictions_dir is not None else resolve_data_dir("predictions")
    )
    feature_paths = sorted(features_dir.glob(f"*{_FEATURE_SUFFIX}")) if features_dir.is_dir() else []
    processed = 0

    for feature_path in feature_paths:
        try:
            output_path = _prediction_path(feature_path, predictions_dir)
        except ValueError as exc:
            print(f"Skipping feature batch {feature_path.name}: {exc}")
            continue
        if output_path.is_file():
            continue

        selected = _load_usable_checkpoint(models_dir)
        if selected is None:
            print("Inference waiting: no usable checkpoint/sidecar pair is available.")
            break
        model, model_version, feature_names = selected
        try:
            output_path = _write_predictions(
                feature_path, predictions_dir, model, model_version, feature_names
            )
        except Exception as exc:
            print(f"Failed to score feature batch {feature_path.name}: {exc}")
            continue
        print(f"Scored {feature_path.name} with {model_version}: {output_path.name}.")
        processed += 1
    return processed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score feature batches with a published late-order model.")
    parser.add_argument("--once", action="store_true", help="score one pass and exit")
    parser.add_argument("--features-dir", type=Path, help="override the feature directory")
    parser.add_argument("--models-dir", type=Path, help="override the model directory")
    parser.add_argument("--predictions-dir", type=Path, help="override the prediction directory")
    parser.add_argument("--poll-interval", type=float, help="seconds between polls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config()
    features_dir = args.features_dir if args.features_dir is not None else resolve_data_dir("features")
    models_dir = args.models_dir if args.models_dir is not None else resolve_data_dir("models")
    predictions_dir = (
        args.predictions_dir if args.predictions_dir is not None else resolve_data_dir("predictions")
    )
    poll_interval = args.poll_interval if args.poll_interval is not None else config["poll_interval_seconds"]

    try:
        while True:
            infer_once(
                features_dir=features_dir,
                models_dir=models_dir,
                predictions_dir=predictions_dir,
            )
            if args.once:
                return 0
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Inference stopped cleanly.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())