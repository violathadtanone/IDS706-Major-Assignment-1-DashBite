import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

import pipeline.infer as infer_module
from pipeline.infer import infer_once


def _write_checkpoint_pair(models_dir: Path, version: int, *, corrupted=False):
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"late_order_v{version:04d}.joblib"
    if corrupted:
        checkpoint_path.write_bytes(b"corrupt checkpoint")
    else:
        model = LogisticRegression().fit(
            [[1, 10], [2, 12], [10, 35], [12, 40]],
            [False, False, True, True],
        )
        model.dashbite_feature_names_ = ("distance_km", "prep_minutes")
        joblib.dump(model, checkpoint_path)
    sidecar = {
        "version": version,
        "model_version": f"v{version:04d}",
        "training_timestamp": "2026-10-01T00:00:00+00:00",
        "feature_names": ["distance_km", "prep_minutes"],
        "total_event_count": 10,
        "target_class_counts": {"false": 5, "true": 5},
        "feature_batch_ids": ["orders_sample_features.csv"],
        "feature_batch_row_counts": {"orders_sample_features.csv": 10},
        "split": {"train_size": 8, "test_size": 2, "test_fraction": 0.2, "random_state": 42},
        "metrics": {"accuracy": 0.8, "precision": 0.75, "recall": 0.8, "f1": 0.77, "roc_auc": 0.9},
        "pending_event_count": 0,
    }
    (models_dir / f"late_order_v{version:04d}.metrics.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )


def _write_feature_batch(path: Path):
    path.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late,hour,is_peak\n"
        "order-low,2026-10-01T11:00:00Z,2,12,10,false,11,True\n"
        "order-high,2026-10-01T18:00:00Z,12,38,20,true,18,True\n",
        encoding="utf-8",
    )


@pytest.mark.integration
def test_waits_without_checkpoint_then_scores_newest_and_skips_completed_batch(tmp_path, capsys):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    predictions_dir = tmp_path / "predictions"
    features_dir.mkdir()
    feature_path = features_dir / "orders_sample_features.csv"
    _write_feature_batch(feature_path)

    assert infer_once(
        features_dir=features_dir, models_dir=models_dir, predictions_dir=predictions_dir
    ) == 0
    assert "Inference waiting" in capsys.readouterr().out
    assert not predictions_dir.exists()

    _write_checkpoint_pair(models_dir, 1)
    _write_checkpoint_pair(models_dir, 2)
    assert infer_once(
        features_dir=features_dir, models_dir=models_dir, predictions_dir=predictions_dir
    ) == 1
    output_path = predictions_dir / "orders_sample_predictions.csv"
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == ["order_id", "late_probability", "predicted_late", "checkpoint_id"]
    assert len(rows) == 2
    assert {row["checkpoint_id"] for row in rows} == {"v0002"}
    assert infer_once(
        features_dir=features_dir, models_dir=models_dir, predictions_dir=predictions_dir
    ) == 0


@pytest.mark.integration
def test_running_inference_uses_new_checkpoint_for_later_batch_only(tmp_path, monkeypatch, capsys):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    predictions_dir = tmp_path / "predictions"
    features_dir.mkdir()
    first_batch = features_dir / "orders_existing_features.csv"
    _write_feature_batch(first_batch)
    _write_checkpoint_pair(models_dir, 1)

    poll_count = 0
    first_prediction = predictions_dir / "orders_existing_predictions.csv"
    first_prediction_before_update = None

    def publish_during_poll(_interval):
        nonlocal poll_count, first_prediction_before_update
        if poll_count == 0:
            first_prediction_before_update = first_prediction.read_bytes()
            _write_checkpoint_pair(models_dir, 2)
            _write_feature_batch(features_dir / "orders_later_features.csv")
            poll_count += 1
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(infer_module.time, "sleep", publish_during_poll)
    assert infer_module.main(
        [
            "--features-dir",
            str(features_dir),
            "--models-dir",
            str(models_dir),
            "--predictions-dir",
            str(predictions_dir),
            "--poll-interval",
            "0.01",
        ]
    ) == 0

    later_prediction = predictions_dir / "orders_later_predictions.csv"
    with first_prediction.open(newline="", encoding="utf-8") as handle:
        first_rows = list(csv.DictReader(handle))
    with later_prediction.open(newline="", encoding="utf-8") as handle:
        later_rows = list(csv.DictReader(handle))
    assert {row["checkpoint_id"] for row in first_rows} == {"v0001"}
    assert {row["checkpoint_id"] for row in later_rows} == {"v0002"}
    assert first_prediction.read_bytes() == first_prediction_before_update
    captured = capsys.readouterr()
    assert "Scored orders_existing_features.csv with v0001" in captured.out
    assert "Scored orders_later_features.csv with v0002" in captured.out


@pytest.mark.integration
@pytest.mark.parametrize(
    "invalid_newest",
    ["mismatched_sidecar", "mismatched_model_features", "unloadable_model"],
)
def test_invalid_newest_checkpoint_falls_back_to_next_valid_version(
    tmp_path, capsys, invalid_newest
):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    predictions_dir = tmp_path / "predictions"
    features_dir.mkdir()
    feature_path = features_dir / "orders_retry_features.csv"
    _write_feature_batch(feature_path)
    _write_checkpoint_pair(models_dir, 1)
    _write_checkpoint_pair(models_dir, 2, corrupted=invalid_newest == "unloadable_model")
    if invalid_newest == "mismatched_sidecar":
        sidecar_path = models_dir / "late_order_v0002.metrics.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["feature_names"] = ["prep_minutes", "distance_km"]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    if invalid_newest == "mismatched_model_features":
        checkpoint_path = models_dir / "late_order_v0002.joblib"
        model = joblib.load(checkpoint_path)
        model.dashbite_feature_names_ = ("prep_minutes", "distance_km")
        joblib.dump(model, checkpoint_path)

    assert infer_once(
        features_dir=features_dir, models_dir=models_dir, predictions_dir=predictions_dir
    ) == 1
    output_path = predictions_dir / "orders_retry_predictions.csv"
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["checkpoint_id"] for row in rows} == {"v0001"}
    assert "Skipping checkpoint v0002" in capsys.readouterr().out


@pytest.mark.integration
def test_failed_prediction_write_leaves_no_partial_file_and_can_retry(tmp_path, monkeypatch, capsys):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    predictions_dir = tmp_path / "predictions"
    features_dir.mkdir()
    _write_feature_batch(features_dir / "orders_retry_features.csv")
    _write_checkpoint_pair(models_dir, 1)
    original_replace = infer_module.os.replace

    def fail_replace(source, destination):
        raise OSError("simulated output publish failure")

    monkeypatch.setattr(infer_module.os, "replace", fail_replace)
    assert infer_once(
        features_dir=features_dir, models_dir=models_dir, predictions_dir=predictions_dir
    ) == 0
    assert "Failed to score feature batch" in capsys.readouterr().out
    assert not list(predictions_dir.glob("*_predictions.csv"))
    assert not list(predictions_dir.glob(".*.tmp"))

    monkeypatch.setattr(infer_module.os, "replace", original_replace)
    assert infer_once(
        features_dir=features_dir, models_dir=models_dir, predictions_dir=predictions_dir
    ) == 1
    with (predictions_dir / "orders_retry_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["checkpoint_id"] for row in rows} == {"v0001"}


@pytest.mark.integration
def test_inference_runs_while_training_imports_are_blocked(tmp_path):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    predictions_dir = tmp_path / "predictions"
    features_dir.mkdir()
    _write_feature_batch(features_dir / "orders_isolated_features.csv")
    _write_checkpoint_pair(models_dir, 1)
    script = """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pipeline.train" or fullname.startswith("pipeline.train."):
            raise AssertionError("inference attempted to import training")
        return None

sys.meta_path.insert(0, BlockTrainingImports())
from pipeline.infer import infer_once

assert infer_once(features_dir=sys.argv[1], models_dir=sys.argv[2], predictions_dir=sys.argv[3]) == 1
assert not any(name == "pipeline.train" or name.startswith("pipeline.train.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(features_dir), str(models_dir), str(predictions_dir)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (predictions_dir / "orders_isolated_predictions.csv").is_file()