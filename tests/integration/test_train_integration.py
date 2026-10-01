import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import joblib
from sklearn.linear_model import LogisticRegression

import pipeline.train as train_module
from pipeline.train import train_once


def _write_batch(path: Path, labels, offset=0):
    columns = [
        "order_id",
        "timestamp",
        "distance_km",
        "prep_minutes",
        "order_value",
        "was_late",
        "hour",
        "is_peak",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow(
                {
                    "order_id": f"order-{offset + index}",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "distance_km": 10 + index if label else 1 + index,
                    "prep_minutes": 30 + index if label else 10 + index,
                    "order_value": 12.5,
                    "was_late": str(label),
                    "hour": 12,
                    "is_peak": True,
                }
            )


@pytest.mark.integration
def test_training_threshold_restart_and_new_batches(tmp_path, monkeypatch, capsys):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    features_dir.mkdir()
    monkeypatch.setenv("TRAIN_EVERY_N_EVENTS", "5")

    first_batch = features_dir / "orders_first_features.csv"
    _write_batch(first_batch, [False, True, False, True])
    assert train_once(features_dir=features_dir, models_dir=models_dir) is None
    assert not models_dir.exists()

    second_batch = features_dir / "orders_second_features.csv"
    _write_batch(second_batch, [False, True, False, True], offset=4)
    assert train_once(features_dir=features_dir, models_dir=models_dir) == 1
    first_metadata = json.loads((models_dir / "late_order_v0001.metrics.json").read_text())
    assert first_metadata["total_event_count"] == 8
    assert first_metadata["pending_event_count"] == 3
    assert first_metadata["feature_batch_ids"] == [first_batch.name, second_batch.name]

    assert train_once(features_dir=features_dir, models_dir=models_dir) is None
    assert len(list(models_dir.glob("late_order_v*.joblib"))) == 1

    third_batch = features_dir / "orders_third_features.csv"
    _write_batch(third_batch, [True, False], offset=8)
    assert train_once(features_dir=features_dir, models_dir=models_dir) == 2
    second_metadata = json.loads((models_dir / "late_order_v0002.metrics.json").read_text())
    assert second_metadata["total_event_count"] == 10
    assert second_metadata["pending_event_count"] == 0
    assert second_metadata["feature_batch_row_counts"][third_batch.name] == 2
    assert "Training waiting: 4/5" in capsys.readouterr().out


@pytest.mark.integration
def test_failed_sidecar_publication_removes_checkpoint_and_temporary_files(tmp_path, monkeypatch, capsys):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    features_dir.mkdir()
    _write_batch(features_dir / "orders_batch_features.csv", [False, True] * 5)
    original_link = train_module.os.link

    def fail_sidecar_link(source, destination):
        if Path(destination).name.endswith(".metrics.json"):
            raise OSError("simulated sidecar publication failure")
        return original_link(source, destination)

    monkeypatch.setattr(train_module.os, "link", fail_sidecar_link)

    assert train_once(features_dir=features_dir, models_dir=models_dir, threshold=1) is None

    assert "Training publication failed" in capsys.readouterr().out
    assert not list(models_dir.glob("late_order_v*.joblib"))
    assert not list(models_dir.glob("late_order_v*.metrics.json"))
    assert not list(models_dir.glob(".*.tmp"))


@pytest.mark.integration
@pytest.mark.parametrize("single_class_label", [False, True])
def test_training_waits_for_both_classes_after_threshold(tmp_path, capsys, single_class_label):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    features_dir.mkdir()
    batch_name = "orders_positive_only_features.csv" if single_class_label else "orders_negative_only_features.csv"
    _write_batch(features_dir / batch_name, [single_class_label] * 6)

    assert train_once(features_dir=features_dir, models_dir=models_dir, threshold=5) is None
    assert "Training waiting for more data" in capsys.readouterr().out
    assert not list(models_dir.glob("late_order_v*.joblib"))
    assert not list(models_dir.glob("late_order_v*.metrics.json"))
    assert not models_dir.exists()

    _write_batch(features_dir / "orders_complementary_features.csv", [not single_class_label] * 6, offset=6)
    assert train_once(features_dir=features_dir, models_dir=models_dir, threshold=5) == 1


@pytest.mark.integration
def test_orphan_checkpoint_is_removed_and_never_reused_as_a_model(tmp_path):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    features_dir.mkdir()
    models_dir.mkdir()
    orphan_checkpoint = models_dir / "late_order_v0001.joblib"
    orphan_model = LogisticRegression().fit([[1, 10], [10, 40]], [False, True])
    joblib.dump(orphan_model, orphan_checkpoint)
    _write_batch(features_dir / "orders_batch_features.csv", [False, True] * 5)

    assert train_module._latest_successful_sidecar(models_dir) is None
    assert train_once(features_dir=features_dir, models_dir=models_dir, threshold=1) == 1

    published_checkpoint = models_dir / "late_order_v0001.joblib"
    published_sidecar = models_dir / "late_order_v0001.metrics.json"
    metadata = json.loads(published_sidecar.read_text(encoding="utf-8"))
    assert isinstance(joblib.load(published_checkpoint), LogisticRegression)
    assert metadata["model_version"] == "v0001"
    assert metadata["feature_names"] == ["distance_km", "prep_minutes"]


@pytest.mark.integration
def test_training_publishes_when_inference_imports_are_blocked(tmp_path):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    features_dir.mkdir()
    _write_batch(features_dir / "orders_batch_features.csv", [False, True] * 5)

    script = """
import importlib.abc
import sys

class BlockInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pipeline.infer" or fullname.startswith("pipeline.infer."):
            raise AssertionError("training attempted to import inference")
        return None

sys.meta_path.insert(0, BlockInferenceImports())
from pipeline.train import train_once

assert train_once(features_dir=sys.argv[1], models_dir=sys.argv[2], threshold=1) == 1
assert not any(name == "pipeline.infer" or name.startswith("pipeline.infer.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(features_dir), str(models_dir)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (models_dir / "late_order_v0001.joblib").is_file()
    assert (models_dir / "late_order_v0001.metrics.json").is_file()