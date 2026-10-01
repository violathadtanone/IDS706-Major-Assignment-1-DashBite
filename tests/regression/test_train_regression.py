import csv
import json

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline.train import train_once


def _write_batch(path, labels, offset=0):
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


@pytest.mark.regression
def test_published_artifacts_have_stable_versioned_provenance_and_metrics(tmp_path):
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    features_dir.mkdir()
    batch_one = features_dir / "orders_one_features.csv"
    _write_batch(batch_one, [False, True] * 5)

    assert train_once(features_dir=features_dir, models_dir=models_dir, threshold=1) == 1
    checkpoint_one = models_dir / "late_order_v0001.joblib"
    sidecar_one = models_dir / "late_order_v0001.metrics.json"
    checkpoint_bytes = checkpoint_one.read_bytes()
    metadata_one = json.loads(sidecar_one.read_text(encoding="utf-8"))

    assert isinstance(joblib.load(checkpoint_one), LogisticRegression)
    assert metadata_one["model_version"] == "v0001"
    assert metadata_one["version"] == 1
    assert metadata_one["feature_names"] == ["distance_km", "prep_minutes"]
    assert metadata_one["total_event_count"] == 10
    assert metadata_one["target_class_counts"] == {"false": 5, "true": 5}
    assert metadata_one["feature_batch_ids"] == [batch_one.name]
    assert metadata_one["feature_batch_row_counts"] == {batch_one.name: 10}
    assert metadata_one["split"] == {
        "train_size": 8,
        "test_size": 2,
        "test_fraction": 0.2,
        "random_state": 42,
    }
    assert set(metadata_one["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    batch_two = features_dir / "orders_two_features.csv"
    _write_batch(batch_two, [False, True], offset=10)
    assert train_once(features_dir=features_dir, models_dir=models_dir, threshold=1) == 2

    assert checkpoint_one.read_bytes() == checkpoint_bytes
    assert (models_dir / "late_order_v0002.joblib").exists()
    assert (models_dir / "late_order_v0002.metrics.json").exists()