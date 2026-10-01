import json
import os

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from pipeline.infer import _load_usable_checkpoint


def _write_checkpoint_pair(models_dir, version):
    model = LogisticRegression().fit(
        [[1, 10], [2, 12], [10, 35], [12, 40]],
        [False, False, True, True],
    )
    model.dashbite_feature_names_ = ("distance_km", "prep_minutes")
    checkpoint_path = models_dir / f"late_order_v{version:04d}.joblib"
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
    return checkpoint_path


@pytest.mark.regression
def test_checkpoint_selection_uses_version_not_modification_time_and_ignores_temporary_files(tmp_path):
    older_checkpoint = _write_checkpoint_pair(tmp_path, 2)
    newest_checkpoint = _write_checkpoint_pair(tmp_path, 10)
    (tmp_path / "late_order_v9999.joblib.tmp").write_bytes(b"not a checkpoint")
    os.utime(older_checkpoint, (2_000_000_000, 2_000_000_000))
    os.utime(newest_checkpoint, (1_000_000_000, 1_000_000_000))

    selected = _load_usable_checkpoint(tmp_path)

    assert selected is not None
    assert selected[1] == "v0010"
    assert selected[2] == ["distance_km", "prep_minutes"]


@pytest.mark.regression
def test_checkpoint_selection_falls_back_from_incomplete_newest_pair(tmp_path, capsys):
    _write_checkpoint_pair(tmp_path, 2)
    _write_checkpoint_pair(tmp_path, 10)
    (tmp_path / "late_order_v0010.metrics.json").unlink()

    selected = _load_usable_checkpoint(tmp_path)

    assert selected is not None
    assert selected[1] == "v0002"
    assert "Skipping checkpoint v0010: incomplete checkpoint/sidecar pair." in capsys.readouterr().out