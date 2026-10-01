from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.paths import ensure_data_dirs


@pytest.mark.regression
def test_required_defaults_are_locked(monkeypatch):
    monkeypatch.delenv("TRAIN_EVERY_N_EVENTS", raising=False)
    monkeypatch.delenv("BATCH_SIZE", raising=False)
    monkeypatch.delenv("POLL_INTERVAL_SECONDS", raising=False)

    config = load_config()

    assert config["train_every_n_events"] == 2000
    assert config["batch_size"] == 50
    assert config["poll_interval_seconds"] == 15.0


@pytest.mark.regression
def test_required_data_directory_names_are_locked():
    directories = ensure_data_dirs()

    assert set(directories) == {"raw", "features", "models", "predictions", "quality"}
    assert all(isinstance(path, Path) for path in directories.values())
