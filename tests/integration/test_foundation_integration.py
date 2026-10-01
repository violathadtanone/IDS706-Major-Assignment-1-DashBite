from pathlib import Path

import pytest

from pipeline.config import load_config
from pipeline.paths import ensure_data_dirs


@pytest.mark.integration
def test_load_config_and_data_dirs_from_another_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIN_EVERY_N_EVENTS", "25")
    monkeypatch.setenv("BATCH_SIZE", "12")

    config = load_config()
    directories = ensure_data_dirs()
    root = Path(__file__).resolve().parents[2]
    expected = {
        "raw": root / "data" / "raw",
        "features": root / "data" / "features",
        "models": root / "data" / "models",
        "predictions": root / "data" / "predictions",
        "quality": root / "data" / "quality",
    }

    assert config["train_every_n_events"] == 25
    assert config["batch_size"] == 12
    assert directories == expected
    assert all(path.exists() for path in directories.values())
    assert all(path == expected[name] for name, path in directories.items())
