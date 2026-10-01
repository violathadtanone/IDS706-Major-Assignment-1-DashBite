from pathlib import Path

import pytest

from pipeline.paths import ensure_data_dirs, resolve_data_dir


@pytest.mark.unit
def test_resolve_data_dir_uses_repo_root():
    root = Path(__file__).resolve().parents[2]

    for name in ("raw", "features", "models", "predictions", "quality"):
        resolved = resolve_data_dir(name)
        assert resolved == root / "data" / name


@pytest.mark.unit
def test_resolve_data_dir_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unsupported data directory name"):
        resolve_data_dir("unknown")


@pytest.mark.unit
def test_ensure_data_dirs_creates_and_is_idempotent(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])

    first = ensure_data_dirs()
    second = ensure_data_dirs()

    assert set(first) == {"raw", "features", "models", "predictions", "quality"}
    assert first == second
    assert all(path.exists() for path in first.values())
    assert all(path.is_dir() for path in first.values())
