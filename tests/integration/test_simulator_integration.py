import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.simulator import main


@pytest.mark.integration
def test_module_once_writes_configured_batch_to_output_dir(tmp_path, monkeypatch):
    output_dir = tmp_path / "custom_raw"
    environment = os.environ.copy()
    environment["BATCH_SIZE"] = "5"
    environment["POLL_INTERVAL_SECONDS"] = "0.1"

    result = subprocess.run(
        [sys.executable, "-m", "pipeline.simulator", "--once", "--output-dir", str(output_dir)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "New orders arrived: batch_size=5" in result.stdout

    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) == 1
    with csv_files[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "order_id",
            "timestamp",
            "distance_km",
            "prep_minutes",
            "order_value",
            "was_late",
        ]
        rows = list(reader)
    assert len(rows) == 5
    assert list(output_dir.iterdir()) == csv_files


@pytest.mark.integration
def test_main_once_uses_only_shared_raw_directory(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    resolved_names = []

    def resolve_data_dir(name):
        resolved_names.append(name)
        return data_root / name

    monkeypatch.setattr("pipeline.simulator.resolve_data_dir", resolve_data_dir)
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "0.1")

    exit_code = main(["--once"])

    assert exit_code == 0
    assert resolved_names == ["raw"]
    assert {path.name for path in data_root.iterdir()} == {"raw"}

    csv_files = list((data_root / "raw").glob("*.csv"))
    assert len(csv_files) == 1
    with csv_files[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "order_id",
            "timestamp",
            "distance_km",
            "prep_minutes",
            "order_value",
            "was_late",
        ]
        assert len(list(reader)) == 2
