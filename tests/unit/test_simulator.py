import csv
from datetime import datetime, timezone

import pytest

import pipeline.simulator as simulator
from pipeline.simulator import CSV_COLUMNS, generate_batch, write_batch_csv


@pytest.mark.unit
def test_generate_batch_uses_configured_size_and_schema():
    rows = generate_batch(4, seed=7)

    assert len(rows) == 4
    assert len({row["order_id"] for row in rows}) == 4
    for row in rows:
        assert set(row) == set(CSV_COLUMNS)
        assert isinstance(row["timestamp"], str)
        assert isinstance(row["distance_km"], float)
        assert isinstance(row["prep_minutes"], int)
        assert isinstance(row["order_value"], float)
        assert isinstance(row["was_late"], bool)


@pytest.mark.unit
def test_generate_batch_matches_seeded_golden_fixture(monkeypatch):
    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 1, 2, 3, 4, 5, 6789, tzinfo=tz or timezone.utc)

    monkeypatch.setattr(simulator, "datetime", FrozenDateTime)

    assert generate_batch(2, seed=7) == [
        {
            "order_id": "order-6513270e269e0d37f2a74de452e6b438",
            "timestamp": "2024-01-02T03:04:05.006789Z",
            "distance_km": 10.05,
            "prep_minutes": 29,
            "order_value": 50.4,
            "was_late": True,
        },
        {
            "order_id": "order-d23f0824128b2f330c5c7fd0a6a3a450",
            "timestamp": "2024-01-02T03:04:05.006789Z",
            "distance_km": 2.64,
            "prep_minutes": 78,
            "order_value": 15.82,
            "was_late": True,
        },
    ]


@pytest.mark.unit
def test_write_batch_csv_writes_expected_header(tmp_path):
    rows = generate_batch(2, seed=3)
    output_dir = tmp_path / "raw"

    final_path = write_batch_csv(rows, output_dir)

    assert final_path.parent == output_dir
    assert final_path.suffix == ".csv"
    with final_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_COLUMNS
        written_rows = list(reader)
    assert len(written_rows) == 2


@pytest.mark.unit
def test_write_batch_csv_removes_temp_file_on_interrupt_and_keeps_completed_csvs(monkeypatch, tmp_path):
    rows = generate_batch(2, seed=4)
    output_dir = tmp_path / "raw"
    output_dir.mkdir(parents=True)

    completed_path = output_dir / "orders_before.csv"
    completed_path.write_text("order_id,timestamp,distance_km,prep_minutes,order_value,was_late\nkeep,2024-01-01T00:00:00Z,1.0,10,2.5,true\n")

    original_replace = __import__("pipeline.simulator").simulator.os.replace

    def fail_replace(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr("pipeline.simulator.os.replace", fail_replace)

    with pytest.raises(KeyboardInterrupt):
        write_batch_csv(rows, output_dir)

    assert completed_path.exists()
    assert completed_path.read_text() == "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\nkeep,2024-01-01T00:00:00Z,1.0,10,2.5,true\n"
    assert not any(path.name.endswith(".tmp") for path in output_dir.iterdir())
    assert not any(path.name.endswith(".csv.tmp") for path in output_dir.iterdir())
    assert list(output_dir.glob("*.csv")) == [completed_path]
