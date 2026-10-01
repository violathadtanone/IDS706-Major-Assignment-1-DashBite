import csv
from pathlib import Path

import pytest

from pipeline.simulator import generate_batch, write_batch_csv


@pytest.mark.regression
def test_simulator_writes_distinct_csv_batches_beneath_raw_dir(tmp_path):
    output_dir = tmp_path / "raw"

    first = write_batch_csv(generate_batch(3, seed=1), output_dir)
    second = write_batch_csv(generate_batch(3, seed=2), output_dir)

    assert first != second
    assert first.parent == output_dir
    assert second.parent == output_dir
    assert first.name.endswith(".csv")
    assert second.name.endswith(".csv")

    batch_order_ids = []
    for batch_path in (first, second):
        with batch_path.open(newline="") as handle:
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
            assert len(rows) == 3
            batch_order_ids.append({row["order_id"] for row in rows})

    assert batch_order_ids[0].isdisjoint(batch_order_ids[1])

    csv_files = sorted(output_dir.glob("*.csv"))
    assert len(csv_files) == 2
    assert set(csv_files) == {first, second}
