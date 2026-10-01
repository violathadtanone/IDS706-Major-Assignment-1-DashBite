import csv

import pytest

from pipeline.preprocess import process_raw_csv


@pytest.mark.regression
def test_process_raw_csv_uses_expected_columns_and_hour_boundaries(tmp_path):
    raw_csv = tmp_path / "orders_2024-01-01T10-00-00Z.csv"
    raw_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "a,2024-01-01T10:59:00Z,5,25,10,true\n"
        "b,2024-01-01T11:00:00Z,5,25,10,false\n"
        "c,2024-01-01T13:59:00Z,5,25,10,true\n"
        "d,2024-01-01T14:00:00Z,5,25,10,false\n"
        "e,2024-01-01T17:00:00Z,5,25,10,true\n"
        "f,2024-01-01T20:59:00Z,5,25,10,false\n"
        "g,2024-01-01T21:00:00Z,5,25,10,true\n",
        encoding="utf-8",
    )

    feature_path = process_raw_csv(raw_csv, tmp_path / "features")

    expected_columns = [
        "order_id",
        "timestamp",
        "distance_km",
        "prep_minutes",
        "order_value",
        "was_late",
        "hour",
        "is_peak",
    ]
    with feature_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert feature_path.name == "orders_2024-01-01T10-00-00Z_features.csv"
    assert rows[0]["hour"] == "10"
    assert rows[1]["hour"] == "11"
    assert rows[1]["is_peak"] == "True"
    assert rows[2]["hour"] == "13"
    assert rows[2]["is_peak"] == "True"
    assert rows[3]["hour"] == "14"
    assert rows[3]["is_peak"] == "False"
    assert rows[4]["hour"] == "17"
    assert rows[4]["is_peak"] == "True"
    assert rows[5]["hour"] == "20"
    assert rows[5]["is_peak"] == "True"
    assert rows[6]["hour"] == "21"
    assert rows[6]["is_peak"] == "False"
    assert list(rows[0].keys()) == expected_columns


@pytest.mark.regression
def test_process_raw_csv_converts_offset_timestamp_to_utc_hour(tmp_path):
    raw_csv = tmp_path / "orders_offset.csv"
    raw_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "offset,2024-01-01T13:00:00+02:00,5,25,10,false\n",
        encoding="utf-8",
    )

    feature_path = process_raw_csv(raw_csv, tmp_path / "features")

    with feature_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["timestamp"] == "2024-01-01T13:00:00+02:00"
    assert row["hour"] == "11"
    assert row["is_peak"] == "True"
    assert row["was_late"] == "False"
