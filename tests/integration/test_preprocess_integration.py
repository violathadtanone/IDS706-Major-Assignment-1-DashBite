import csv
import json
from pathlib import Path

import pytest

from pipeline.preprocess import preprocess_once, process_raw_csv


@pytest.mark.integration
def test_preprocess_once_processes_finalized_raw_batch(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()

    raw_csv = raw_dir / "orders_2024-01-01T15-00-00Z.csv"
    raw_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "ok-1,2024-01-01T11:30:00Z,12.5,35,26.75,true\n"
        "bad-1,2024-01-01T12:00:00Z,0,35,26.75,true\n"
        "ok-2,2024-01-01T17:45:00+00:00,7.0,18,15.5,false\n",
        encoding="utf-8",
    )

    processed = preprocess_once(raw_dir=raw_dir, features_dir=features_dir)
    captured = capsys.readouterr()

    assert processed == 1
    assert "accepted=2 rejected=1" in captured.out
    assert raw_csv.exists()
    feature_files = list(features_dir.glob("*.csv"))
    assert len(feature_files) == 1
    with feature_files[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["order_id"] == "ok-1"
    assert rows[0]["was_late"] == "True"
    assert rows[0]["hour"] == "11"
    assert rows[0]["is_peak"] == "True"
    assert rows[1]["order_id"] == "ok-2"
    assert rows[1]["was_late"] == "False"
    assert rows[1]["hour"] == "17"
    assert rows[1]["is_peak"] == "True"

    # repeated one-pass run should skip already processed files
    assert preprocess_once(raw_dir=raw_dir, features_dir=features_dir) == 0


@pytest.mark.integration
def test_preprocess_ignores_temporary_and_changing_raw_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()
    raw_csv = raw_dir / "orders_changing.csv"
    original_content = (
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "ok,2024-01-01T11:30:00Z,2.5,12.5,8.25,true\n"
    )
    raw_csv.write_text(original_content, encoding="utf-8")
    temporary_raw = raw_dir / ".orders_pending.csv.tmp"
    temporary_raw.write_text(original_content, encoding="utf-8")

    def change_during_stability_check(_delay):
        raw_csv.write_text(original_content + "later,2024-01-01T12:00:00Z,3,15,9,false\n", encoding="utf-8")

    monkeypatch.setattr("pipeline.preprocess.time.sleep", change_during_stability_check)

    assert preprocess_once(raw_dir=raw_dir, features_dir=features_dir) == 0
    assert list(features_dir.iterdir()) == []

    monkeypatch.setattr("pipeline.preprocess.time.sleep", lambda _delay: None)
    assert preprocess_once(raw_dir=raw_dir, features_dir=features_dir) == 1
    feature_file = features_dir / "orders_changing_features.csv"
    with feature_file.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert temporary_raw.exists()


@pytest.mark.integration
def test_preprocess_once_continues_after_a_failed_batch(tmp_path, capsys, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    raw_dir.mkdir()
    features_dir.mkdir()

    broken_csv = raw_dir / "orders_broken.csv"
    broken_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "ok-1,2024-01-01T11:30:00Z,12.5,35,26.75,true\n",
        encoding="utf-8",
    )

    valid_csv = raw_dir / "orders_valid.csv"
    valid_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "ok-2,2024-01-01T17:45:00+00:00,7.0,18,15.5,false\n",
        encoding="utf-8",
    )

    import pipeline.preprocess as preprocess_module

    original_replace = preprocess_module.os.replace
    failed_once = False

    def fail_first_publish(source, destination):
        nonlocal failed_once
        if Path(destination).name == "orders_broken_features.csv" and not failed_once:
            failed_once = True
            raise OSError("simulated batch failure")
        return original_replace(source, destination)

    monkeypatch.setattr(preprocess_module.os, "replace", fail_first_publish)

    processed = preprocess_once(raw_dir=raw_dir, features_dir=features_dir)
    captured = capsys.readouterr()

    assert processed == 1
    assert any("Skipping batch orders_broken.csv" in line for line in captured.out.splitlines())
    assert broken_csv.exists()
    assert not (features_dir / "orders_broken_features.csv").exists()
    assert not list(features_dir.glob(".*.tmp"))
    feature_files = list(features_dir.glob("*.csv"))
    assert len(feature_files) == 1
    with feature_files[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["order_id"] == "ok-2"

    assert preprocess_once(raw_dir=raw_dir, features_dir=features_dir) == 1
    assert (features_dir / "orders_broken_features.csv").exists()
    assert len(list(features_dir.glob("*.csv"))) == 2
    assert not list(features_dir.glob(".*.tmp"))


@pytest.mark.integration
def test_quality_sidecar_counts_each_invalid_field_and_drop_rate(tmp_path):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    quality_dir = tmp_path / "quality"
    raw_dir.mkdir()
    raw_csv = raw_dir / "orders_quality.csv"
    raw_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        ",invalid,0,-1,-4,maybe\n"
        "valid,2026-10-01T11:30:00Z,5,25,10,true\n",
        encoding="utf-8",
    )

    feature_path = process_raw_csv(raw_csv, features_dir, quality_dir)

    sidecar_path = quality_dir / "orders_quality_quality.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["feature_file"] == feature_path.name
    assert sidecar["source_file"] == raw_csv.name
    assert sidecar["input_rows"] == 2
    assert sidecar["accepted_rows"] == 1
    assert sidecar["rejected_rows"] == 1
    assert sidecar["drop_rate"] == 0.5
    assert sidecar["field_failures"] == {
        "order_id": 1,
        "timestamp": 1,
        "distance_km": 1,
        "prep_minutes": 1,
        "order_value": 1,
        "was_late": 1,
    }
    with feature_path.open(newline="", encoding="utf-8") as handle:
        assert [row["order_id"] for row in csv.DictReader(handle)] == ["valid"]


@pytest.mark.integration
def test_preprocess_backfills_missing_quality_sidecar_without_rewriting_feature(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    quality_dir = tmp_path / "quality"
    raw_dir.mkdir()
    raw_csv = raw_dir / "orders_backfill.csv"
    raw_csv.write_text(
        "order_id,timestamp,distance_km,prep_minutes,order_value,was_late\n"
        "valid,2026-10-01T11:30:00Z,5,25,10,true\n"
        "invalid,2026-10-01T11:30:00Z,0,25,10,false\n",
        encoding="utf-8",
    )
    feature_path = process_raw_csv(raw_csv, features_dir, quality_dir)
    feature_bytes = feature_path.read_bytes()
    sidecar_path = quality_dir / "orders_backfill_quality.json"
    sidecar_path.unlink()

    assert preprocess_once(raw_dir=raw_dir, features_dir=features_dir, quality_dir=quality_dir) == 0

    assert feature_path.read_bytes() == feature_bytes
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["input_rows"] == 2
    assert sidecar["accepted_rows"] == 1
    assert sidecar["rejected_rows"] == 1
    assert "Backfilled quality sidecar" in capsys.readouterr().out


@pytest.mark.integration
def test_missing_raw_source_leaves_feature_untouched_and_quality_unavailable(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    quality_dir = tmp_path / "quality"
    raw_dir.mkdir()
    features_dir.mkdir()
    feature_path = features_dir / "orders_orphan_features.csv"
    feature_path.write_text("already published features\n", encoding="utf-8")
    feature_bytes = feature_path.read_bytes()

    assert preprocess_once(raw_dir=raw_dir, features_dir=features_dir, quality_dir=quality_dir) == 0

    assert feature_path.read_bytes() == feature_bytes
    assert not (quality_dir / "orders_orphan_quality.json").exists()
    assert "cannot backfill" in capsys.readouterr().out
