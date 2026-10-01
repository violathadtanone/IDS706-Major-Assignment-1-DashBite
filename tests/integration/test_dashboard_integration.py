import csv
import json
from datetime import datetime, timezone

import plotly.graph_objects as go
import pytest

from pipeline.dashboard import _render_dashboard, build_dashboard_view, load_dashboard_data


class StubColumn:
    def __init__(self, sink):
        self.sink = sink

    def metric(self, label, value):
        self.sink["metrics"].append((label, value))

    def plotly_chart(self, figure, **_kwargs):
        self.sink["figures"].append(figure)

    def info(self, message):
        self.sink["messages"].append(message)


class StubStreamlit:
    def __init__(self):
        self.sink = {"metrics": [], "figures": [], "messages": []}

    def columns(self, count):
        column_count = count if isinstance(count, int) else len(count)
        return [StubColumn(self.sink) for _ in range(column_count)]

    def plotly_chart(self, figure, **_kwargs):
        self.sink["figures"].append(figure)

    def info(self, message):
        self.sink["messages"].append(message)


def _write_feature_batch(path, rows):
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
        writer.writerows(rows)


@pytest.mark.integration
def test_dashboard_loads_artifacts_builds_three_stories_and_never_mutates_files(tmp_path):
    features_dir = tmp_path / "features"
    predictions_dir = tmp_path / "predictions"
    quality_dir = tmp_path / "quality"
    features_dir.mkdir()
    predictions_dir.mkdir()
    quality_dir.mkdir()
    first_batch = features_dir / "orders_first_features.csv"
    second_batch = features_dir / "orders_second_features.csv"
    _write_feature_batch(
        first_batch,
        [
            {
                "order_id": "a",
                "timestamp": "2026-10-01T11:40:00Z",
                "distance_km": 2,
                "prep_minutes": 12,
                "order_value": 10,
                "was_late": False,
                "hour": 11,
                "is_peak": True,
            },
            {
                "order_id": "b",
                "timestamp": "2026-10-01T11:45:00Z",
                "distance_km": 12,
                "prep_minutes": 38,
                "order_value": 20,
                "was_late": True,
                "hour": 11,
                "is_peak": True,
            },
        ],
    )
    _write_feature_batch(
        second_batch,
        [
            {
                "order_id": "c",
                "timestamp": "2026-10-01T11:50:00Z",
                "distance_km": 4,
                "prep_minutes": 20,
                "order_value": 15,
                "was_late": False,
                "hour": 11,
                "is_peak": True,
            }
        ],
    )
    prediction_path = predictions_dir / "orders_first_predictions.csv"
    prediction_path.write_text(
        "order_id,late_probability,predicted_late,checkpoint_id\n"
        "a,0.8,True,v0001\n"
        "unmatched,0.2,False,v0001\n",
        encoding="utf-8",
    )
    quality_path = quality_dir / "orders_first_quality.json"
    quality_path.write_text(
        json.dumps(
            {
                "feature_file": first_batch.name,
                "input_rows": 2,
                "accepted_rows": 1,
                "rejected_rows": 1,
                "drop_rate": 0.5,
                "field_failures": {"timestamp": 3, "distance_km": 1},
            }
        ),
        encoding="utf-8",
    )
    artifacts = [first_batch, second_batch, prediction_path, quality_path]
    original_bytes = {path: path.read_bytes() for path in artifacts}

    data = load_dashboard_data(
        features_dir=features_dir,
        predictions_dir=predictions_dir,
        quality_dir=quality_dir,
    )
    view = build_dashboard_view(
        data,
        now=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
    )
    stub = StubStreamlit()
    _render_dashboard(stub, go, view)

    assert data["total_feature_batches"] == 2
    assert len(data["quality_records"]) == 1
    assert data["unavailable_quality_batches"] == [second_batch.name]
    assert view["kpis"] == {
        "valid_samples_recent": 3,
        "drop_rate_pct": 50.0,
        "predictions_scored": 2,
    }
    assert view["scores"]["unmatched_count"] == 1
    assert view["field_failures"]["title"].endswith("(history incomplete)")
    assert len(stub.sink["metrics"]) == 3
    assert [label for label, _value in stub.sink["metrics"]] == [
        "Valid samples (60 min)",
        "Data drop rate",
        "Predictions scored",
    ]
    assert len(stub.sink["figures"]) == 3
    assert all(len(figure.data) == 1 for figure in stub.sink["figures"])
    assert stub.sink["figures"][0].data[0].type == "scatter"
    assert stub.sink["figures"][0].layout.title.text == view["volume"]["title"]
    assert stub.sink["figures"][0].layout.yaxis.title.text == "Valid orders"
    assert stub.sink["figures"][1].data[0].type == "scatter"
    assert stub.sink["figures"][1].layout.title.text == view["late_flags"]["title"]
    assert stub.sink["figures"][1].layout.yaxis.title.text == "Predicted late (%)"
    assert stub.sink["figures"][2].data[0].orientation == "h"
    assert list(stub.sink["figures"][2].data[0].y) == ["Timestamp", "Distance"]
    assert stub.sink["figures"][2].layout.yaxis.categoryorder == "array"
    assert list(stub.sink["figures"][2].layout.yaxis.categoryarray) == ["Timestamp", "Distance"]
    assert stub.sink["figures"][2].layout.yaxis.autorange == "reversed"
    assert any("incomplete" in message for message in stub.sink["messages"])
    assert {path: path.read_bytes() for path in artifacts} == original_bytes


@pytest.mark.integration
def test_missing_quality_is_unavailable_not_zero_drop_rate(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    feature_path = features_dir / "orders_no_quality_features.csv"
    _write_feature_batch(
        feature_path,
        [
            {
                "order_id": "a",
                "timestamp": "2026-10-01T11:40:00Z",
                "distance_km": 2,
                "prep_minutes": 12,
                "order_value": 10,
                "was_late": False,
                "hour": 11,
                "is_peak": True,
            }
        ],
    )
    (predictions_dir / "orders_no_quality_predictions.csv").write_text(
        "order_id,late_probability,predicted_late,checkpoint_id\n"
        "a,0.8,True,v0001\n",
        encoding="utf-8",
    )

    data = load_dashboard_data(
        features_dir=features_dir,
        predictions_dir=predictions_dir,
        quality_dir=tmp_path / "quality",
    )
    view = build_dashboard_view(
        data,
        now=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
    )
    stub = StubStreamlit()
    _render_dashboard(stub, go, view)

    assert view["kpis"]["drop_rate_pct"] is None
    assert view["kpis"] == {
        "valid_samples_recent": 1,
        "drop_rate_pct": None,
        "predictions_scored": 1,
    }
    assert view["drop_rate"]["input_rows"] == 0
    assert view["field_failures"]["items"] == []
    assert view["field_failures"]["title"] == "Field-failure data unavailable"
    assert view["volume"]["title"] == "Not enough recent orders to determine a volume trend"
    assert view["late_flags"]["title"] == "Not enough matched predictions to determine a late-flag trend"
    assert len(stub.sink["metrics"]) == 3
    assert len(stub.sink["figures"]) == 2
    assert all(len(figure.data) == 1 for figure in stub.sink["figures"])
    assert [figure.layout.title.text for figure in stub.sink["figures"]] == [
        view["volume"]["title"],
        view["late_flags"]["title"],
    ]
    assert any("no usable quality sidecars" in message for message in stub.sink["messages"])
    assert any(view["field_failures"]["title"] in message for message in stub.sink["messages"])