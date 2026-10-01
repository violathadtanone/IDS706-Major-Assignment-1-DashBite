from datetime import datetime, timedelta, timezone

import pytest

from pipeline.pulse import (
    drop_rate_summary,
    late_flag_rate_over_time,
    ranked_field_failures,
    sample_volume,
    score_summary,
    volume_over_time,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_sample_volume_and_utc_minute_buckets_filter_to_recent_window():
    rows = [
        {"order_id": "old", "timestamp": "2026-10-01T11:00:59Z"},
        {"order_id": "first", "timestamp": "2026-10-01T10:01:00-01:00"},
        {"order_id": "last", "timestamp": "2026-10-01T12:00:00Z"},
        {"order_id": "future", "timestamp": "2026-10-01T12:01:00Z"},
        {"order_id": "naive", "timestamp": "2026-10-01T11:45:00"},
    ]

    assert sample_volume(rows, now=NOW) == 2
    result = volume_over_time(rows, now=NOW)
    assert len(result["points"]) == 60
    assert result["points"][0] == {"minute": "2026-10-01T11:01:00+00:00", "sample_count": 1}
    assert result["points"][-1] == {"minute": "2026-10-01T12:00:00+00:00", "sample_count": 1}


@pytest.mark.unit
def test_score_summary_counts_matched_unmatched_flags_and_mean_probability():
    features = [{"order_id": "a"}, {"order_id": "b"}]
    predictions = [
        {"order_id": "a", "predicted_late": "True", "late_probability": "0.8"},
        {"order_id": "missing", "predicted_late": "False", "late_probability": "0.2"},
    ]

    assert score_summary(features, predictions) == {
        "scored_count": 2,
        "matched_count": 1,
        "unmatched_count": 1,
        "flagged_count": 1,
        "mean_late_probability": pytest.approx(0.5),
    }
    assert score_summary([], []) == {
        "scored_count": 0,
        "matched_count": 0,
        "unmatched_count": 0,
        "flagged_count": 0,
        "mean_late_probability": None,
    }


@pytest.mark.unit
def test_late_flag_rate_joins_by_order_id_and_excludes_unmatched_predictions():
    features = [
        {"order_id": "a", "timestamp": "2026-10-01T11:10:30Z"},
        {"order_id": "b", "timestamp": "2026-10-01T11:10:50Z"},
    ]
    predictions = [
        {"order_id": "a", "predicted_late": "true"},
        {"order_id": "b", "predicted_late": "false"},
        {"order_id": "unmatched", "predicted_late": "true"},
    ]

    result = late_flag_rate_over_time(features, predictions, now=NOW)
    point = next(point for point in result["points"] if point["minute"] == "2026-10-01T11:10:00+00:00")
    assert point == {"minute": "2026-10-01T11:10:00+00:00", "matched_count": 2, "late_rate_pct": 50.0}
    assert result["title"] == "Not enough matched predictions to determine a late-flag trend"


@pytest.mark.unit
def test_late_flag_rate_excludes_invalid_and_missing_prediction_labels():
    features = [
        {"order_id": order_id, "timestamp": "2026-10-01T11:10:00Z"}
        for order_id in ("true", "false", "empty", "invalid", "missing")
    ]
    predictions = [
        {"order_id": "true", "predicted_late": "true"},
        {"order_id": "false", "predicted_late": "false"},
        {"order_id": "empty", "predicted_late": ""},
        {"order_id": "invalid", "predicted_late": "sometimes"},
        {"order_id": "missing"},
    ]

    result = late_flag_rate_over_time(features, predictions, now=NOW)
    point = next(point for point in result["points"] if point["minute"] == "2026-10-01T11:10:00+00:00")

    assert point["matched_count"] == 2
    assert point["late_rate_pct"] == 50.0


@pytest.mark.unit
def test_drop_rate_uses_only_available_quality_rows_and_handles_no_input():
    records = [
        {"input_rows": 10, "accepted_rows": 7, "rejected_rows": 3},
        {"input_rows": 5, "accepted_rows": 4, "rejected_rows": 1},
    ]

    result = drop_rate_summary(records, total_batches=3)

    assert result == {
        "input_rows": 15,
        "accepted_rows": 11,
        "rejected_rows": 4,
        "drop_rate_pct": pytest.approx(100 * 4 / 15),
        "available_batches": 2,
        "total_batches": 3,
        "history_complete": False,
    }
    assert drop_rate_summary([])["drop_rate_pct"] is None


@pytest.mark.unit
def test_ranked_field_failures_orders_by_count_and_marks_incomplete_history():
    result = ranked_field_failures(
        [
            {"field_failures": {"timestamp": 3, "distance_km": 1}},
            {"field_failures": {"timestamp": 1, "was_late": 2}},
        ],
        total_batches=3,
    )

    assert result["items"][:2] == [
        {"field": "timestamp", "label": "Timestamp", "failure_count": 4},
        {"field": "was_late", "label": "Observed late flag", "failure_count": 2},
    ]
    assert result["title"] == "Most-rejected field: Timestamp (history incomplete)"
    assert ranked_field_failures([])["title"] == "Field-failure data unavailable"


@pytest.mark.unit
def test_volume_over_time_rejects_naive_now():
    with pytest.raises(ValueError, match="timezone"):
        volume_over_time([], now=datetime(2026, 10, 1))