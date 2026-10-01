from datetime import datetime, timedelta, timezone

import pytest

from pipeline.pulse import late_flag_rate_over_time, ranked_field_failures, volume_over_time

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _minute_records(start, count, flagged_count=0, prefix="order"):
    features = []
    predictions = []
    for index in range(count):
        order_id = f"{prefix}-{index}"
        minute = start + timedelta(minutes=index % 30)
        features.append({"order_id": order_id, "timestamp": minute.isoformat()})
        predictions.append({"order_id": order_id, "predicted_late": index < flagged_count})
    return features, predictions


@pytest.mark.regression
def test_volume_trend_titles_cover_insufficient_more_fewer_steady_and_output_schema():
    insufficient = volume_over_time([], now=NOW)
    assert insufficient["title"] == "Not enough recent orders to determine a volume trend"
    assert set(insufficient) == {"points", "title", "window_start", "window_end"}

    first_half_start = NOW - timedelta(minutes=59)
    fewer_rows = [
        {"timestamp": (first_half_start + timedelta(minutes=index % 30)).isoformat()}
        for index in range(20)
    ] + [
        {"timestamp": (first_half_start + timedelta(minutes=30 + index % 30)).isoformat()}
        for index in range(10)
    ]
    more_rows = [
        {"timestamp": (first_half_start + timedelta(minutes=index % 30)).isoformat()}
        for index in range(10)
    ] + [
        {"timestamp": (first_half_start + timedelta(minutes=30 + index % 30)).isoformat()}
        for index in range(20)
    ]
    assert volume_over_time(fewer_rows, now=NOW)["title"] == "Fewer orders in the last 30 minutes"
    assert volume_over_time(more_rows, now=NOW)["title"] == "More orders in the last 30 minutes"

    steady_rows = [
        {"timestamp": (first_half_start + timedelta(minutes=index % 30)).isoformat()}
        for index in range(10)
    ] + [
        {"timestamp": (first_half_start + timedelta(minutes=30 + index % 30)).isoformat()}
        for index in range(11)
    ]
    assert volume_over_time(steady_rows, now=NOW)["title"] == "Order volume is steady"


@pytest.mark.regression
@pytest.mark.parametrize(
    ("first_count", "second_count", "expected"),
    [
        (4, 10, "Not enough recent orders to determine a volume trend"),
        (5, 4, "Not enough recent orders to determine a volume trend"),
        (5, 5, "Order volume is steady"),
        (20, 22, "Order volume is steady"),
        (20, 23, "More orders in the last 30 minutes"),
    ],
)
def test_volume_trend_titles_respect_five_sample_minimum_and_ten_percent_band(
    first_count, second_count, expected
):
    first_minute = NOW.replace(second=0) - timedelta(minutes=59)
    rows = [
        {"timestamp": (first_minute + timedelta(minutes=index % 30)).isoformat()}
        for index in range(first_count)
    ] + [
        {"timestamp": (first_minute + timedelta(minutes=30 + index % 30)).isoformat()}
        for index in range(second_count)
    ]

    assert volume_over_time(rows, now=NOW)["title"] == expected


@pytest.mark.regression
@pytest.mark.parametrize(
    ("count_per_half", "first_flagged", "second_flagged", "expected"),
    [
        (10, 0, 10, "Model is flagging more orders late"),
        (10, 10, 0, "Model is flagging fewer orders late"),
        (20, 8, 9, "Late-flag rate is steady"),
        (20, 8, 10, "Model is flagging more orders late"),
        (20, 10, 8, "Model is flagging fewer orders late"),
        (20, 8, 8, "Late-flag rate is steady"),
    ],
)
def test_late_flag_trend_titles_respect_minimums_and_percentage_point_threshold(
    count_per_half, first_flagged, second_flagged, expected
):
    first_features, first_predictions = _minute_records(
        NOW - timedelta(minutes=59), count_per_half, first_flagged, "first"
    )
    second_features, second_predictions = _minute_records(
        NOW - timedelta(minutes=29), count_per_half, second_flagged, "second"
    )

    result = late_flag_rate_over_time(
        first_features + second_features,
        first_predictions + second_predictions,
        now=NOW,
    )

    assert result["title"] == expected
    assert set(result) == {"points", "title", "window_start", "window_end"}
    assert all(set(point) == {"minute", "matched_count", "late_rate_pct"} for point in result["points"])


@pytest.mark.regression
def test_field_failure_titles_cover_unique_tie_zero_unavailable_and_incomplete():
    assert ranked_field_failures(
        [{"field_failures": {"timestamp": 4, "was_late": 1}}], total_batches=1
    )["title"] == "Most-rejected field: Timestamp"
    assert ranked_field_failures(
        [{"field_failures": {"timestamp": 4, "was_late": 4}}], total_batches=1
    )["title"].startswith("Most-rejected fields are tied:")
    assert ranked_field_failures(
        [{"field_failures": {"timestamp": 0}}], total_batches=1
    )["title"] == "No field failures were recorded"
    assert ranked_field_failures([])["title"] == "Field-failure data unavailable"
    assert ranked_field_failures(
        [{"field_failures": {"timestamp": 2}}], total_batches=2
    )["title"].endswith("(history incomplete)")