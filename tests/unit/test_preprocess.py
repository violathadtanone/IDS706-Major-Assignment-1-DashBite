import csv
from datetime import datetime

import pytest

from pipeline.preprocess import normalize_order_row


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {
                "order_id": "A-1",
                "timestamp": "2024-01-01T11:30:00Z",
                "distance_km": "12.5",
                "prep_minutes": "35",
                "order_value": "26.75",
                "was_late": "true",
            },
            {
                "order_id": "A-1",
                "timestamp": "2024-01-01T11:30:00Z",
                "distance_km": 12.5,
                "prep_minutes": 35,
                "order_value": 26.75,
                "was_late": True,
                "hour": 11,
                "is_peak": True,
            },
        ),
        (
            {
                "order_id": "A-2",
                "timestamp": "2024-01-01T17:00:00+00:00",
                "distance_km": "9.0",
                "prep_minutes": "22",
                "order_value": "15",
                "was_late": "FALSE",
            },
            {
                "order_id": "A-2",
                "timestamp": "2024-01-01T17:00:00+00:00",
                "distance_km": 9.0,
                "prep_minutes": 22,
                "order_value": 15.0,
                "was_late": False,
                "hour": 17,
                "is_peak": True,
            },
        ),
        (
            {
                "order_id": "A-3",
                "timestamp": "2024-01-01T18:15:00Z",
                "distance_km": "4.5",
                "prep_minutes": "12",
                "order_value": "9",
                "was_late": "1",
            },
            {
                "order_id": "A-3",
                "timestamp": "2024-01-01T18:15:00Z",
                "distance_km": 4.5,
                "prep_minutes": 12,
                "order_value": 9.0,
                "was_late": True,
                "hour": 18,
                "is_peak": True,
            },
        ),
        (
            {
                "order_id": "A-4",
                "timestamp": "2024-01-01T08:00:00Z",
                "distance_km": "3.2",
                "prep_minutes": "8",
                "order_value": "4",
                "was_late": "0",
            },
            {
                "order_id": "A-4",
                "timestamp": "2024-01-01T08:00:00Z",
                "distance_km": 3.2,
                "prep_minutes": 8,
                "order_value": 4.0,
                "was_late": False,
                "hour": 8,
                "is_peak": False,
            },
        ),
    ],
)
def test_normalize_order_row_accepts_valid_input(row, expected):
    normalized = normalize_order_row(row)

    assert normalized == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "row",
    [
        {"order_id": "", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "1", "prep_minutes": "10", "order_value": "5", "was_late": "false"},
        {"order_id": "A", "timestamp": "2024-01-01 12:00:00", "distance_km": "1", "prep_minutes": "10", "order_value": "5", "was_late": "false"},
        {"order_id": "A", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "0", "prep_minutes": "10", "order_value": "5", "was_late": "false"},
        {"order_id": "A", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "1", "prep_minutes": "-1", "order_value": "5", "was_late": "false"},
        {"order_id": "A", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "1", "prep_minutes": "10", "order_value": "-1", "was_late": "false"},
        {"order_id": "A", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "1", "prep_minutes": "10", "order_value": "5", "was_late": "maybe"},
        {"order_id": "A", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "1", "prep_minutes": "10", "order_value": "5", "was_late": "yes"},
        {"order_id": "A", "timestamp": "2024-01-01T12:00:00Z", "distance_km": "1", "prep_minutes": "10", "order_value": "5", "was_late": "2"},
    ],
)
def test_normalize_order_row_rejects_invalid_input(row):
    assert normalize_order_row(row) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("order_id", "   "),
        ("timestamp", ""),
        ("timestamp", "not-a-timestamp"),
        ("distance_km", None),
        ("distance_km", "not-a-number"),
        ("distance_km", "NaN"),
        ("distance_km", "inf"),
        ("distance_km", "-1"),
        ("prep_minutes", None),
        ("prep_minutes", "not-a-number"),
        ("prep_minutes", "NaN"),
        ("prep_minutes", "inf"),
        ("prep_minutes", "0"),
        ("order_value", None),
        ("order_value", "not-a-number"),
        ("order_value", "NaN"),
        ("order_value", "inf"),
        ("was_late", None),
    ],
)
def test_normalize_order_row_rejects_missing_non_numeric_and_non_finite_values(field, invalid_value):
    row = {
        "order_id": "A-1",
        "timestamp": "2024-01-01T12:00:00Z",
        "distance_km": "1",
        "prep_minutes": "10",
        "order_value": "5",
        "was_late": "false",
    }
    row[field] = invalid_value

    assert normalize_order_row(row) is None


@pytest.mark.unit
def test_normalize_order_row_preserves_fractional_prep_minutes():
    normalized = normalize_order_row(
        {
            "order_id": "fractional-prep",
            "timestamp": "2024-01-01T12:00:00Z",
            "distance_km": "2.5",
            "prep_minutes": "12.5",
            "order_value": "8.25",
            "was_late": "true",
        }
    )

    assert normalized is not None
    assert normalized["prep_minutes"] == 12.5
