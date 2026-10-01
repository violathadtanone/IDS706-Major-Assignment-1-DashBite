import math
from datetime import datetime, timedelta, timezone

WINDOW_MINUTES = 60
FIELD_LABELS = {
    "order_id": "Order ID",
    "timestamp": "Timestamp",
    "distance_km": "Distance",
    "prep_minutes": "Preparation time",
    "order_value": "Order value",
    "was_late": "Observed late flag",
}
QUALITY_FIELDS = tuple(FIELD_LABELS)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _window(now: datetime | None, window_minutes: int) -> tuple[datetime, datetime, datetime]:
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must include a timezone")
    current = current.astimezone(timezone.utc)
    last_minute = current.replace(second=0, microsecond=0)
    first_minute = last_minute - timedelta(minutes=window_minutes - 1)
    return current, first_minute, last_minute


def _recent_timestamp(value: object, current: datetime, first_minute: datetime) -> datetime | None:
    parsed = _parse_timestamp(value)
    if parsed is None or parsed < first_minute or parsed > current:
        return None
    return parsed.replace(second=0, microsecond=0)


def sample_volume(
    feature_rows: list[dict[str, object]],
    *,
    now: datetime | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> int:
    current, first_minute, _ = _window(now, window_minutes)
    return sum(
        _recent_timestamp(row.get("timestamp"), current, first_minute) is not None
        for row in feature_rows
    )


def volume_over_time(
    feature_rows: list[dict[str, object]],
    *,
    now: datetime | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> dict[str, object]:
    current, first_minute, last_minute = _window(now, window_minutes)
    counts = {first_minute + timedelta(minutes=offset): 0 for offset in range(window_minutes)}
    for row in feature_rows:
        minute = _recent_timestamp(row.get("timestamp"), current, first_minute)
        if minute in counts:
            counts[minute] += 1
    points = [
        {"minute": minute.isoformat(), "sample_count": count}
        for minute, count in counts.items()
    ]
    first_half = sum(point["sample_count"] for point in points[: window_minutes // 2])
    second_half = sum(point["sample_count"] for point in points[window_minutes // 2 :])
    title = _volume_title(first_half, second_half)
    return {"points": points, "title": title, "window_start": first_minute.isoformat(), "window_end": last_minute.isoformat()}


def _volume_title(first_half: int, second_half: int) -> str:
    if first_half < 5 or second_half < 5:
        return "Not enough recent orders to determine a volume trend"
    mean_volume = (first_half + second_half) / 2
    change = second_half - first_half
    if abs(change) <= mean_volume * 0.1:
        return "Order volume is steady"
    return "More orders in the last 30 minutes" if change > 0 else "Fewer orders in the last 30 minutes"


def score_summary(
    feature_rows: list[dict[str, object]], prediction_rows: list[dict[str, object]]
) -> dict[str, int | float | None]:
    feature_ids = {str(row.get("order_id", "")).strip() for row in feature_rows}
    matched_rows = [
        row for row in prediction_rows if str(row.get("order_id", "")).strip() in feature_ids
    ]
    valid_probabilities = []
    for row in prediction_rows:
        try:
            probability = float(row.get("late_probability", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(probability) and 0 <= probability <= 1:
            valid_probabilities.append(probability)
    flagged_count = sum(_parse_bool(row.get("predicted_late")) is True for row in prediction_rows)
    return {
        "scored_count": len(prediction_rows),
        "matched_count": len(matched_rows),
        "unmatched_count": len(prediction_rows) - len(matched_rows),
        "flagged_count": flagged_count,
        "mean_late_probability": (
            sum(valid_probabilities) / len(valid_probabilities) if valid_probabilities else None
        ),
    }


def late_flag_rate_over_time(
    feature_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    *,
    now: datetime | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> dict[str, object]:
    current, first_minute, last_minute = _window(now, window_minutes)
    timestamps = {}
    for row in feature_rows:
        order_id = str(row.get("order_id", "")).strip()
        if order_id:
            timestamps[order_id] = row.get("timestamp")
    counts = {
        first_minute + timedelta(minutes=offset): {"matched_count": 0, "flagged_count": 0}
        for offset in range(window_minutes)
    }
    for row in prediction_rows:
        order_id = str(row.get("order_id", "")).strip()
        minute = _recent_timestamp(timestamps.get(order_id), current, first_minute)
        if minute not in counts:
            continue
        predicted_late = _parse_bool(row.get("predicted_late"))
        if predicted_late is None:
            continue
        counts[minute]["matched_count"] += 1
        counts[minute]["flagged_count"] += predicted_late is True

    points = []
    for minute, values in counts.items():
        matched_count = values["matched_count"]
        points.append(
            {
                "minute": minute.isoformat(),
                "matched_count": matched_count,
                "late_rate_pct": (
                    100 * values["flagged_count"] / matched_count if matched_count else None
                ),
            }
        )
    title = _late_flag_title(points, window_minutes)
    return {"points": points, "title": title, "window_start": first_minute.isoformat(), "window_end": last_minute.isoformat()}


def _late_flag_title(points: list[dict[str, object]], window_minutes: int) -> str:
    split_at = window_minutes // 2
    first_points = points[:split_at]
    second_points = points[split_at:]
    first_matched = sum(point["matched_count"] for point in first_points)
    second_matched = sum(point["matched_count"] for point in second_points)
    if first_matched < 10 or second_matched < 10:
        return "Not enough matched predictions to determine a late-flag trend"
    first_flagged = sum(
        (point["late_rate_pct"] or 0) * point["matched_count"] / 100 for point in first_points
    )
    second_flagged = sum(
        (point["late_rate_pct"] or 0) * point["matched_count"] / 100 for point in second_points
    )
    first_pct = 100 * first_flagged / first_matched
    second_pct = 100 * second_flagged / second_matched
    difference = second_pct - first_pct
    if difference > 5:
        return "Model is flagging more orders late"
    if difference < -5:
        return "Model is flagging fewer orders late"
    return "Late-flag rate is steady"


def drop_rate_summary(
    quality_records: list[dict[str, object]], *, total_batches: int | None = None
) -> dict[str, int | float | bool | None]:
    input_rows = sum(int(record.get("input_rows", 0)) for record in quality_records)
    accepted_rows = sum(int(record.get("accepted_rows", 0)) for record in quality_records)
    rejected_rows = sum(int(record.get("rejected_rows", 0)) for record in quality_records)
    available_batches = len(quality_records)
    return {
        "input_rows": input_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "drop_rate_pct": 100 * rejected_rows / input_rows if input_rows else None,
        "available_batches": available_batches,
        "total_batches": total_batches,
        "history_complete": total_batches is not None and available_batches >= total_batches,
    }


def ranked_field_failures(
    quality_records: list[dict[str, object]], *, total_batches: int | None = None
) -> dict[str, object]:
    available_batches = len(quality_records)
    history_complete = total_batches is not None and available_batches >= total_batches
    if not quality_records:
        return {
            "items": [],
            "title": "Field-failure data unavailable",
            "available_batches": 0,
            "total_batches": total_batches,
            "history_complete": False,
        }

    counts = {field: 0 for field in QUALITY_FIELDS}
    for record in quality_records:
        failures = record.get("field_failures", {})
        if not isinstance(failures, dict):
            continue
        for field, count in failures.items():
            if field in counts:
                counts[field] += int(count)
    items = [
        {"field": field, "label": FIELD_LABELS[field], "failure_count": count}
        for field, count in sorted(counts.items(), key=lambda item: (-item[1], FIELD_LABELS[item[0]]))
        if count > 0
    ]
    if not items:
        title = "No field failures were recorded"
    else:
        highest = items[0]["failure_count"]
        leaders = [item["label"] for item in items if item["failure_count"] == highest]
        if len(leaders) > 1:
            title = f"Most-rejected fields are tied: {', '.join(leaders)}"
        else:
            title = f"Most-rejected field: {leaders[0]}"
    if total_batches is not None and available_batches < total_batches:
        title += " (history incomplete)"
    return {
        "items": items,
        "title": title,
        "available_batches": available_batches,
        "total_batches": total_batches,
        "history_complete": history_complete,
    }