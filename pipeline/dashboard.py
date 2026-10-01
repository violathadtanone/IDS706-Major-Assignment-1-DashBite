import csv
import json
import math
from datetime import datetime
from pathlib import Path

from pipeline.paths import resolve_data_dir
from pipeline.pulse import (
    drop_rate_summary,
    late_flag_rate_over_time,
    ranked_field_failures,
    sample_volume,
    score_summary,
    volume_over_time,
)

FEATURE_SUFFIX = "_features.csv"
PREDICTION_SUFFIX = "_predictions.csv"


def _read_csv_rows(directory: Path, pattern: str) -> list[dict[str, str]]:
    rows = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob(pattern)):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        except (OSError, csv.Error) as exc:
            print(f"Dashboard could not read {path.name}: {exc}")
    return rows


def _quality_sidecar_is_valid(payload: object, feature_name: str) -> bool:
    if not isinstance(payload, dict) or payload.get("feature_file") != feature_name:
        return False
    input_rows = payload.get("input_rows")
    accepted_rows = payload.get("accepted_rows")
    rejected_rows = payload.get("rejected_rows")
    field_failures = payload.get("field_failures")
    if (
        not isinstance(input_rows, int)
        or not isinstance(accepted_rows, int)
        or not isinstance(rejected_rows, int)
        or min(input_rows, accepted_rows, rejected_rows) < 0
        or accepted_rows + rejected_rows != input_rows
        or not isinstance(field_failures, dict)
        or not all(isinstance(value, int) and value >= 0 for value in field_failures.values())
    ):
        return False
    drop_rate = payload.get("drop_rate")
    expected = rejected_rows / input_rows if input_rows else None
    if expected is None:
        return drop_rate is None
    return isinstance(drop_rate, (int, float)) and math.isclose(drop_rate, expected)


def load_dashboard_data(
    *,
    features_dir: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    quality_dir: str | Path | None = None,
) -> dict[str, object]:
    """Read existing dashboard artifacts without creating or changing files."""
    features_dir = Path(features_dir) if features_dir is not None else resolve_data_dir("features")
    predictions_dir = (
        Path(predictions_dir) if predictions_dir is not None else resolve_data_dir("predictions")
    )
    quality_dir = Path(quality_dir) if quality_dir is not None else resolve_data_dir("quality")

    feature_paths = sorted(features_dir.glob(f"*{FEATURE_SUFFIX}")) if features_dir.is_dir() else []
    feature_rows = _read_csv_rows(features_dir, f"*{FEATURE_SUFFIX}")
    prediction_rows = _read_csv_rows(predictions_dir, f"*{PREDICTION_SUFFIX}")
    quality_records = []
    unavailable_quality_batches = []
    for feature_path in feature_paths:
        quality_name = feature_path.name[: -len(FEATURE_SUFFIX)] + "_quality.json"
        quality_path = quality_dir / quality_name
        if not quality_path.is_file():
            unavailable_quality_batches.append(feature_path.name)
            continue
        try:
            with quality_path.open(encoding="utf-8") as handle:
                quality = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Dashboard could not read {quality_path.name}: {exc}")
            unavailable_quality_batches.append(feature_path.name)
            continue
        if not _quality_sidecar_is_valid(quality, feature_path.name):
            unavailable_quality_batches.append(feature_path.name)
            continue
        quality_records.append(quality)

    return {
        "feature_rows": feature_rows,
        "prediction_rows": prediction_rows,
        "quality_records": quality_records,
        "total_feature_batches": len(feature_paths),
        "unavailable_quality_batches": unavailable_quality_batches,
    }


def build_dashboard_view(data: dict[str, object], *, now: datetime | None = None) -> dict[str, object]:
    feature_rows = data["feature_rows"]
    prediction_rows = data["prediction_rows"]
    quality_records = data["quality_records"]
    total_batches = data["total_feature_batches"]
    volume = volume_over_time(feature_rows, now=now)
    late_flags = late_flag_rate_over_time(feature_rows, prediction_rows, now=now)
    scores = score_summary(feature_rows, prediction_rows)
    drop_rate = drop_rate_summary(quality_records, total_batches=total_batches)
    failures = ranked_field_failures(quality_records, total_batches=total_batches)
    return {
        "kpis": {
            "valid_samples_recent": sample_volume(feature_rows, now=now),
            "drop_rate_pct": drop_rate["drop_rate_pct"],
            "predictions_scored": scores["scored_count"],
        },
        "volume": volume,
        "late_flags": late_flags,
        "scores": scores,
        "drop_rate": drop_rate,
        "field_failures": failures,
    }


def _render_dashboard(st, go, view: dict[str, object]) -> None:
    kpis = view["kpis"]
    metrics = st.columns(3)
    metrics[0].metric("Valid samples (60 min)", kpis["valid_samples_recent"])
    drop_rate = kpis["drop_rate_pct"]
    metrics[1].metric("Data drop rate", f"{drop_rate:.1f}%" if drop_rate is not None else "Unavailable")
    metrics[2].metric("Predictions scored", kpis["predictions_scored"])

    if view["scores"]["unmatched_count"]:
        st.info(f"{view['scores']['unmatched_count']} prediction(s) have no matching feature order.")
    if not view["drop_rate"]["history_complete"]:
        if view["drop_rate"]["available_batches"]:
            st.info(
                "Quality history is incomplete; batches without quality sidecars are excluded from quality rates."
            )
        elif view["drop_rate"]["total_batches"]:
            st.info("Quality history is unavailable; no usable quality sidecars were found.")

    volume = view["volume"]
    volume_figure = go.Figure(
        go.Scatter(
            x=[point["minute"] for point in volume["points"]],
            y=[point["sample_count"] for point in volume["points"]],
            mode="lines",
            line={"color": "#087f73", "width": 2, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(8, 127, 115, 0.12)",
            name="Orders",
        )
    )
    volume_figure.update_layout(
        title=volume["title"],
        xaxis_title="Minute (UTC)",
        yaxis_title="Valid orders",
        showlegend=False,
        margin={"l": 18, "r": 18, "t": 58, "b": 18},
        height=340,
    )
    st.plotly_chart(volume_figure, width="stretch", key="pulse-volume")

    lower_charts = st.columns([1.25, 1])
    late_flags = view["late_flags"]
    late_figure = go.Figure(
        go.Scatter(
            x=[point["minute"] for point in late_flags["points"]],
            y=[point["late_rate_pct"] for point in late_flags["points"]],
            mode="lines",
            line={"color": "#d66b35", "width": 2, "shape": "hv"},
            connectgaps=False,
            name="Predicted late",
        )
    )
    late_figure.update_layout(
        title=late_flags["title"],
        xaxis_title="Minute (UTC)",
        yaxis_title="Predicted late (%)",
        yaxis={"range": [0, 100], "ticksuffix": "%"},
        showlegend=False,
        margin={"l": 18, "r": 18, "t": 58, "b": 18},
        height=320,
    )
    lower_charts[0].plotly_chart(late_figure, width="stretch", key="pulse-late-rate")

    failures = view["field_failures"]
    if failures["items"]:
        items = failures["items"]
        failure_figure = go.Figure(
            go.Bar(
                x=[item["failure_count"] for item in items],
                y=[item["label"] for item in items],
                orientation="h",
                marker_color="#bd4b4b",
                name="Rejected rows",
            )
        )
        failure_figure.update_layout(
            title=failures["title"],
            xaxis_title="Rows rejected for field",
            yaxis_title="Field",
            yaxis={
                "categoryorder": "array",
                "categoryarray": [item["label"] for item in items],
                "autorange": "reversed",
            },
            showlegend=False,
            margin={"l": 18, "r": 18, "t": 58, "b": 18},
            height=320,
        )
        lower_charts[1].plotly_chart(failure_figure, width="stretch", key="pulse-field-failures")
    else:
        lower_charts[1].info(failures["title"])


def main() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Model Pulse", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f3f7f5; color: #152c2a; }
        h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #123d37; }
        [data-testid="stMetric"] { background: #ffffff; border-left: 3px solid #087f73; padding: 12px 16px; }
        [data-testid="stMetricLabel"] { color: #526a65; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Model Pulse")
    st.caption("Live pipeline health · file-backed view · refreshes every 5 seconds")

    @st.fragment(run_every="5s")
    def render_live_view():
        data = load_dashboard_data()
        view = build_dashboard_view(data)
        _render_dashboard(st, go, view)

    render_live_view()


if __name__ == "__main__":
    main()