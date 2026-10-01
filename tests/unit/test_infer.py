import csv

import pytest

from pipeline.infer import PREDICTION_COLUMNS, _prediction_path, _write_predictions


class KnownProbabilityModel:
    classes_ = [False, True]
    n_features_in_ = 2

    def __init__(self):
        self.seen_inputs = None

    def predict_proba(self, inputs):
        self.seen_inputs = inputs
        return [[0.5, 0.5], [0.8, 0.2]]


@pytest.mark.unit
def test_scoring_uses_feature_order_true_probability_and_half_threshold(tmp_path):
    feature_path = tmp_path / "orders_sample_features.csv"
    feature_path.write_text(
        "order_id,distance_km,prep_minutes\n"
        "order-1,12.5,35\n"
        "order-2,7.0,18\n",
        encoding="utf-8",
    )
    model = KnownProbabilityModel()

    output_path = _write_predictions(
        feature_path,
        tmp_path / "predictions",
        model,
        "v0003",
        ["distance_km", "prep_minutes"],
    )

    assert output_path.name == "orders_sample_predictions.csv"
    assert model.seen_inputs == [[12.5, 35.0], [7.0, 18.0]]
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == list(PREDICTION_COLUMNS)
    assert rows == [
        {
            "order_id": "order-1",
            "late_probability": "0.5",
            "predicted_late": "True",
            "checkpoint_id": "v0003",
        },
        {
            "order_id": "order-2",
            "late_probability": "0.2",
            "predicted_late": "False",
            "checkpoint_id": "v0003",
        },
    ]


@pytest.mark.unit
def test_prediction_path_maps_feature_batch_identity(tmp_path):
    feature_path = tmp_path / "orders_2024-01-01_features.csv"

    assert _prediction_path(feature_path, tmp_path / "predictions") == (
        tmp_path / "predictions" / "orders_2024-01-01_predictions.csv"
    )