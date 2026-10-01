import math

import pytest
from sklearn.linear_model import LogisticRegression

from pipeline.train import InsufficientTrainingData, RANDOM_STATE, _evaluate_and_fit, _stratified_split


def _balanced_records(count_per_class=10):
    records = []
    for index in range(count_per_class):
        records.append({"distance_km": 2 + index / 10, "prep_minutes": 12 + index / 10, "was_late": False})
        records.append({"distance_km": 12 + index / 10, "prep_minutes": 40 + index / 10, "was_late": True})
    return records


@pytest.mark.unit
def test_stratified_split_is_deterministic_and_has_no_overlap():
    labels = [record["was_late"] for record in _balanced_records()]

    train_indices, test_indices = _stratified_split(labels)
    repeated_train_indices, repeated_test_indices = _stratified_split(labels)

    assert set(train_indices).isdisjoint(test_indices)
    assert set(train_indices) | set(test_indices) == set(range(len(labels)))
    assert (train_indices, test_indices) == (repeated_train_indices, repeated_test_indices)
    assert len(test_indices) == math.ceil(len(labels) * 0.2)
    assert RANDOM_STATE == 42


@pytest.mark.unit
def test_evaluation_uses_two_features_true_positive_class_and_all_metrics():
    model, metrics, split_sizes = _evaluate_and_fit(_balanced_records())

    assert isinstance(model, LogisticRegression)
    assert model.n_features_in_ == 2
    assert model.dashbite_feature_names_ == ("distance_km", "prep_minutes")
    assert list(model.classes_) == [False, True]
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(math.isfinite(value) for value in metrics.values())
    assert split_sizes == {"train_size": 16, "test_size": 4}


@pytest.mark.unit
def test_stratified_split_rejects_insufficient_class_coverage():
    with pytest.raises(InsufficientTrainingData):
        _stratified_split([False, False, False, True])