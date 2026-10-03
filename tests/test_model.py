"""Synthetic tests for the Weighted K-Nearest Neighbors localizer."""

import numpy as np
import pandas as pd
import pytest

from src.config import RSSI_FEATURE_COLUMNS
from src.model.wknn_localizer import (
    LocalizationPrediction,
    WiFiWKNNLocalizer,
)


TARGET_COLUMNS = ["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]


def make_dataset(
    first_wap: list[float],
    longitudes: list[float],
    latitudes: list[float],
    buildings: list[int],
    floors: list[int],
) -> pd.DataFrame:
    """Create a complete 520-WAP dataset with labels chosen for a test."""
    data = pd.DataFrame(
        -110.0,
        index=range(len(first_wap)),
        columns=RSSI_FEATURE_COLUMNS,
    )
    data["WAP001"] = first_wap
    data["LONGITUDE"] = longitudes
    data["LATITUDE"] = latitudes
    data["BUILDINGID"] = buildings
    data["FLOOR"] = floors
    return data


def test_weighted_neighbors_predict_coordinates_and_labels() -> None:
    """Find the obvious two neighbors and calculate inverse-distance outputs."""
    training = make_dataset(
        first_wap=[-50, -53, -60],
        longitudes=[0, 30, 900],
        latitudes=[10, 70, 900],
        buildings=[1, 2, 9],
        floors=[0, 1, 9],
    )
    query = training.loc[[0], list(RSSI_FEATURE_COLUMNS)].copy()
    query.loc[:, "WAP001"] = -51
    model = WiFiWKNNLocalizer(k=2).fit(training)

    prediction = model.predict_single(query)

    assert isinstance(prediction, LocalizationPrediction)
    assert [neighbor.index for neighbor in prediction.neighbors] == [0, 1]
    assert [neighbor.distance for neighbor in prediction.neighbors] == [1.0, 2.0]
    assert [neighbor.weight for neighbor in prediction.neighbors] == pytest.approx(
        [2 / 3, 1 / 3]
    )
    assert prediction.x == pytest.approx(10.0)
    assert prediction.y == pytest.approx(30.0)
    assert prediction.building == 1
    assert prediction.floor == 0


def test_zero_distance_neighbors_get_all_inverse_distance_weight() -> None:
    """Handle exact training matches without dividing by zero."""
    training = make_dataset(
        first_wap=[-50, -50, -55],
        longitudes=[10, 30, 100],
        latitudes=[20, 60, 200],
        buildings=[1, 2, 3],
        floors=[0, 1, 2],
    )
    query = training.loc[[0], list(RSSI_FEATURE_COLUMNS)]
    model = WiFiWKNNLocalizer(k=3).fit(training)

    prediction = model.predict_single(query)

    assert [neighbor.distance for neighbor in prediction.neighbors] == [0.0, 0.0, 5.0]
    assert [neighbor.weight for neighbor in prediction.neighbors] == pytest.approx(
        [0.5, 0.5, 0.0]
    )
    assert prediction.x == pytest.approx(20.0)
    assert prediction.y == pytest.approx(40.0)
    assert prediction.building == 1
    assert prediction.floor == 0


def test_only_ordered_wap_features_are_used_as_model_input() -> None:
    """Ignore metadata and accept feature DataFrames in a different column order."""
    training = make_dataset(
        first_wap=[-50, -60],
        longitudes=[1, 2],
        latitudes=[3, 4],
        buildings=[0, 1],
        floors=[0, 1],
    )
    features = training.loc[:, list(reversed(RSSI_FEATURE_COLUMNS))].copy()
    features["USERID"] = [999, 999]
    features["TIMESTAMP"] = [-50, -60]
    model = WiFiWKNNLocalizer(k=1).fit(training)

    prediction = model.predict(features.iloc[[0]])

    assert len(prediction) == 1
    assert prediction[0].neighbors[0].index == 0
    assert prediction[0].x == 1
    assert prediction[0].y == 3


def test_fit_accepts_features_and_targets_separately() -> None:
    """Fit using WAP-only inputs and the four target columns as a separate frame."""
    training = make_dataset(
        first_wap=[-50, -60],
        longitudes=[1, 2],
        latitudes=[3, 4],
        buildings=[0, 1],
        floors=[0, 1],
    )
    features = training.loc[:, RSSI_FEATURE_COLUMNS]
    targets = training.loc[:, TARGET_COLUMNS]

    model = WiFiWKNNLocalizer(k=1).fit(features, targets)
    prediction = model.predict_single(features.iloc[[1]])

    assert prediction.x == 2
    assert prediction.y == 4
    assert prediction.building == 1
    assert prediction.floor == 1


def test_supports_numpy_arrays_and_uniform_weighting() -> None:
    """Return one prediction per row for a canonical NumPy feature matrix."""
    training = make_dataset(
        first_wap=[-50, -52],
        longitudes=[0, 10],
        latitudes=[0, 20],
        buildings=[0, 1],
        floors=[0, 1],
    )
    model = WiFiWKNNLocalizer(k=2, weighting_strategy="uniform").fit(training)
    queries = training.loc[:, RSSI_FEATURE_COLUMNS].to_numpy()

    predictions = model.predict(queries)

    assert len(predictions) == 2
    assert predictions[0].x == pytest.approx(5)
    assert predictions[0].y == pytest.approx(10)
    assert all(sum(item.weight for item in prediction.neighbors) == pytest.approx(1) for prediction in predictions)


def test_save_and_load_preserve_predictions(tmp_path) -> None:
    """Serialize fitted estimator state and reproduce inference after loading."""
    training = make_dataset(
        first_wap=[-50, -52],
        longitudes=[10, 20],
        latitudes=[30, 40],
        buildings=[1, 1],
        floors=[0, 1],
    )
    query = training.loc[[0], RSSI_FEATURE_COLUMNS]
    model = WiFiWKNNLocalizer(k=1, models_dir=tmp_path).fit(training)
    expected = model.predict_single(query)

    saved_path = model.save("wknn.joblib")
    restored = WiFiWKNNLocalizer.load(saved_path)

    assert saved_path.is_file()
    assert restored.predict_single(query) == expected


def test_requires_wap_features_and_target_columns() -> None:
    """Reject missing feature and target fields at fit time."""
    training = make_dataset(
        first_wap=[-50],
        longitudes=[10],
        latitudes=[30],
        buildings=[1],
        floors=[0],
    )

    with pytest.raises(ValueError, match="WAP520"):
        WiFiWKNNLocalizer(k=1).fit(training.drop(columns=["WAP520"]))
    with pytest.raises(ValueError, match="BUILDINGID"):
        WiFiWKNNLocalizer(k=1).fit(training.drop(columns=["BUILDINGID"]))


def test_rejects_invalid_k_and_too_many_neighbors() -> None:
    """Check configuration and training-size constraints."""
    with pytest.raises(ValueError, match="positive integer"):
        WiFiWKNNLocalizer(k=0)
    with pytest.raises(ValueError, match="weighting_strategy"):
        WiFiWKNNLocalizer(weighting_strategy="not-a-strategy")  # type: ignore[arg-type]

    training = make_dataset([-50], [10], [30], [1], [0])
    with pytest.raises(ValueError, match="cannot exceed"):
        WiFiWKNNLocalizer(k=2).fit(training)


def test_rejects_non_finite_fingerprints_and_targets() -> None:
    """Fail clearly for unprocessed fingerprints or invalid target values."""
    training = make_dataset([-50, -60], [10, 20], [30, 40], [1, 1], [0, 1])
    invalid_features = training.copy()
    invalid_features.loc[0, "WAP001"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        WiFiWKNNLocalizer(k=1).fit(invalid_features)

    invalid_targets = training[TARGET_COLUMNS].copy()
    invalid_targets["LONGITUDE"] = invalid_targets["LONGITUDE"].astype(float)
    invalid_targets.loc[0, "LONGITUDE"] = np.inf
    with pytest.raises(ValueError, match="LONGITUDE"):
        WiFiWKNNLocalizer(k=1).fit(
            training.loc[:, RSSI_FEATURE_COLUMNS],
            invalid_targets,
        )


def test_predict_single_checks_single_row_and_predict_requires_fit() -> None:
    """Guard prediction before fitting and multi-row single predictions."""
    features = np.full((2, len(RSSI_FEATURE_COLUMNS)), -50.0)
    with pytest.raises(RuntimeError, match="must be fitted"):
        WiFiWKNNLocalizer().predict(features)

    training = make_dataset([-50, -60], [10, 20], [30, 40], [1, 1], [0, 1])
    model = WiFiWKNNLocalizer(k=1).fit(training)
    with pytest.raises(ValueError, match="exactly one"):
        model.predict_single(features)
