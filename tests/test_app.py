"""Tests for Streamlit inference input validation and label isolation."""

import numpy as np
import pandas as pd
import pytest

from app.app import (
    BuildingFloorLocations,
    GROUND_TRUTH_COLUMNS,
    InferenceResult,
    _detected_wap_count,
    _get_ground_truth,
    _local_coordinates,
    _make_location_chart,
    build_manual_fingerprint,
    validate_upload,
)
from src.config import RSSI_FEATURE_COLUMNS
from src.model.wknn_localizer import LocalizationPrediction, NeighborInfo


def make_upload() -> pd.DataFrame:
    """Create a small complete fingerprint table with optional labels."""
    fingerprint = {column: 100 for column in RSSI_FEATURE_COLUMNS}
    fingerprint["WAP001"] = -62
    fingerprint.update(
        {
            "LONGITUDE": -7541.0,
            "LATITUDE": 4864920.0,
            "BUILDINGID": 1,
            "FLOOR": 2,
            "USERID": 42,
        }
    )
    return pd.DataFrame([fingerprint])


def test_upload_inference_features_include_only_canonical_waps() -> None:
    """Ignore labels/metadata and maintain canonical feature ordering."""
    features = validate_upload(make_upload())

    assert tuple(features.columns) == RSSI_FEATURE_COLUMNS
    assert not set(GROUND_TRUTH_COLUMNS).intersection(features.columns)
    assert "USERID" not in features.columns


def test_upload_requires_all_wap_columns() -> None:
    """Report incomplete fingerprints clearly."""
    upload = make_upload().drop(columns=["WAP520"])

    with pytest.raises(ValueError, match="WAP520"):
        validate_upload(upload)


def test_upload_rejects_empty_and_duplicate_columns() -> None:
    """Reject unusable CSV table shapes before model inference."""
    with pytest.raises(ValueError, match="no samples"):
        validate_upload(make_upload().iloc[0:0])

    duplicate_columns = list(make_upload().columns)
    duplicate_columns[-1] = duplicate_columns[-2]
    duplicate_upload = make_upload()
    duplicate_upload.columns = duplicate_columns
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_upload(duplicate_upload)


def test_ground_truth_is_optional_and_kept_separate() -> None:
    """Only return actual labels when every target is present and numeric."""
    upload = make_upload()
    features = validate_upload(upload)
    actual = _get_ground_truth(upload.iloc[0])

    assert actual == {
        "x": -7541.0,
        "y": 4864920.0,
        "building": 1,
        "floor": 2,
    }
    assert tuple(features.columns) == RSSI_FEATURE_COLUMNS
    assert _get_ground_truth(upload.drop(columns=["FLOOR"]).iloc[0]) is None


def test_detected_wap_count_excludes_unavailable_and_missing_values() -> None:
    """Count only actual signal RSSI readings."""
    fingerprint = {column: 100 for column in RSSI_FEATURE_COLUMNS}
    fingerprint.update({"WAP001": -54, "WAP007": -67, "WAP021": -72})
    fingerprint["WAP030"] = None

    assert _detected_wap_count(pd.DataFrame([fingerprint])) == 3


def test_manual_fingerprint_fills_only_unlisted_waps_with_sentinel() -> None:
    """Build full ordered model input from a short list of detected access points."""
    entries = pd.DataFrame(
        {
            "WAP ID": ["WAP001", "WAP007", "WAP021"],
            "RSSI": [-54, -67, -72],
        }
    )

    fingerprint = build_manual_fingerprint(entries)

    assert tuple(fingerprint.columns) == RSSI_FEATURE_COLUMNS
    assert fingerprint.loc[0, ["WAP001", "WAP007", "WAP021"]].tolist() == [
        -54.0,
        -67.0,
        -72.0,
    ]
    assert fingerprint.loc[0, "WAP002"] == 100
    assert _detected_wap_count(fingerprint) == 3


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (pd.DataFrame({"WAP ID": [None], "RSSI": [None]}), "at least one"),
        (
            pd.DataFrame({"WAP ID": ["WAP001", "WAP001"], "RSSI": [-54, -60]}),
            "listed more than once",
        ),
        (
            pd.DataFrame({"WAP ID": ["WAP001"], "RSSI": [100]}),
            "100 means not detected",
        ),
        (
            pd.DataFrame({"WAP ID": ["WAP999"], "RSSI": [-54]}),
            "not a valid",
        ),
    ],
)
def test_manual_fingerprint_rejects_invalid_rows(
    entries: pd.DataFrame,
    message: str,
) -> None:
    """Reject empty, duplicate, out-of-range, and unknown manual AP entries."""
    with pytest.raises(ValueError, match=message):
        build_manual_fingerprint(entries)


def test_local_coordinates_are_offsets_from_building_floor_origin() -> None:
    """Convert large projected values to useful local visualization offsets."""
    locations = BuildingFloorLocations(
        origin_x=-7500.0,
        origin_y=4_864_800.0,
        local_x=np.array([0.0, 100.0]),
        local_y=np.array([0.0, 80.0]),
    )

    assert _local_coordinates(-7450.0, 4_864_840.0, locations) == (50.0, 40.0)


def test_plot_uses_local_indoor_axes_and_building_floor_title() -> None:
    """Avoid projected-coordinate axes and label the relevant floor."""
    neighbor = NeighborInfo(
        index=12,
        distance=2.0,
        weight=1.0,
        building=1,
        floor=3,
        x=-7450.0,
        y=4_864_840.0,
    )
    prediction = LocalizationPrediction(
        building=1,
        floor=3,
        x=-7450.0,
        y=4_864_840.0,
        neighbors=(neighbor,),
    )
    result = InferenceResult(
        prediction=prediction,
        latitude=39.99,
        longitude=-0.06,
        detected_wap_count=1,
    )
    locations = BuildingFloorLocations(
        origin_x=-7500.0,
        origin_y=4_864_800.0,
        local_x=np.array([0.0, 100.0]),
        local_y=np.array([0.0, 80.0]),
    )
    actual = {"building": 1, "floor": 3, "x": -7460.0, "y": 4_864_830.0}

    figure = _make_location_chart(
        result,
        locations,
        actual,
        (40.0, 30.0),
        12.5,
    )

    axis = figure.axes[0]
    assert axis.get_title() == "Building 1 — Floor 3"
    assert axis.get_xlabel() == "Indoor X (meters)"
    assert axis.get_ylabel() == "Indoor Y (meters)"
    assert axis.get_xlim()[0] < 40.0 < axis.get_xlim()[1]
    assert axis.get_ylim()[0] < 30.0 < axis.get_ylim()[1]
    assert axis.get_xlim()[1] < 1000
    assert axis.get_ylim()[1] < 1000
    import matplotlib.pyplot as plt

    plt.close(figure)
