"""Tests for Streamlit inference input validation and label isolation."""

import pandas as pd
import pytest

from app.app import GROUND_TRUTH_COLUMNS, _get_ground_truth, validate_upload
from src.config import RSSI_FEATURE_COLUMNS


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
