"""Tests for UJIIndoorLoc schema and data-quality validation."""

import pandas as pd
import pytest

from src.config import RSSI_FEATURE_COLUMNS
from src.data.validator import TARGET_COLUMNS, validate_dataframe


def make_synthetic_frame() -> pd.DataFrame:
    """Build a small complete UJIIndoorLoc-shaped test DataFrame."""
    data = {
        column: [-50, 100]
        for column in RSSI_FEATURE_COLUMNS
    }
    data.update(
        {
            "LONGITUDE": [-7000.0, -6990.0],
            "LATITUDE": [4_864_800.0, 4_864_810.0],
            "FLOOR": [1, 2],
            "BUILDINGID": [0, 2],
            "SPACEID": [1, 2],
            "RELATIVEPOSITION": [1, 2],
            "USERID": [1, 2],
            "PHONEID": [1, 2],
            "TIMESTAMP": [1_000, 2_000],
        }
    )
    return pd.DataFrame(data, columns=[*RSSI_FEATURE_COLUMNS, *TARGET_COLUMNS])


def test_valid_dataframe_reports_schema_and_observed_ranges() -> None:
    """Accept the expected schema and expose useful summary details."""
    frame = make_synthetic_frame()

    report = validate_dataframe(frame, dataset_name="synthetic")

    assert report.is_valid
    assert report.shape == (2, 529)
    assert len(report.rssi_columns_present) == 520
    assert report.target_columns_present == TARGET_COLUMNS
    assert report.rssi_value_range == (-50.0, -50.0)
    assert report.rssi_missing_sentinel_values == 520
    assert report.coordinate_ranges["LONGITUDE"] == (-7000.0, -6990.0)
    assert report.category_values["BUILDINGID"] == (0, 2)
    assert "Status: PASS" in report.to_text()
    assert "source units; no CRS bounds assumed" in report.to_text()


def test_missing_wap_and_target_columns_are_errors() -> None:
    """Flag absent required fingerprint and target columns."""
    frame = make_synthetic_frame().drop(
        columns=["WAP001", "WAP520", "BUILDINGID"]
    )

    report = validate_dataframe(frame)

    assert not report.is_valid
    assert report.missing_rssi_columns == ("WAP001", "WAP520")
    assert report.missing_target_columns == ("BUILDINGID",)


def test_invalid_rssi_and_non_numeric_dtype_are_errors() -> None:
    """Reject out-of-range RSSI and non-numeric feature columns."""
    frame = make_synthetic_frame()
    frame.loc[0, "WAP001"] = -120
    frame["WAP002"] = ["not-a-number", "-50"]

    report = validate_dataframe(frame)

    assert not report.is_valid
    assert report.invalid_rssi_values == 1
    assert any("RSSI values outside" in error for error in report.errors)
    assert any("WAP002" in error and "numeric data type" in error for error in report.errors)


def test_missing_values_and_duplicate_rows_are_reported_as_warnings() -> None:
    """Report null cells and exact duplicate rows without modifying the frame."""
    frame = make_synthetic_frame()
    frame.loc[0, "WAP001"] = None
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)

    report = validate_dataframe(frame)

    assert report.is_valid
    assert report.missing_values == {"WAP001": 1}
    assert report.duplicate_rows == 1
    assert any("missing cells" in warning for warning in report.warnings)
    assert any("duplicate rows" in warning for warning in report.warnings)


def test_coordinate_bounds_and_building_floor_values_are_checked() -> None:
    """Apply optional coordinate bounds and reject invalid building/floor IDs."""
    frame = make_synthetic_frame()
    frame.loc[0, "BUILDINGID"] = -1
    frame["FLOOR"] = frame["FLOOR"].astype(float)
    frame.loc[1, "FLOOR"] = 2.5

    report = validate_dataframe(
        frame,
        coordinate_bounds={"LONGITUDE": (-6995.0, -6980.0)},
    )

    assert not report.is_valid
    assert any("LONGITUDE" in error and "configured bounds" in error for error in report.errors)
    assert any("BUILDINGID" in error and "negative" in error for error in report.errors)
    assert any("FLOOR" in error and "non-integer" in error for error in report.errors)


def test_duplicate_column_names_are_reported_without_crashing() -> None:
    """Handle a malformed schema with ambiguous repeated columns safely."""
    frame = make_synthetic_frame()
    frame["WAP001_copy"] = frame["WAP001"]
    frame = frame.rename(columns={"WAP001_copy": "WAP001"})

    report = validate_dataframe(frame)

    assert not report.is_valid
    assert report.duplicate_column_names == ("WAP001",)


def test_unknown_coordinate_bounds_are_rejected() -> None:
    """Reject bounds for columns that are not coordinate targets."""
    with pytest.raises(ValueError, match="LONGITUDE and LATITUDE"):
        validate_dataframe(make_synthetic_frame(), coordinate_bounds={"FLOOR": (0, 4)})
