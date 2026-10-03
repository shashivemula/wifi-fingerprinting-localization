"""Tests for reusable WAP fingerprint preprocessing."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.config import RSSI_FEATURE_COLUMNS
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor


def make_fingerprint_frame() -> pd.DataFrame:
    """Create small samples with every required WAP feature."""
    frame = pd.DataFrame(
        -50,
        index=range(2),
        columns=RSSI_FEATURE_COLUMNS,
        dtype=np.int64,
    )
    frame.loc[0, "WAP001"] = 100
    frame.loc[1, "WAP002"] = -75
    return frame


def test_selects_only_wap_features_and_ignores_targets() -> None:
    """Output includes every WAP feature but excludes target/metadata columns."""
    frame = make_fingerprint_frame()
    frame["LONGITUDE"] = [-7000.0, -6990.0]
    frame["BUILDINGID"] = [0, 2]

    result = RSSIPreprocessor().fit_transform(frame)

    assert tuple(result.columns) == RSSI_FEATURE_COLUMNS
    assert result.shape == (2, 520)
    assert "LONGITUDE" not in result
    assert "BUILDINGID" not in result


def test_replaces_uji_unavailable_values_and_nulls_configurably() -> None:
    """Replace the sentinel and nulls using the configured replacement value."""
    frame = make_fingerprint_frame()
    frame.loc[1, "WAP003"] = np.nan

    result = RSSIPreprocessor(replacement_value=-120).fit_transform(frame)

    assert result.loc[0, "WAP001"] == -120
    assert result.loc[1, "WAP003"] == -120
    assert result.loc[0, "WAP002"] == -50


def test_rejects_non_numeric_and_out_of_range_rssi_values() -> None:
    """Raise clear errors for values that cannot represent valid RSSI."""
    frame = make_fingerprint_frame()
    frame["WAP001"] = frame["WAP001"].astype(object)
    frame.loc[0, "WAP001"] = "not-rssi"

    with pytest.raises(ValueError, match="non-numeric"):
        RSSIPreprocessor().fit_transform(frame)

    frame = make_fingerprint_frame()
    frame.loc[0, "WAP001"] = -105
    preprocessor = RSSIPreprocessor().fit(frame)
    with pytest.raises(ValueError, match="invalid RSSI"):
        preprocessor.transform(frame)


def test_rejects_missing_wap_column() -> None:
    """Identify required WAP feature omissions."""
    frame = make_fingerprint_frame().drop(columns=["WAP520"])

    with pytest.raises(ValueError, match="WAP520"):
        RSSIPreprocessor().fit(frame)


def test_preserves_canonical_order_for_reordered_input() -> None:
    """Return features in WAP001-to-WAP520 order regardless of input order."""
    frame = make_fingerprint_frame().loc[:, list(reversed(RSSI_FEATURE_COLUMNS))]
    frame["FLOOR"] = [0, 1]

    result = RSSIPreprocessor().fit_transform(frame)

    assert tuple(result.columns) == RSSI_FEATURE_COLUMNS
    assert result.loc[0, "WAP001"] == -110
    assert result.loc[0, "WAP520"] == -50


def test_transform_matches_fit_transform_and_survives_serialization() -> None:
    """Reuse a serialized fitted instance for consistent inference transforms."""
    training = make_fingerprint_frame()
    inference = training.iloc[[1]].copy()
    preprocessor = RSSIPreprocessor()
    fitted_output = preprocessor.fit_transform(training)
    reused_output = preprocessor.transform(inference)

    restored = pickle.loads(pickle.dumps(preprocessor))
    restored_output = restored.transform(inference)

    pd.testing.assert_frame_equal(reused_output, restored_output)
    pd.testing.assert_frame_equal(
        preprocessor.transform(training),
        fitted_output,
    )
    assert restored.get_feature_names_out() == RSSI_FEATURE_COLUMNS


def test_transform_requires_fitted_preprocessor() -> None:
    """Avoid accidental inference with an unfitted preprocessing object."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        RSSIPreprocessor().transform(make_fingerprint_frame())


def test_replacement_cannot_overlap_valid_rssi_values() -> None:
    """Prevent detected RSSI values from colliding with the unavailable marker."""
    with pytest.raises(ValueError, match="outside the valid RSSI range"):
        RSSIPreprocessor(replacement_value=-100)
