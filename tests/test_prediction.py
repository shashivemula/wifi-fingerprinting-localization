"""Tests for saved-artifact inference using synthetic fingerprints."""

import joblib
import pandas as pd
import pytest

from src.config import RSSI_FEATURE_COLUMNS
from src.model.wknn_localizer import WiFiWKNNLocalizer
from src.prediction import predictor as predictor_module
from src.prediction.predictor import (
    PredictionPipelineError,
    WiFiFingerprintPredictor,
)
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor


@pytest.fixture
def saved_predictor(tmp_path) -> WiFiFingerprintPredictor:
    """Create serialized model/preprocessing artifacts with obvious results."""
    training = pd.DataFrame(
        100,
        index=range(2),
        columns=RSSI_FEATURE_COLUMNS,
        dtype="int64",
    )
    training["WAP001"] = [-50, -60]
    targets = pd.DataFrame(
        {
            "LONGITUDE": [-7000.0, -6900.0],
            "LATITUDE": [4_864_800.0, 4_864_900.0],
            "BUILDINGID": [1, 2],
            "FLOOR": [0, 3],
        }
    )
    preprocessor = RSSIPreprocessor().fit(training)
    transformed = preprocessor.transform(training)
    model = WiFiWKNNLocalizer(k=1).fit(transformed, targets)

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    return WiFiFingerprintPredictor(model_path, preprocessor_path)


def make_fingerprint(value: int = -50) -> dict[str, int]:
    """Return a complete single WAP fingerprint mapping."""
    fingerprint = {column: 100 for column in RSSI_FEATURE_COLUMNS}
    fingerprint["WAP001"] = value
    return fingerprint


def test_predict_single_returns_required_prediction_fields(
    saved_predictor: WiFiFingerprintPredictor,
) -> None:
    """Use saved artifacts and return the exact public output mapping."""
    prediction = saved_predictor.predict_single(make_fingerprint())

    assert prediction == {
        "building": 1,
        "floor": 0,
        "x": -7000.0,
        "y": 4_864_800.0,
    }


def test_predict_batch_accepts_mappings_and_dataframe(
    saved_predictor: WiFiFingerprintPredictor,
) -> None:
    """Predict a sequence or DataFrame while preserving row order."""
    mappings = [
        make_fingerprint(-50),
        make_fingerprint(-60),
    ]
    mapping_results = saved_predictor.predict_batch(mappings)
    dataframe_results = saved_predictor.predict_batch(pd.DataFrame(mappings))

    assert [result["building"] for result in mapping_results] == [1, 2]
    assert [result["floor"] for result in mapping_results] == [0, 3]
    assert [result["x"] for result in mapping_results] == [-7000.0, -6900.0]
    assert mapping_results == dataframe_results


def test_metadata_and_targets_are_discarded_before_prediction(
    saved_predictor: WiFiFingerprintPredictor,
) -> None:
    """Ground-truth and arbitrary metadata cannot alter prediction input."""
    fingerprint = make_fingerprint(-50)
    fingerprint.update(
        {
            "LONGITUDE": 999999.0,
            "LATITUDE": 999999.0,
            "BUILDINGID": 99,
            "FLOOR": 99,
            "USERID": 999,
            "PHONEID": 999,
            "TIMESTAMP": 999999,
        }
    )

    prediction = saved_predictor.predict_single(fingerprint)

    assert prediction["building"] == 1
    assert prediction["floor"] == 0
    assert prediction["x"] == -7000.0


def test_missing_wap_columns_raise_clear_error(
    saved_predictor: WiFiFingerprintPredictor,
) -> None:
    """Report the absent WAP feature instead of passing incomplete data."""
    fingerprint = make_fingerprint()
    del fingerprint["WAP520"]

    with pytest.raises(ValueError, match="WAP520"):
        saved_predictor.predict_single(fingerprint)


def test_invalid_rssi_values_are_reported(
    saved_predictor: WiFiFingerprintPredictor,
) -> None:
    """Reject invalid and non-numeric RSSI data through the fitted preprocessor."""
    invalid = make_fingerprint(-105)
    with pytest.raises(ValueError, match="invalid RSSI"):
        saved_predictor.predict_single(invalid)

    non_numeric = make_fingerprint()
    non_numeric["WAP010"] = "not-rssi"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="non-numeric"):
        saved_predictor.predict_single(non_numeric)


def test_single_and_batch_validate_input_cardinality(
    saved_predictor: WiFiFingerprintPredictor,
) -> None:
    """Reject empty batches and multi-row input to predict_single."""
    with pytest.raises(ValueError, match="at least one"):
        saved_predictor.predict_batch([])
    with pytest.raises(ValueError, match="exactly one"):
        saved_predictor.predict_single(
            pd.DataFrame([make_fingerprint(), make_fingerprint()])
        )


def test_cli_demo_displays_actual_labels_but_sends_only_wap_features(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep validation targets separate from features sent to inference."""
    validation = pd.DataFrame(
        [make_fingerprint()],
    )
    validation["LONGITUDE"] = [-7012.5]
    validation["LATITUDE"] = [4_864_812.5]
    validation["BUILDINGID"] = [2]
    validation["FLOOR"] = [4]
    validation["USERID"] = [100]

    def fake_load_validation_data(path: str) -> pd.DataFrame:
        assert path == "unused.csv"
        return validation

    monkeypatch.setattr(
        predictor_module,
        "load_validation_data",
        fake_load_validation_data,
    )

    class CapturingPredictor:
        """Record model inputs for the CLI boundary test."""

        received_columns: tuple[str, ...] | None = None

        def predict_single(self, fingerprint: pd.DataFrame) -> dict[str, int | float]:
            self.received_columns = tuple(fingerprint.columns)
            return {"building": 1, "floor": 3, "x": -7001.0, "y": 4_864_810.0}

    fake_predictor = CapturingPredictor()
    predictor_module._print_validation_demo(fake_predictor, "unused.csv", 0)  # type: ignore[arg-type]

    output = capsys.readouterr().out
    assert fake_predictor.received_columns == RSSI_FEATURE_COLUMNS
    assert "Actual:" in output
    assert "Building: 2" in output
    assert "Floor: 4" in output
    assert "X: -7012.5" in output
    assert "Y: 4864812.5" in output
    assert "Predicted:" in output
    assert output.count("Building:") == 2
    assert output.count("Floor:") == 2


def test_missing_saved_artifacts_have_clear_errors(tmp_path) -> None:
    """Fail during initialization when model or preprocessing artifact is absent."""
    model_path = tmp_path / "missing-model.pkl"
    preprocessor_path = tmp_path / "missing-preprocessor.pkl"

    with pytest.raises(FileNotFoundError, match="WKNN model"):
        WiFiFingerprintPredictor(model_path, preprocessor_path)


def test_preprocessor_artifact_must_be_fitted_model_configuration(
    tmp_path,
) -> None:
    """Reject a serialized object that is not a fitted RSSI preprocessor."""
    model = WiFiWKNNLocalizer(k=1)
    model_path = tmp_path / "model.pkl"
    invalid_preprocessor_path = tmp_path / "wrong-preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump(model, invalid_preprocessor_path)

    with pytest.raises(PredictionPipelineError, match="must contain an RSSIPreprocessor"):
        WiFiFingerprintPredictor(model_path, invalid_preprocessor_path)
