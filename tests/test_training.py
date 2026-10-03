"""Tests for the training pipeline using synthetic UJIIndoorLoc rows."""

import joblib
import pandas as pd

from src.config import RSSI_FEATURE_COLUMNS
from src.data.validator import TARGET_COLUMNS as VALIDATOR_TARGET_COLUMNS
from src.training.train import train_pipeline


def make_training_frame(sample_count: int = 8) -> pd.DataFrame:
    """Construct a small valid dataset with distinct WAP fingerprints."""
    frame = pd.DataFrame(
        100,
        index=range(sample_count),
        columns=RSSI_FEATURE_COLUMNS,
        dtype="int64",
    )
    frame["WAP001"] = [-50 - index for index in range(sample_count)]
    frame["LONGITUDE"] = [-7000.0 + index for index in range(sample_count)]
    frame["LATITUDE"] = [4_864_800.0 + index for index in range(sample_count)]
    frame["BUILDINGID"] = [index % 2 for index in range(sample_count)]
    frame["FLOOR"] = [index % 3 for index in range(sample_count)]
    frame["SPACEID"] = [index + 1 for index in range(sample_count)]
    frame["RELATIVEPOSITION"] = [1] * sample_count
    frame["USERID"] = [1] * sample_count
    frame["PHONEID"] = [1] * sample_count
    frame["TIMESTAMP"] = [1_000 + index for index in range(sample_count)]
    return frame.loc[:, [*RSSI_FEATURE_COLUMNS, *VALIDATOR_TARGET_COLUMNS]]


def test_training_pipeline_saves_model_and_preprocessor(tmp_path) -> None:
    """Fit on training CSV, evaluate a training-only holdout, save artifacts."""
    training_path = tmp_path / "trainingData.csv"
    model_path = tmp_path / "wknn_localizer.pkl"
    preprocessor_path = tmp_path / "rssi_preprocessor.pkl"
    make_training_frame().to_csv(training_path, index=False)

    result = train_pipeline(
        training_path=training_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        k=2,
        holdout_fraction=0.25,
        random_state=17,
    )

    assert result.training_sample_count == 8
    assert result.feature_count == 520
    assert result.holdout_metrics is not None
    assert result.holdout_metrics.sample_count == 2
    assert result.model_path == model_path
    assert result.preprocessor_path == preprocessor_path
    assert model_path.is_file()
    assert preprocessor_path.is_file()

    restored_model = result.model.load(model_path)
    restored_preprocessor = joblib.load(preprocessor_path)
    sample = make_training_frame().iloc[[0]]
    transformed = restored_preprocessor.transform(sample)
    prediction = restored_model.predict_single(transformed)
    assert prediction.x == -7000.0
    assert prediction.y == 4_864_800.0


def test_training_pipeline_can_skip_holdout_for_tiny_datasets(tmp_path) -> None:
    """Fit small data when it is too small for a useful held-out split."""
    training_path = tmp_path / "trainingData.csv"
    make_training_frame(sample_count=2).to_csv(training_path, index=False)

    result = train_pipeline(
        training_path=training_path,
        model_path=tmp_path / "model.pkl",
        preprocessor_path=tmp_path / "preprocessor.pkl",
        k=1,
    )

    assert result.holdout_metrics is None
    assert result.model_path.is_file()
    assert result.preprocessor_path.is_file()
