"""Evaluation tests using saved synthetic WKNN artifacts and validation rows."""

import joblib
import pandas as pd
import pytest

from src.config import RSSI_FEATURE_COLUMNS
from src.data.validator import TARGET_COLUMNS
from src.evaluation.metrics import evaluate_validation_set
from src.model.wknn_localizer import WiFiWKNNLocalizer
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor


def make_dataset() -> pd.DataFrame:
    """Create a valid small dataset with actual fingerprints as exact neighbors."""
    frame = pd.DataFrame(
        100,
        index=range(3),
        columns=RSSI_FEATURE_COLUMNS,
        dtype="int64",
    )
    frame["WAP001"] = [-50, -60, -70]
    frame["LONGITUDE"] = [-7541.2643, -7515.916799, -7490.0]
    frame["LATITUDE"] = [4_864_920.7782, 4_864_890.0, 4_864_860.0]
    frame["BUILDINGID"] = [0, 1, 1]
    frame["FLOOR"] = [0, 1, 2]
    frame["SPACEID"] = [10, 20, 30]
    frame["RELATIVEPOSITION"] = [1, 1, 1]
    frame["USERID"] = [1, 1, 1]
    frame["PHONEID"] = [1, 1, 1]
    frame["TIMESTAMP"] = [1_000, 2_000, 3_000]
    return frame.loc[:, [*RSSI_FEATURE_COLUMNS, *TARGET_COLUMNS]]


def test_evaluation_uses_validation_labels_only_for_comparison_and_saves_outputs(
    tmp_path,
) -> None:
    """Compute actual validation metrics and produce CSV and both plots."""
    training = make_dataset()
    validation = training.iloc[[0, 1]].copy()
    validation_path = tmp_path / "validationData.csv"
    validation.to_csv(validation_path, index=False)

    preprocessor = RSSIPreprocessor().fit(training)
    transformed_training = preprocessor.transform(training)
    model = WiFiWKNNLocalizer(k=1).fit(
        transformed_training,
        training.loc[:, ["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]],
    )
    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)

    result = evaluate_validation_set(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        validation_path=validation_path,
        reports_dir=tmp_path / "reports",
    )

    assert len(result.sample_predictions) == 2
    assert result.metrics["building_accuracy"] == 1.0
    assert result.metrics["floor_accuracy"] == 1.0
    assert result.metrics["mean_localization_error_m"] == pytest.approx(0.0)
    assert result.metrics["localization_rmse_m"] == pytest.approx(0.0)
    assert result.metrics["within_1_meter_percent"] == 100.0
    assert result.results_path.is_file()
    assert result.error_plot_path.is_file()
    assert result.coordinate_plot_path.is_file()

    results_table = pd.read_csv(result.results_path)
    assert set(results_table["metric"]) == set(result.metrics)
    assert "localization_rmse_m" in set(results_table["metric"])
