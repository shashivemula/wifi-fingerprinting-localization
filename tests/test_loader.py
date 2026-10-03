"""Tests for loading training and validation CSV files."""

import pandas as pd
import pytest

from src.data.loader import (
    DatasetLoadError,
    load_datasets,
    load_training_data,
    load_validation_data,
)


def test_load_datasets_discovers_expected_files(tmp_path) -> None:
    """Load both datasets from a supplied directory."""
    training_frame = pd.DataFrame({"WAP001": [-50], "BUILDINGID": [0]})
    validation_frame = pd.DataFrame({"WAP001": [-60], "BUILDINGID": [1]})
    training_frame.to_csv(tmp_path / "trainingData.csv", index=False)
    validation_frame.to_csv(tmp_path / "validationData.csv", index=False)

    training, validation = load_datasets(tmp_path)

    pd.testing.assert_frame_equal(training, training_frame)
    pd.testing.assert_frame_equal(validation, validation_frame)


def test_loader_matches_expected_filename_case_insensitively(tmp_path) -> None:
    """Discover expected dataset names without relying on exact filename case."""
    expected = pd.DataFrame({"WAP001": [-42]})
    expected.to_csv(tmp_path / "TrainingData.CSV", index=False)

    actual = load_training_data(tmp_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_missing_validation_file_lists_available_csvs(tmp_path) -> None:
    """Give a clear error when the validation CSV is missing."""
    (tmp_path / "trainingData.csv").write_text("WAP001\n-50\n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("text\nexample\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError) as error:
        load_validation_data(tmp_path)

    message = str(error.value)
    assert "validation" in message.lower()
    assert "validationData.csv" in message
    assert "trainingData.csv" in message
    assert "notes.csv" in message


def test_loader_accepts_an_explicit_csv_path(tmp_path) -> None:
    """Allow callers to override configured filenames with a file path."""
    expected = pd.DataFrame({"WAP001": [-70]})
    csv_path = tmp_path / "custom-training.csv"
    expected.to_csv(csv_path, index=False)

    actual = load_training_data(csv_path)

    pd.testing.assert_frame_equal(actual, expected)
