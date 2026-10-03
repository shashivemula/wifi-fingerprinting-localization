"""Load UJIIndoorLoc CSV datasets from configurable paths."""

from pathlib import Path
from typing import TypeAlias

import pandas as pd

from src.config import RAW_DATA_DIR, TRAINING_DATA_PATH, VALIDATION_DATA_PATH

PathLike: TypeAlias = str | Path


class DatasetLoadError(Exception):
    """Raised when a requested dataset cannot be located or read."""


def _available_csv_files(directory: Path) -> list[Path]:
    """Return CSV files in a directory, sorted by filename."""
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: path.name.casefold(),
    )


def _resolve_dataset_path(path: PathLike | None, expected_path: Path, role: str) -> Path:
    """Resolve a requested file or discover its expected name in a directory."""
    requested_path = Path(path) if path is not None else expected_path
    directory = requested_path if requested_path.is_dir() else requested_path.parent
    expected_name = expected_path.name if requested_path.is_dir() else requested_path.name
    matching_files = [
        candidate
        for candidate in _available_csv_files(directory)
        if candidate.name.casefold() == expected_name.casefold()
    ]

    if requested_path.is_dir():
        if len(matching_files) == 1:
            return matching_files[0]
        if len(matching_files) > 1:
            names = ", ".join(candidate.name for candidate in matching_files)
            raise DatasetLoadError(
                f"Multiple {role} CSV files match {expected_name!r} in "
                f"{requested_path}: {names}."
            )
        missing_path = requested_path / expected_name
    else:
        if requested_path.is_file():
            return requested_path
        if len(matching_files) == 1:
            return matching_files[0]
        missing_path = requested_path

    available = _available_csv_files(directory)
    available_text = ", ".join(candidate.name for candidate in available) or "none"
    raise DatasetLoadError(
        f"Required {role} CSV file {missing_path} was not found. "
        f"CSV files available in {directory}: {available_text}."
    )


def _load_csv(path: PathLike | None, expected_path: Path, role: str) -> pd.DataFrame:
    """Read one CSV file with a clear, role-specific error on failure."""
    resolved_path = _resolve_dataset_path(path, expected_path, role)
    try:
        return pd.read_csv(resolved_path, low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DatasetLoadError(
            f"Unable to read {role} CSV file {resolved_path}: {exc}"
        ) from exc


def load_training_data(path: PathLike | None = None) -> pd.DataFrame:
    """Load the training CSV, or find it in the supplied directory.

    Args:
        path: CSV file or directory containing ``trainingData.csv``. If omitted,
            the path configured in :mod:`src.config` is used.

    Returns:
        The training dataset as a pandas DataFrame.

    Raises:
        DatasetLoadError: If the expected CSV is missing or cannot be read.
    """
    return _load_csv(path, TRAINING_DATA_PATH, "training")


def load_validation_data(path: PathLike | None = None) -> pd.DataFrame:
    """Load the validation CSV, or find it in the supplied directory.

    Args:
        path: CSV file or directory containing ``validationData.csv``. If
            omitted, the path configured in :mod:`src.config` is used.

    Returns:
        The validation dataset as a pandas DataFrame.

    Raises:
        DatasetLoadError: If the expected CSV is missing or cannot be read.
    """
    return _load_csv(path, VALIDATION_DATA_PATH, "validation")


def load_datasets(
    data_dir: PathLike = RAW_DATA_DIR,
    *,
    training_path: PathLike | None = None,
    validation_path: PathLike | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation CSVs once each from configurable paths.

    Args:
        data_dir: Directory containing the expected CSV filenames.
        training_path: Optional explicit training CSV path.
        validation_path: Optional explicit validation CSV path.

    Returns:
        A ``(training, validation)`` pair of pandas DataFrames.
    """
    training = load_training_data(training_path or data_dir)
    validation = load_validation_data(validation_path or data_dir)
    return training, validation
