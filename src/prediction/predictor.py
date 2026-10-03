"""Inference pipeline for saved WiFi WKNN model artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypeAlias

import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    RSSI_FEATURE_COLUMNS,
    VALIDATION_DATA_PATH,
)
from src.data.loader import PathLike, load_validation_data
from src.model.wknn_localizer import (
    TARGET_COLUMNS,
    WiFiWKNNLocalizer,
)
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor

Prediction: TypeAlias = dict[str, int | float]
Fingerprint: TypeAlias = Mapping[str, object] | pd.DataFrame


class PredictionPipelineError(RuntimeError):
    """Raised when saved inference artifacts cannot be used."""


class WiFiFingerprintPredictor:
    """Load the fitted preprocessing configuration and WKNN model for inference.

    Ground-truth or other metadata fields in an input DataFrame are ignored:
    only the canonical WAP columns are extracted before preprocessing.
    """

    def __init__(
        self,
        model_path: PathLike = MODELS_DIR / "wknn_localizer.pkl",
        preprocessor_path: PathLike = MODELS_DIR / "rssi_preprocessor.pkl",
    ) -> None:
        """Load and verify the saved model and preprocessor.

        Args:
            model_path: Path to the serialized ``WiFiWKNNLocalizer``.
            preprocessor_path: Path to the serialized fitted
                ``RSSIPreprocessor``.

        Raises:
            FileNotFoundError: If either artifact is missing.
            PredictionPipelineError: If an artifact has the wrong type or the
                preprocessing configuration is not fitted.
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"WKNN model file not found: {self.model_path}")
        if not self.preprocessor_path.is_file():
            raise FileNotFoundError(
                f"RSSI preprocessing configuration not found: "
                f"{self.preprocessor_path}"
            )

        try:
            self.model = WiFiWKNNLocalizer.load(self.model_path)
            loaded_preprocessor = joblib.load(self.preprocessor_path)
        except (OSError, ValueError, EOFError, KeyError) as exc:
            raise PredictionPipelineError(
                f"Unable to load inference artifacts: {exc}"
            ) from exc
        if not isinstance(loaded_preprocessor, RSSIPreprocessor):
            raise PredictionPipelineError(
                f"Preprocessing artifact must contain an RSSIPreprocessor; "
                f"found {type(loaded_preprocessor).__name__}."
            )
        if loaded_preprocessor.feature_columns_ != RSSI_FEATURE_COLUMNS:
            raise PredictionPipelineError(
                "Saved RSSI preprocessor is not fitted with the expected "
                "WAP001-WAP520 feature order."
            )
        self.preprocessor = loaded_preprocessor

    @staticmethod
    def _fingerprint_frame(fingerprint: Fingerprint) -> pd.DataFrame:
        """Convert input to a one-or-more-row DataFrame and check WAP columns."""
        if isinstance(fingerprint, pd.DataFrame):
            frame = fingerprint
        elif isinstance(fingerprint, Mapping):
            frame = pd.DataFrame([fingerprint])
        else:
            raise TypeError("fingerprint must be a mapping or pandas DataFrame.")

        if frame.empty:
            raise ValueError("fingerprint must contain at least one sample.")
        duplicate_columns = frame.columns[frame.columns.duplicated()]
        if not duplicate_columns.empty:
            duplicate_names = sorted({str(column) for column in duplicate_columns})
            raise ValueError(f"Fingerprint has duplicate columns: {duplicate_names}.")

        missing = [
            column for column in RSSI_FEATURE_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Fingerprint is missing {len(missing)} required WAP columns: "
                f"{', '.join(missing[:10])}."
            )
        return frame.loc[:, RSSI_FEATURE_COLUMNS]

    def _predict_frame(self, frame: pd.DataFrame) -> list[Prediction]:
        """Apply saved preprocessing and map model results to the public shape."""
        fingerprint_frame = self._fingerprint_frame(frame)
        transformed = self.preprocessor.transform(fingerprint_frame)
        model_predictions = self.model.predict(transformed)
        return [
            {
                "building": prediction.building,
                "floor": prediction.floor,
                "x": prediction.x,
                "y": prediction.y,
            }
            for prediction in model_predictions
        ]

    def predict_single(self, fingerprint: Fingerprint) -> Prediction:
        """Predict building, floor, and coordinates for one WiFi fingerprint.

        Args:
            fingerprint: A mapping containing WAP001-WAP520, or a one-row
                DataFrame. Any extra columns are discarded before inference.

        Returns:
            A dictionary with ``building``, ``floor``, ``x``, and ``y``.

        Raises:
            ValueError: If the fingerprint has missing features, multiple rows,
                or invalid RSSI values.
        """
        frame = self._fingerprint_frame(fingerprint)
        if len(frame) != 1:
            raise ValueError("predict_single requires exactly one fingerprint row.")
        return self._predict_frame(frame)[0]

    def predict_batch(
        self,
        fingerprints: pd.DataFrame | Iterable[Mapping[str, object]],
    ) -> list[Prediction]:
        """Predict all rows in a DataFrame or iterable of WAP mappings.

        Args:
            fingerprints: A DataFrame or iterable of mappings, each containing
                WAP001-WAP520.

        Returns:
            A prediction dictionary for each input fingerprint.

        Raises:
            ValueError: If the batch is empty, missing features, or contains
                invalid RSSI values.
        """
        if isinstance(fingerprints, pd.DataFrame):
            frame = fingerprints
        else:
            rows = list(fingerprints)
            if not rows:
                raise ValueError("predict_batch requires at least one fingerprint.")
            frame = pd.DataFrame(rows)
        return self._predict_frame(frame)


def _print_validation_demo(
    predictor: WiFiFingerprintPredictor,
    validation_path: PathLike,
    sample_index: int,
) -> Prediction:
    """Print actual validation labels and predict using WAP features only."""
    validation_data = load_validation_data(validation_path)
    if sample_index < 0 or sample_index >= len(validation_data):
        raise IndexError(
            f"Validation sample index {sample_index} is out of range for "
            f"{len(validation_data)} rows."
        )

    row = validation_data.iloc[[sample_index]]
    actual_values = row.loc[:, TARGET_COLUMNS].iloc[0]
    fingerprint = row.loc[:, RSSI_FEATURE_COLUMNS]
    prediction = predictor.predict_single(fingerprint)

    print("Actual:")
    print(f"Building: {actual_values['BUILDINGID']}")
    print(f"Floor: {actual_values['FLOOR']}")
    print(f"X: {actual_values['LONGITUDE']}")
    print(f"Y: {actual_values['LATITUDE']}")
    print("\nPredicted:")
    print(f"Building: {prediction['building']}")
    print(f"Floor: {prediction['floor']}")
    print(f"X: {prediction['x']}")
    print(f"Y: {prediction['y']}")
    return prediction


def main() -> None:
    """Run a one-sample prediction demo from the official validation CSV."""
    parser = argparse.ArgumentParser(
        description="Predict a validation sample using a saved WiFi WKNN model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "wknn_localizer.pkl",
        help="Saved WKNN model path.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=MODELS_DIR / "rssi_preprocessor.pkl",
        help="Saved RSSI preprocessing configuration path.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=VALIDATION_DATA_PATH,
        help="Validation CSV path or directory containing validationData.csv.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based validation row index to demonstrate.",
    )
    arguments = parser.parse_args()

    predictor = WiFiFingerprintPredictor(
        model_path=arguments.model,
        preprocessor_path=arguments.preprocessor,
    )
    _print_validation_demo(
        predictor,
        arguments.validation,
        arguments.sample_index,
    )


if __name__ == "__main__":
    main()
