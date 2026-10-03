"""Train and persist the WiFi WKNN localizer from UJIIndoorLoc training data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import MODELS_DIR, RSSI_FEATURE_COLUMNS, TRAINING_DATA_PATH
from src.data.loader import PathLike, load_training_data
from src.data.validator import ValidationReport, validate_dataframe
from src.model.wknn_localizer import (
    TARGET_COLUMNS,
    LocalizationPrediction,
    WiFiWKNNLocalizer,
)
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor

DEFAULT_K = 5
DEFAULT_DISTANCE_METRIC = "euclidean"
DEFAULT_WEIGHTING_STRATEGY = "inverse_distance"
DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class HoldoutMetrics:
    """Metrics measured on a held-out subset of training data."""

    sample_count: int
    coordinate_mean_error: float
    coordinate_median_error: float
    building_accuracy: float
    floor_accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    """Fitted artifacts and optional held-out training metrics."""

    model: WiFiWKNNLocalizer
    preprocessor: RSSIPreprocessor
    validation_report: ValidationReport
    holdout_metrics: HoldoutMetrics | None
    training_sample_count: int
    feature_count: int
    model_path: Path
    preprocessor_path: Path


def _evaluate_holdout(
    training_data: pd.DataFrame,
    *,
    k: int,
    distance_metric: str,
    weighting_strategy: str,
    holdout_fraction: float,
    random_state: int,
) -> HoldoutMetrics | None:
    """Evaluate training-only, grouping identical WAP fingerprints together."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be greater than 0 and less than 1.")
    if len(training_data) < 3:
        return None

    indices = np.arange(len(training_data))
    wap_groups = pd.util.hash_pandas_object(
        training_data.loc[:, RSSI_FEATURE_COLUMNS],
        index=False,
    ).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_fraction,
        random_state=random_state,
    )
    fit_indices, holdout_indices = next(
        splitter.split(indices, groups=wap_groups)
    )
    if len(fit_indices) < k:
        raise ValueError(
            f"Training split for holdout evaluation contains {len(fit_indices)} "
            f"samples, fewer than k={k}."
        )

    fit_data = training_data.iloc[fit_indices]
    holdout_data = training_data.iloc[holdout_indices]
    holdout_preprocessor = RSSIPreprocessor().fit(fit_data)
    fit_features = holdout_preprocessor.transform(fit_data)
    holdout_features = holdout_preprocessor.transform(holdout_data)

    holdout_model = WiFiWKNNLocalizer(
        k=k,
        distance_metric=distance_metric,
        weighting_strategy=weighting_strategy,  # type: ignore[arg-type]
    )
    holdout_model.fit(fit_features, fit_data.loc[:, TARGET_COLUMNS])
    predictions: list[LocalizationPrediction] = holdout_model.predict(
        holdout_features
    )

    predicted_coordinates = np.array(
        [(prediction.x, prediction.y) for prediction in predictions],
        dtype=np.float64,
    )
    actual_coordinates = holdout_data.loc[
        :, ["LONGITUDE", "LATITUDE"]
    ].to_numpy(dtype=np.float64, copy=False)
    coordinate_errors = np.linalg.norm(
        predicted_coordinates - actual_coordinates,
        axis=1,
    )
    building_accuracy = np.mean(
        [
            prediction.building == int(actual)
            for prediction, actual in zip(
                predictions,
                holdout_data["BUILDINGID"].to_numpy(),
            )
        ]
    )
    floor_accuracy = np.mean(
        [
            prediction.floor == int(actual)
            for prediction, actual in zip(
                predictions,
                holdout_data["FLOOR"].to_numpy(),
            )
        ]
    )
    return HoldoutMetrics(
        sample_count=len(holdout_data),
        coordinate_mean_error=float(coordinate_errors.mean()),
        coordinate_median_error=float(np.median(coordinate_errors)),
        building_accuracy=float(building_accuracy),
        floor_accuracy=float(floor_accuracy),
    )


def train_pipeline(
    training_path: PathLike = TRAINING_DATA_PATH,
    model_path: PathLike = MODELS_DIR / "wknn_localizer.pkl",
    preprocessor_path: PathLike = MODELS_DIR / "rssi_preprocessor.pkl",
    *,
    k: int = DEFAULT_K,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
    weighting_strategy: str = DEFAULT_WEIGHTING_STRATEGY,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainingResult:
    """Load, validate, preprocess, evaluate, fit, and save a WKNN localizer.

    Only the designated training CSV is loaded. The official validation dataset
    is never read or used for fitting or hold-out evaluation.

    Args:
        training_path: Training CSV path or containing directory.
        model_path: Destination for the fitted localizer.
        preprocessor_path: Destination for the fitted RSSI preprocessing
            configuration used by the model.
        k: WKNN neighbor count.
        distance_metric: Metric passed to scikit-learn nearest-neighbor search.
        weighting_strategy: WKNN weighting strategy.
        holdout_fraction: Fraction of training rows reserved for an optional
            internal training-only evaluation.
        random_state: Seed for reproducible training-only evaluation split.

    Returns:
        Paths, fitted components, validation report, and actual holdout metrics.

    Raises:
        ValueError: If training data fails validation or configuration is
            inconsistent.
    """
    training_data = load_training_data(training_path)
    report = validate_dataframe(training_data, dataset_name="training")
    if not report.is_valid:
        raise ValueError(report.to_text())

    preprocessor = RSSIPreprocessor()
    training_features = preprocessor.fit_transform(training_data)
    targets = training_data.loc[:, TARGET_COLUMNS]

    holdout_metrics = _evaluate_holdout(
        training_data,
        k=k,
        distance_metric=distance_metric,
        weighting_strategy=weighting_strategy,
        holdout_fraction=holdout_fraction,
        random_state=random_state,
    )

    model = WiFiWKNNLocalizer(
        k=k,
        distance_metric=distance_metric,
        weighting_strategy=weighting_strategy,  # type: ignore[arg-type]
    )
    model.fit(training_features, targets)

    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_absolute():
        resolved_model_path = MODELS_DIR / resolved_model_path
    resolved_preprocessor_path = Path(preprocessor_path)
    if not resolved_preprocessor_path.is_absolute():
        resolved_preprocessor_path = MODELS_DIR / resolved_preprocessor_path

    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(resolved_model_path)
    joblib.dump(preprocessor, resolved_preprocessor_path)

    return TrainingResult(
        model=model,
        preprocessor=preprocessor,
        validation_report=report,
        holdout_metrics=holdout_metrics,
        training_sample_count=len(training_data),
        feature_count=len(preprocessor.get_feature_names_out()),
        model_path=resolved_model_path,
        preprocessor_path=resolved_preprocessor_path,
    )


def _print_result(result: TrainingResult) -> None:
    """Print training configuration, artifact paths, and measured metrics."""
    print("Training data validation:")
    print(result.validation_report.to_text())
    print("\nTraining complete")
    print(f"Number of training samples: {result.training_sample_count:,}")
    print(f"Number of WAP features: {result.feature_count}")
    print(f"K value: {result.model.k}")
    print(f"Distance metric: {result.model.distance_metric}")
    print(f"Weighting strategy: {result.model.weighting_strategy}")
    print(f"Model save path: {result.model_path}")
    print(f"Preprocessing configuration save path: {result.preprocessor_path}")

    if result.holdout_metrics is None:
        print("Held-out training metrics: unavailable (not enough training rows).")
        return

    metrics = result.holdout_metrics
    print(
        f"Held-out training subset: {metrics.sample_count:,} samples "
        f"(drawn only from trainingData.csv)"
    )
    print(
        "Coordinate mean/median Euclidean error (dataset coordinate units): "
        f"{metrics.coordinate_mean_error:.4f} / "
        f"{metrics.coordinate_median_error:.4f}"
    )
    print(f"Building accuracy: {metrics.building_accuracy:.4f}")
    print(f"Floor accuracy: {metrics.floor_accuracy:.4f}")


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the training pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Train the UJIIndoorLoc WiFi WKNN localizer."
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=TRAINING_DATA_PATH,
        help="Training CSV path or directory containing trainingData.csv.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODELS_DIR / "wknn_localizer.pkl",
        help="Output path for the fitted WKNN model.",
    )
    parser.add_argument(
        "--preprocessor-output",
        type=Path,
        default=MODELS_DIR / "rssi_preprocessor.pkl",
        help="Output path for the fitted RSSI preprocessor.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--distance-metric", default=DEFAULT_DISTANCE_METRIC)
    parser.add_argument(
        "--weighting-strategy",
        choices=("inverse_distance", "uniform"),
        default=DEFAULT_WEIGHTING_STRATEGY,
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Training-only holdout fraction for basic behavior evaluation.",
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    options = parser.parse_args(arguments)

    result = train_pipeline(
        training_path=options.training,
        model_path=options.model_output,
        preprocessor_path=options.preprocessor_output,
        k=options.k,
        distance_metric=options.distance_metric,
        weighting_strategy=options.weighting_strategy,
        holdout_fraction=options.holdout_fraction,
        random_state=options.random_state,
    )
    _print_result(result)


if __name__ == "__main__":
    main()
