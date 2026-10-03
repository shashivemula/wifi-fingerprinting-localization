"""Evaluate a saved WiFi WKNN model on the official validation dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod

from src.config import MODELS_DIR, REPORTS_DIR, VALIDATION_DATA_PATH
from src.data.loader import PathLike, load_validation_data
from src.data.validator import ValidationReport, validate_dataframe
from src.geo.coordinate_converter import projected_to_latlon
from src.model.wknn_localizer import TARGET_COLUMNS
from src.prediction.predictor import WiFiFingerprintPredictor

_WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics, per-sample predictions, and generated output paths."""

    metrics: dict[str, float]
    sample_predictions: pd.DataFrame
    validation_report: ValidationReport
    results_path: Path
    error_plot_path: Path
    coordinate_plot_path: Path


def _calculate_metrics(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    geodesic_errors: np.ndarray,
    projected_errors: np.ndarray,
) -> dict[str, float]:
    """Calculate requested classification and localization metrics."""
    metrics = {
        "building_accuracy": float(
            np.mean(actual["BUILDINGID"].to_numpy() == predicted["building"].to_numpy())
        ),
        "floor_accuracy": float(
            np.mean(actual["FLOOR"].to_numpy() == predicted["floor"].to_numpy())
        ),
        "mean_localization_error_m": float(np.mean(geodesic_errors)),
        "median_localization_error_m": float(np.median(geodesic_errors)),
        "localization_rmse_m": float(np.sqrt(np.mean(np.square(geodesic_errors)))),
        "error_50th_percentile_m": float(np.percentile(geodesic_errors, 50)),
        "error_75th_percentile_m": float(np.percentile(geodesic_errors, 75)),
        "error_90th_percentile_m": float(np.percentile(geodesic_errors, 90)),
        "within_1_meter_percent": float(np.mean(geodesic_errors <= 1.0) * 100),
        "within_3_meters_percent": float(np.mean(geodesic_errors <= 3.0) * 100),
        "within_5_meters_percent": float(np.mean(geodesic_errors <= 5.0) * 100),
        "mean_projected_coordinate_error_m": float(np.mean(projected_errors)),
        "median_projected_coordinate_error_m": float(np.median(projected_errors)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced non-finite metric values.")
    return metrics


def _save_metric_table(
    metrics: dict[str, float],
    sample_count: int,
    destination: Path,
) -> None:
    """Write aggregate metrics as a clear, machine-readable CSV table."""
    accuracy_metrics = {"building_accuracy", "floor_accuracy"}
    percentage_metrics = {
        "within_1_meter_percent",
        "within_3_meters_percent",
        "within_5_meters_percent",
    }
    rows = []
    for metric, value in metrics.items():
        unit = (
            "proportion"
            if metric in accuracy_metrics
            else "percent"
            if metric in percentage_metrics
            else "meters"
        )
        rows.append(
            {
                "metric": metric,
                "value": value,
                "unit": unit,
                "validation_sample_count": sample_count,
            }
        )
    pd.DataFrame(rows).to_csv(destination, index=False)


def _save_error_distribution(
    errors: np.ndarray,
    destination: Path,
) -> None:
    """Save validation geodesic localization errors in meters."""
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(errors, bins="auto", edgecolor="black", alpha=0.8)
    axis.set(
        title="Validation Localization Error Distribution",
        xlabel="WGS84 geodesic error (meters)",
        ylabel="Validation samples",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _save_coordinate_comparison(
    sample_predictions: pd.DataFrame,
    destination: Path,
) -> None:
    """Save actual/predicted coordinate scatter in the dataset projected CRS."""
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(
        sample_predictions["actual_x"],
        sample_predictions["actual_y"],
        s=20,
        alpha=0.65,
        label="Actual",
    )
    axis.scatter(
        sample_predictions["predicted_x"],
        sample_predictions["predicted_y"],
        s=20,
        alpha=0.65,
        label="Predicted",
        marker="x",
    )
    axis.set(
        title="Actual vs Predicted Indoor Coordinates",
        xlabel="X (EPSG:3857 meters)",
        ylabel="Y (EPSG:3857 meters)",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def evaluate_validation_set(
    model_path: PathLike = MODELS_DIR / "wknn_localizer.pkl",
    preprocessor_path: PathLike = MODELS_DIR / "rssi_preprocessor.pkl",
    validation_path: PathLike = VALIDATION_DATA_PATH,
    reports_dir: PathLike = REPORTS_DIR,
) -> EvaluationResult:
    """Evaluate saved model predictions on the official validation dataset.

    The predictor receives only WAP001-WAP520. Actual coordinates and
    classification labels are kept separate and used only after prediction.
    Geographic localization error is computed using the WGS84 ellipsoidal
    geodesic; indoor-coordinate error is reported separately in projected
    EPSG:3857 coordinate units (meters).

    Args:
        model_path: Saved WKNN model artifact.
        preprocessor_path: Saved fitted RSSI preprocessor artifact.
        validation_path: Official validation CSV path or its directory.
        reports_dir: Directory for the CSV and PNG outputs.

    Returns:
        Computed metrics, sample-level predictions, validator report, and paths.

    Raises:
        ValueError: If validation data fails schema/data validation.
    """
    validation_data = load_validation_data(validation_path)
    validation_report = validate_dataframe(
        validation_data,
        dataset_name="official validation",
    )
    if not validation_report.is_valid:
        raise ValueError(validation_report.to_text())

    predictor = WiFiFingerprintPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )
    actual = validation_data.loc[:, TARGET_COLUMNS].reset_index(drop=True)
    fingerprints = validation_data.loc[
        :,
        predictor.preprocessor.get_feature_names_out(),
    ]
    predictions = predictor.predict_batch(fingerprints)
    predicted = pd.DataFrame(predictions)

    actual_latitude, actual_longitude = projected_to_latlon(
        actual["LONGITUDE"].to_numpy(dtype=np.float64, copy=False),
        actual["LATITUDE"].to_numpy(dtype=np.float64, copy=False),
    )
    predicted_latitude, predicted_longitude = projected_to_latlon(
        predicted["x"].to_numpy(dtype=np.float64, copy=False),
        predicted["y"].to_numpy(dtype=np.float64, copy=False),
    )
    _, _, geodesic_errors = _WGS84_GEOD.inv(
        actual_longitude,
        actual_latitude,
        predicted_longitude,
        predicted_latitude,
    )
    geodesic_errors = np.asarray(geodesic_errors, dtype=np.float64)

    projected_errors = np.hypot(
        actual["LONGITUDE"].to_numpy(dtype=np.float64, copy=False)
        - predicted["x"].to_numpy(dtype=np.float64, copy=False),
        actual["LATITUDE"].to_numpy(dtype=np.float64, copy=False)
        - predicted["y"].to_numpy(dtype=np.float64, copy=False),
    )
    metrics = _calculate_metrics(actual, predicted, geodesic_errors, projected_errors)

    sample_predictions = pd.DataFrame(
        {
            "actual_building": actual["BUILDINGID"],
            "predicted_building": predicted["building"],
            "actual_floor": actual["FLOOR"],
            "predicted_floor": predicted["floor"],
            "actual_x": actual["LONGITUDE"],
            "actual_y": actual["LATITUDE"],
            "predicted_x": predicted["x"],
            "predicted_y": predicted["y"],
            "actual_latitude": actual_latitude,
            "actual_longitude": actual_longitude,
            "predicted_latitude": predicted_latitude,
            "predicted_longitude": predicted_longitude,
            "geodesic_error_m": geodesic_errors,
            "projected_coordinate_error_m": projected_errors,
        }
    )

    output_directory = Path(reports_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "evaluation_results.csv"
    error_plot_path = output_directory / "localization_error_distribution.png"
    coordinate_plot_path = (
        output_directory / "actual_vs_predicted_coordinates.png"
    )
    _save_metric_table(metrics, len(validation_data), results_path)
    _save_error_distribution(geodesic_errors, error_plot_path)
    _save_coordinate_comparison(sample_predictions, coordinate_plot_path)

    return EvaluationResult(
        metrics=metrics,
        sample_predictions=sample_predictions,
        validation_report=validation_report,
        results_path=results_path,
        error_plot_path=error_plot_path,
        coordinate_plot_path=coordinate_plot_path,
    )


def _print_evaluation(result: EvaluationResult) -> None:
    """Print every aggregate validation metric and output location."""
    print("Official UJIIndoorLoc validation evaluation")
    print(f"Validation samples: {len(result.sample_predictions):,}")
    print(f"Validation status: {result.validation_report.to_text().splitlines()[1]}")
    print("\nMetrics:")
    for name, value in result.metrics.items():
        unit = (
            "proportion"
            if name in {"building_accuracy", "floor_accuracy"}
            else "%"
            if name.endswith("_percent")
            else "m"
        )
        print(f"  {name}: {value:.6f} {unit}")
    print(f"\nResults CSV: {result.results_path}")
    print(f"Localization error plot: {result.error_plot_path}")
    print(f"Actual vs predicted coordinates plot: {result.coordinate_plot_path}")


def main(arguments: Sequence[str] | None = None) -> None:
    """Evaluate the saved model from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved WKNN model on official validation data."
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
        help="Saved RSSI preprocessor path.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=VALIDATION_DATA_PATH,
        help="Official validation CSV path or containing directory.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Output directory for CSV and PNG evaluation results.",
    )
    options = parser.parse_args(arguments)
    result = evaluate_validation_set(
        model_path=options.model,
        preprocessor_path=options.preprocessor,
        validation_path=options.validation,
        reports_dir=options.reports_dir,
    )
    _print_evaluation(result)


if __name__ == "__main__":
    main()
