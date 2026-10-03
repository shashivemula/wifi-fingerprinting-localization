"""Interactive Folium maps for actual and predicted validation locations."""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR, VALIDATION_DATA_PATH
from src.evaluation.metrics import evaluate_validation_set


def _validate_sample_predictions(sample_predictions: pd.DataFrame) -> None:
    """Check the columns required for geographic position visualization."""
    required = (
        "actual_x",
        "actual_y",
        "predicted_x",
        "predicted_y",
        "actual_latitude",
        "actual_longitude",
        "predicted_latitude",
        "predicted_longitude",
        "geodesic_error_m",
    )
    missing = [column for column in required if column not in sample_predictions]
    if missing:
        raise ValueError(f"Sample predictions are missing columns: {missing}.")
    if sample_predictions.empty:
        raise ValueError("Sample predictions must contain at least one row.")
    numeric_values = sample_predictions.loc[:, required].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Sample locations and errors must be finite.")
    if (sample_predictions["geodesic_error_m"] < 0).any():
        raise ValueError("Localization errors must be non-negative.")


def create_individual_location_map(
    sample_predictions: pd.DataFrame,
    sample_index: int = 0,
    output_path: str | Path = REPORTS_DIR / "individual_validation_location.html",
) -> Path:
    """Create a Folium map with actual/predicted markers and connecting line.

    Args:
        sample_predictions: Sample-level rows produced by
            :func:`src.evaluation.metrics.evaluate_validation_set`.
        sample_index: Zero-based positional row to visualize.
        output_path: HTML destination.

    Returns:
        Path to the generated interactive map.
    """
    _validate_sample_predictions(sample_predictions)
    if sample_index < 0 or sample_index >= len(sample_predictions):
        raise IndexError(
            f"Sample index {sample_index} is out of range for "
            f"{len(sample_predictions)} predictions."
        )

    row = sample_predictions.iloc[sample_index]
    actual = (float(row["actual_latitude"]), float(row["actual_longitude"]))
    predicted = (
        float(row["predicted_latitude"]),
        float(row["predicted_longitude"]),
    )
    distance = float(row["geodesic_error_m"])
    map_view = folium.Map(
        location=[
            (actual[0] + predicted[0]) / 2,
            (actual[1] + predicted[1]) / 2,
        ],
        zoom_start=19,
        control_scale=True,
    )
    folium.Marker(
        location=actual,
        tooltip="Actual location",
        popup=folium.Popup(
            f"<b>Actual location</b><br>"
            f"Latitude: {actual[0]:.8f}<br>"
            f"Longitude: {actual[1]:.8f}<br>"
            f"Localization error: {distance:.3f} m",
            max_width=300,
        ),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(map_view)
    folium.Marker(
        location=predicted,
        tooltip="Predicted location",
        popup=folium.Popup(
            f"<b>Predicted location</b><br>"
            f"Latitude: {predicted[0]:.8f}<br>"
            f"Longitude: {predicted[1]:.8f}<br>"
            f"Localization error: {distance:.3f} m",
            max_width=300,
        ),
        icon=folium.Icon(color="red", icon="cross"),
    ).add_to(map_view)
    folium.PolyLine(
        locations=[actual, predicted],
        color="darkred",
        weight=3,
        tooltip=f"Localization error: {distance:.3f} m",
    ).add_to(map_view)
    map_view.fit_bounds([[actual[0], actual[1]], [predicted[0], predicted[1]]])

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(destination))
    print(
        f"Validation sample {sample_index} location:\n"
        f"  Actual: latitude={actual[0]:.8f}, longitude={actual[1]:.8f}\n"
        f"  Predicted: latitude={predicted[0]:.8f}, "
        f"longitude={predicted[1]:.8f}\n"
        f"  Localization error: {distance:.3f} m"
    )
    return destination


def visualize_validation_predictions(
    sample_predictions: pd.DataFrame,
    reports_dir: str | Path = REPORTS_DIR,
    sample_index: int = 0,
) -> dict[str, Path]:
    """Generate actual/predicted/error plots and an individual HTML map."""
    from src.visualization.location_plot import (
        plot_actual_indoor_positions,
        plot_actual_vs_predicted_locations,
        plot_geographic_positions,
        plot_localization_error,
        plot_predicted_indoor_positions,
    )

    _validate_sample_predictions(sample_predictions)
    destination = Path(reports_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "actual_indoor_positions": plot_actual_indoor_positions(
            sample_predictions,
            destination / "actual_indoor_positions.png",
        ),
        "predicted_indoor_positions": plot_predicted_indoor_positions(
            sample_predictions,
            destination / "predicted_indoor_positions.png",
        ),
        "actual_vs_predicted_locations": plot_actual_vs_predicted_locations(
            sample_predictions,
            destination / "actual_vs_predicted_locations.png",
        ),
        "localization_error": plot_localization_error(
            sample_predictions,
            destination / "localization_error.png",
        ),
        "geographic_positions": plot_geographic_positions(
            sample_predictions,
            destination / "geographic_positions.png",
        ),
        "individual_location_map": create_individual_location_map(
            sample_predictions,
            sample_index=sample_index,
            output_path=destination / "individual_validation_location.html",
        ),
    }
    return outputs


def main() -> None:
    """Generate visualizations using real predictions on official validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize saved WKNN predictions on validation data."
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
        help="Directory for generated visualizations.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based validation sample to show in the individual map.",
    )
    options = parser.parse_args()

    evaluation = evaluate_validation_set(
        model_path=options.model,
        preprocessor_path=options.preprocessor,
        validation_path=options.validation,
        reports_dir=options.reports_dir,
    )
    output_paths = visualize_validation_predictions(
        evaluation.sample_predictions,
        reports_dir=options.reports_dir,
        sample_index=options.sample_index,
    )
    print("\nGenerated visualizations:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
