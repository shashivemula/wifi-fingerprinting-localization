"""Matplotlib visualizations for actual and predicted indoor locations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR

_ACTUAL_COLOR = "#1f77b4"
_PREDICTED_COLOR = "#ff7f0e"


def _validate_location_columns(
    sample_predictions: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    """Raise a clear error when visualization input lacks required columns."""
    missing = [column for column in required_columns if column not in sample_predictions]
    if missing:
        raise ValueError(f"Location data is missing required columns: {missing}.")
    if sample_predictions.empty:
        raise ValueError("Location data must contain at least one sample.")
    try:
        values = sample_predictions.loc[:, required_columns].to_numpy(
            dtype=np.float64,
            copy=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Location coordinates and errors must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Location coordinates and errors must be finite.")


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    """Create the output directory and save a Matplotlib figure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_actual_indoor_positions(
    sample_predictions: pd.DataFrame,
    output_path: str | Path = REPORTS_DIR / "actual_indoor_positions.png",
) -> Path:
    """Plot actual validation positions in the source projected coordinate CRS."""
    _validate_location_columns(sample_predictions, ("actual_x", "actual_y"))
    figure, axis = plt.subplots(figsize=(9, 8))
    scatter = axis.scatter(
        sample_predictions["actual_x"],
        sample_predictions["actual_y"],
        c=sample_predictions.get("actual_building"),
        cmap="tab10",
        s=20,
        alpha=0.7,
        label="Actual",
    )
    axis.set(
        title="Actual Indoor Positions",
        xlabel="X (EPSG:3857 meters)",
        ylabel="Y (EPSG:3857 meters)",
    )
    if "actual_building" in sample_predictions:
        figure.colorbar(scatter, ax=axis, label="Building ID")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    return _save_figure(figure, Path(output_path))


def plot_predicted_indoor_positions(
    sample_predictions: pd.DataFrame,
    output_path: str | Path = REPORTS_DIR / "predicted_indoor_positions.png",
) -> Path:
    """Plot predicted validation positions in the source projected CRS."""
    _validate_location_columns(sample_predictions, ("predicted_x", "predicted_y"))
    figure, axis = plt.subplots(figsize=(9, 8))
    scatter = axis.scatter(
        sample_predictions["predicted_x"],
        sample_predictions["predicted_y"],
        c=sample_predictions.get("predicted_building"),
        cmap="tab10",
        marker="x",
        s=25,
        alpha=0.7,
        label="Predicted",
    )
    axis.set(
        title="Predicted Indoor Positions",
        xlabel="X (EPSG:3857 meters)",
        ylabel="Y (EPSG:3857 meters)",
    )
    if "predicted_building" in sample_predictions:
        figure.colorbar(scatter, ax=axis, label="Predicted building ID")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    return _save_figure(figure, Path(output_path))


def plot_actual_vs_predicted_locations(
    sample_predictions: pd.DataFrame,
    output_path: str | Path = REPORTS_DIR / "actual_vs_predicted_locations.png",
    *,
    max_connection_lines: int = 300,
) -> Path:
    """Overlay actual and predicted positions and connect a readable sample."""
    _validate_location_columns(
        sample_predictions,
        ("actual_x", "actual_y", "predicted_x", "predicted_y"),
    )
    if max_connection_lines < 0:
        raise ValueError("max_connection_lines cannot be negative.")

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        sample_predictions["actual_x"],
        sample_predictions["actual_y"],
        s=20,
        alpha=0.65,
        color=_ACTUAL_COLOR,
        label="Actual location",
    )
    axis.scatter(
        sample_predictions["predicted_x"],
        sample_predictions["predicted_y"],
        s=22,
        alpha=0.65,
        color=_PREDICTED_COLOR,
        marker="x",
        label="Predicted location",
    )

    line_count = min(max_connection_lines, len(sample_predictions))
    if line_count:
        row_indices = pd.RangeIndex(len(sample_predictions))
        selected_positions = row_indices[
            :: max(1, len(sample_predictions) // line_count)
        ][:line_count]
        for row in sample_predictions.iloc[selected_positions].itertuples():
            axis.plot(
                (row.actual_x, row.predicted_x),
                (row.actual_y, row.predicted_y),
                color="gray",
                alpha=0.2,
                linewidth=0.6,
                zorder=0,
            )

    axis.set(
        title="Actual vs Predicted Indoor Locations",
        xlabel="X (EPSG:3857 meters)",
        ylabel="Y (EPSG:3857 meters)",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save_figure(figure, Path(output_path))


def plot_localization_error(
    sample_predictions: pd.DataFrame,
    output_path: str | Path = REPORTS_DIR / "localization_error.png",
) -> Path:
    """Plot geodesic localization errors, requiring errors in meters."""
    _validate_location_columns(sample_predictions, ("geodesic_error_m",))
    errors = pd.to_numeric(sample_predictions["geodesic_error_m"], errors="raise")
    if not np.isfinite(errors.to_numpy(dtype=float)).all() or (errors < 0).any():
        raise ValueError("Geodesic errors must be non-negative finite meter values.")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(errors, bins="auto", color="#4c78a8", edgecolor="white")
    axis.axvline(
        errors.median(),
        color=_PREDICTED_COLOR,
        linestyle="--",
        label=f"Median: {errors.median():.2f} m",
    )
    axis.set(
        title="Validation Localization Error",
        xlabel="WGS84 geodesic error (meters)",
        ylabel="Validation samples",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, Path(output_path))


def plot_geographic_positions(
    sample_predictions: pd.DataFrame,
    output_path: str | Path = REPORTS_DIR / "geographic_positions.png",
) -> Path:
    """Plot actual and predicted positions in WGS84 longitude/latitude."""
    _validate_location_columns(
        sample_predictions,
        (
            "actual_latitude",
            "actual_longitude",
            "predicted_latitude",
            "predicted_longitude",
        ),
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        sample_predictions["actual_longitude"],
        sample_predictions["actual_latitude"],
        s=20,
        alpha=0.65,
        color=_ACTUAL_COLOR,
        label="Actual location",
    )
    axis.scatter(
        sample_predictions["predicted_longitude"],
        sample_predictions["predicted_latitude"],
        s=22,
        alpha=0.65,
        color=_PREDICTED_COLOR,
        marker="x",
        label="Predicted location",
    )
    axis.set(
        title="Validation Positions (WGS84)",
        xlabel="Longitude (degrees)",
        ylabel="Latitude (degrees)",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save_figure(figure, Path(output_path))
