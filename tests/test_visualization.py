"""Tests for indoor and geographic location visualizations."""

import pandas as pd
import pytest

from src.visualization.location_plot import (
    plot_actual_indoor_positions,
    plot_actual_vs_predicted_locations,
    plot_geographic_positions,
    plot_localization_error,
    plot_predicted_indoor_positions,
)
from src.visualization.map_visualization import (
    create_individual_location_map,
    visualize_validation_predictions,
)


def make_sample_predictions() -> pd.DataFrame:
    """Create two actual/predicted position pairs for visualization tests."""
    return pd.DataFrame(
        {
            "actual_building": [0, 1],
            "predicted_building": [0, 1],
            "actual_x": [-7541.0, -7510.0],
            "actual_y": [4_864_920.0, 4_864_890.0],
            "predicted_x": [-7540.0, -7512.0],
            "predicted_y": [4_864_921.0, 4_864_888.0],
            "actual_latitude": [39.99297, 39.99270],
            "actual_longitude": [-0.06774, -0.06750],
            "predicted_latitude": [39.99298, 39.99268],
            "predicted_longitude": [-0.06773, -0.06752],
            "geodesic_error_m": [1.4, 2.5],
        }
    )


@pytest.mark.parametrize(
    "plot_function,filename",
    [
        (plot_actual_indoor_positions, "actual.png"),
        (plot_predicted_indoor_positions, "predicted.png"),
        (plot_actual_vs_predicted_locations, "comparison.png"),
        (plot_localization_error, "error.png"),
        (plot_geographic_positions, "geographic.png"),
    ],
)
def test_location_plot_functions_write_png(
    plot_function,
    filename: str,
    tmp_path,
) -> None:
    """Each requested static visualization is saved as a PNG."""
    output_path = tmp_path / filename

    actual_path = plot_function(make_sample_predictions(), output_path)

    assert actual_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_individual_map_contains_actual_predicted_line_and_error(tmp_path) -> None:
    """HTML map distinguishes locations and associates their geodesic error."""
    output_path = tmp_path / "individual.html"

    result = create_individual_location_map(
        make_sample_predictions(),
        sample_index=0,
        output_path=output_path,
    )

    html = output_path.read_text(encoding="utf-8")
    assert result == output_path
    assert "Actual location" in html
    assert "Predicted location" in html
    assert "Localization error: 1.400 m" in html
    assert "polyline" in html


def test_visualize_validation_predictions_creates_all_outputs(tmp_path) -> None:
    """Create all requested PNGs and an HTML individual sample map."""
    outputs = visualize_validation_predictions(
        make_sample_predictions(),
        reports_dir=tmp_path,
        sample_index=1,
    )

    assert set(outputs) == {
        "actual_indoor_positions",
        "predicted_indoor_positions",
        "actual_vs_predicted_locations",
        "localization_error",
        "geographic_positions",
        "individual_location_map",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())


def test_location_visualizations_reject_missing_columns(tmp_path) -> None:
    """Reject data that cannot support all markers instead of plotting partials."""
    with pytest.raises(ValueError, match="actual_x"):
        plot_actual_vs_predicted_locations(
            pd.DataFrame({"actual_y": [1]}),
            tmp_path / "incomplete.png",
        )
    with pytest.raises(ValueError, match="missing columns"):
        create_individual_location_map(
            pd.DataFrame({"actual_x": [1]}),
            output_path=tmp_path / "incomplete.html",
        )


def test_individual_map_checks_sample_index() -> None:
    """Report an invalid sample index clearly."""
    with pytest.raises(IndexError, match="out of range"):
        create_individual_location_map(make_sample_predictions(), sample_index=2)
