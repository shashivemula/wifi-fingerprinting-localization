"""Tests for UJIIndoorLoc projected-coordinate conversion."""

import numpy as np
import pytest

from src.geo.coordinate_converter import (
    DATASET_PROJECTED_CRS,
    latlon_to_projected,
    projected_to_latlon,
)


def test_known_uji_coordinate_transforms_to_uji_campus_reference() -> None:
    """Check a recorded UJI point falls at the known Castellon campus."""
    latitude, longitude = projected_to_latlon(-7541.2643, 4864920.7782)

    assert DATASET_PROJECTED_CRS == "EPSG:3857"
    assert latitude == pytest.approx(39.9929702, abs=1e-6)
    assert longitude == pytest.approx(-0.0677443, abs=1e-6)


def test_projected_geographic_conversion_round_trip() -> None:
    """Recover projected coordinates after a WGS84 round trip."""
    x = np.array([-7541.2643, -7515.916799])
    y = np.array([4864920.7782, 4864890.0])

    latitude, longitude = projected_to_latlon(x, y)
    recovered_x, recovered_y = latlon_to_projected(latitude, longitude)

    np.testing.assert_allclose(recovered_x, x, atol=1e-6)
    np.testing.assert_allclose(recovered_y, y, atol=1e-6)


def test_scalar_transform_returns_float_values() -> None:
    """Return geographic scalar coordinates in latitude, longitude order."""
    latitude, longitude = projected_to_latlon(-7541.2643, 4864920.7782)

    assert isinstance(latitude, float)
    assert isinstance(longitude, float)


def test_non_finite_coordinates_are_rejected() -> None:
    """Reject invalid input rather than returning success-shaped coordinates."""
    with pytest.raises(ValueError, match="finite"):
        projected_to_latlon(float("nan"), 4864920.7782)
    with pytest.raises(ValueError, match="finite"):
        latlon_to_projected(39.9, float("inf"))
