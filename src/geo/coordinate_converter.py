"""Coordinate conversion for UJIIndoorLoc projected coordinates."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from pyproj import Transformer

CoordinateValues: TypeAlias = float | np.ndarray

# UJIIndoorLoc does not identify the EPSG code in its UCI dataset description.
# The recorded coordinates (e.g. x=-7541.2643, y=4864920.7782) transform from
# EPSG:3857 to (lat=39.9929702, lon=-0.0677443), inside the mapped Universitat
# Jaume I campus boundary. The corresponding EPSG:3395 result is about 40.1825
# degrees north, outside the campus. Reference sources are documented in
# README.md; the UCI page itself does not name an EPSG code.
DATASET_PROJECTED_CRS = "EPSG:3857"
WGS84_GEOGRAPHIC_CRS = "EPSG:4326"

_PROJECTED_TO_WGS84 = Transformer.from_crs(
    DATASET_PROJECTED_CRS,
    WGS84_GEOGRAPHIC_CRS,
    always_xy=True,
)
_WGS84_TO_PROJECTED = Transformer.from_crs(
    WGS84_GEOGRAPHIC_CRS,
    DATASET_PROJECTED_CRS,
    always_xy=True,
)


def _as_output(value: float | np.ndarray) -> CoordinateValues:
    """Return scalar transform results as floats and arrays as NumPy arrays."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def projected_to_latlon(
    x: CoordinateValues,
    y: CoordinateValues,
) -> tuple[CoordinateValues, CoordinateValues]:
    """Convert UJIIndoorLoc EPSG:3857 x/y values to WGS84 latitude/longitude.

    Args:
        x: Projected easting in meters, scalar or array-like.
        y: Projected northing in meters, scalar or array-like.

    Returns:
        ``(latitude, longitude)`` in WGS84 decimal degrees, as floats for
        scalar input or NumPy arrays for array input.

    Raises:
        ValueError: If coordinates contain non-finite values or cannot be
            transformed by PROJ.
    """
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Projected coordinates must contain only finite values.")

    longitude, latitude = _PROJECTED_TO_WGS84.transform(
        x_values,
        y_values,
        errcheck=True,
    )
    return _as_output(latitude), _as_output(longitude)


def latlon_to_projected(
    latitude: CoordinateValues,
    longitude: CoordinateValues,
) -> tuple[CoordinateValues, CoordinateValues]:
    """Convert WGS84 decimal-degree latitude/longitude to UJIIndoorLoc x/y.

    Args:
        latitude: WGS84 latitude in decimal degrees, scalar or array-like.
        longitude: WGS84 longitude in decimal degrees, scalar or array-like.

    Returns:
        ``(x, y)`` in EPSG:3857 meters.

    Raises:
        ValueError: If coordinates contain non-finite values or cannot be
            transformed by PROJ.
    """
    latitude_values = np.asarray(latitude, dtype=np.float64)
    longitude_values = np.asarray(longitude, dtype=np.float64)
    if not np.isfinite(latitude_values).all() or not np.isfinite(
        longitude_values
    ).all():
        raise ValueError("Latitude and longitude must contain only finite values.")

    x, y = _WGS84_TO_PROJECTED.transform(
        longitude_values,
        latitude_values,
        errcheck=True,
    )
    return _as_output(x), _as_output(y)
