"""Schema and data-quality validation for UJIIndoorLoc DataFrames."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import RSSI_FEATURE_COLUMNS

TARGET_COLUMNS = (
    "LONGITUDE",
    "LATITUDE",
    "FLOOR",
    "BUILDINGID",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
)
NUMERIC_TARGET_COLUMNS = tuple(
    column for column in TARGET_COLUMNS if column != "TIMESTAMP"
)
RSSI_MIN_VALUE = -104
RSSI_MAX_VALUE = 0
RSSI_MISSING_VALUE = 100


@dataclass
class ValidationReport:
    """Collected schema, data-quality, and observed-value information."""

    dataset_name: str
    shape: tuple[int, int]
    column_names: tuple[str, ...]
    rssi_columns_present: tuple[str, ...]
    missing_rssi_columns: tuple[str, ...]
    target_columns_present: tuple[str, ...]
    missing_target_columns: tuple[str, ...]
    duplicate_column_names: tuple[str, ...]
    dtype_counts: dict[str, int]
    missing_values: dict[str, int]
    duplicate_rows: int
    rssi_value_range: tuple[float, float] | None
    rssi_missing_sentinel_values: int
    invalid_rssi_values: int
    coordinate_ranges: dict[str, tuple[float, float] | None]
    category_values: dict[str, tuple[int | float, ...]]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether no validation errors were found."""
        return not self.errors

    def to_text(self) -> str:
        """Format the report as a readable multi-line summary."""
        status = "PASS" if self.is_valid else "FAIL"
        if self.warnings and self.is_valid:
            status += " (with warnings)"

        lines = [
            f"Validation report: {self.dataset_name}",
            f"Status: {status}",
            f"Dataset shape: {self.shape[0]} rows x {self.shape[1]} columns",
            (
                f"Column names: {len(self.column_names)} names; "
                f"duplicate names={list(self.duplicate_column_names) or 'none'}"
            ),
            (
                f"RSSI columns: {len(self.rssi_columns_present)}/"
                f"{len(RSSI_FEATURE_COLUMNS)} present"
            ),
            f"Missing RSSI columns: {list(self.missing_rssi_columns) or 'none'}",
            (
                f"Target columns: {len(self.target_columns_present)}/"
                f"{len(TARGET_COLUMNS)} present"
            ),
            f"Missing target columns: {list(self.missing_target_columns) or 'none'}",
            f"Data types: {self.dtype_counts}",
            (
                "Missing values: "
                f"{sum(self.missing_values.values())} cells across "
                f"{len(self.missing_values)} columns"
            ),
            f"Exact duplicate rows: {self.duplicate_rows}",
            (
                "RSSI signal range (excluding missing sentinel): "
                f"{self.rssi_value_range if self.rssi_value_range is not None else 'unavailable'}"
            ),
            f"RSSI missing-signal sentinel ({RSSI_MISSING_VALUE}) count: "
            f"{self.rssi_missing_sentinel_values}",
            (
                "Invalid RSSI values outside "
                f"[{RSSI_MIN_VALUE}, {RSSI_MAX_VALUE}] or "
                f"{RSSI_MISSING_VALUE}: {self.invalid_rssi_values}"
            ),
            "Coordinate observed ranges (source units; no CRS bounds assumed):",
        ]
        for column, value_range in self.coordinate_ranges.items():
            lines.append(f"  {column}: {value_range or 'unavailable'}")
        for column, values in self.category_values.items():
            lines.append(f"  {column} values: {list(values)}")

        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {message}" for message in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {message}" for message in self.warnings)
        return "\n".join(lines)


def _finite_range(values: pd.Series) -> tuple[float, float] | None:
    """Return the min/max of finite numeric values, if any."""
    finite_values = values.dropna()
    if finite_values.empty:
        return None
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.empty:
        return None
    return float(finite_values.min()), float(finite_values.max())


def _normalized_bounds(
    coordinate_bounds: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    """Validate and normalize optional per-coordinate bounds."""
    if coordinate_bounds is None:
        return {}

    allowed_columns = {"LONGITUDE", "LATITUDE"}
    unknown_columns = set(coordinate_bounds) - allowed_columns
    if unknown_columns:
        raise ValueError(
            "Coordinate bounds may only be supplied for LONGITUDE and LATITUDE; "
            f"received: {sorted(unknown_columns)}."
        )

    normalized: dict[str, tuple[float, float]] = {}
    for column, bounds in coordinate_bounds.items():
        if len(bounds) != 2:
            raise ValueError(f"Bounds for {column} must contain a minimum and maximum.")
        minimum, maximum = bounds
        if not isfinite(minimum) or not isfinite(maximum) or minimum > maximum:
            raise ValueError(
                f"Bounds for {column} must be finite and ordered minimum-to-maximum."
            )
        normalized[column] = (float(minimum), float(maximum))
    return normalized


def validate_dataframe(
    dataframe: pd.DataFrame,
    dataset_name: str = "dataset",
    coordinate_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> ValidationReport:
    """Validate the UJIIndoorLoc schema and summarize data quality.

    RSSI values from ``-104`` through ``0`` and the dataset's ``100`` missing
    signal sentinel are accepted. Coordinates are summarized in their source
    units; optional bounds can be supplied when a coordinate reference system
    and valid extent are known.

    Args:
        dataframe: DataFrame to inspect. This function does not modify it.
        dataset_name: Human-readable name shown in the report.
        coordinate_bounds: Optional inclusive bounds for LONGITUDE and/or
            LATITUDE, expressed in the CSV's coordinate units.

    Returns:
        A report containing validation errors, warnings, and observed ranges.
    """
    bounds = _normalized_bounds(coordinate_bounds)
    column_names = tuple(str(column) for column in dataframe.columns)
    duplicate_column_names = tuple(
        sorted(
            {
                str(column)
                for column in dataframe.columns[dataframe.columns.duplicated()]
            }
        )
    )
    rssi_columns_present = tuple(
        column for column in RSSI_FEATURE_COLUMNS if column in dataframe.columns
    )
    missing_rssi_columns = tuple(
        column for column in RSSI_FEATURE_COLUMNS if column not in dataframe.columns
    )
    target_columns_present = tuple(
        column for column in TARGET_COLUMNS if column in dataframe.columns
    )
    missing_target_columns = tuple(
        column for column in TARGET_COLUMNS if column not in dataframe.columns
    )
    dtype_counts = {
        str(dtype): int(count)
        for dtype, count in dataframe.dtypes.astype(str).value_counts().items()
    }
    missing_by_column = dataframe.isna().sum()
    missing_values = {
        str(column): int(count)
        for column, count in missing_by_column.items()
        if count > 0
    }
    duplicate_rows = int(dataframe.duplicated().sum())
    errors: list[str] = []
    warnings: list[str] = []

    if duplicate_column_names:
        errors.append(f"Duplicate column names found: {list(duplicate_column_names)}.")
    if missing_rssi_columns:
        errors.append(
            f"Missing {len(missing_rssi_columns)} required RSSI columns, including "
            f"{list(missing_rssi_columns[:5])}."
        )
    if missing_target_columns:
        errors.append(f"Missing target columns: {list(missing_target_columns)}.")

    if missing_values:
        warnings.append(
            f"Found {sum(missing_values.values())} missing cells in "
            f"{len(missing_values)} columns."
        )
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} exact duplicate rows.")

    invalid_rssi_values = 0
    rssi_minimum: float | None = None
    rssi_maximum: float | None = None
    rssi_missing_sentinel_values = 0
    numeric_required_columns = set(RSSI_FEATURE_COLUMNS) | set(
        NUMERIC_TARGET_COLUMNS
    )
    for column in dataframe.columns:
        column_name = str(column)
        if column_name not in numeric_required_columns:
            continue
        if column_name in duplicate_column_names:
            continue

        series = dataframe[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if not pd.api.types.is_numeric_dtype(series.dtype):
            errors.append(
                f"Column {column_name} must have a numeric data type; "
                f"found {series.dtype}."
            )
        nonnumeric_count = int((numeric_series.isna() & series.notna()).sum())
        if nonnumeric_count:
            errors.append(
                f"Column {column_name} contains {nonnumeric_count} non-numeric values."
            )

        if column_name in RSSI_FEATURE_COLUMNS:
            rssi_missing_sentinel_values += int(
                numeric_series.eq(RSSI_MISSING_VALUE).sum()
            )
            observed = numeric_series[
                numeric_series.notna() & numeric_series.ne(RSSI_MISSING_VALUE)
            ]
            if not observed.empty:
                column_minimum = float(observed.min())
                column_maximum = float(observed.max())
                rssi_minimum = (
                    column_minimum
                    if rssi_minimum is None
                    else min(rssi_minimum, column_minimum)
                )
                rssi_maximum = (
                    column_maximum
                    if rssi_maximum is None
                    else max(rssi_maximum, column_maximum)
                )
            valid_rssi = (
                numeric_series.eq(RSSI_MISSING_VALUE)
                | numeric_series.between(RSSI_MIN_VALUE, RSSI_MAX_VALUE)
                | numeric_series.isna()
            )
            invalid_rssi_values += int((~valid_rssi).sum())

    rssi_value_range = (
        (rssi_minimum, rssi_maximum)
        if rssi_minimum is not None and rssi_maximum is not None
        else None
    )
    if invalid_rssi_values:
        errors.append(
            f"Found {invalid_rssi_values} RSSI values outside "
            f"[{RSSI_MIN_VALUE}, {RSSI_MAX_VALUE}] and not equal to "
            f"{RSSI_MISSING_VALUE}."
        )

    coordinate_ranges: dict[str, tuple[float, float] | None] = {}
    for column in ("LONGITUDE", "LATITUDE"):
        if column not in dataframe.columns:
            coordinate_ranges[column] = None
            continue
        if column in duplicate_column_names:
            coordinate_ranges[column] = None
            continue

        series = dataframe[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        coordinate_ranges[column] = _finite_range(numeric_series)
        nonfinite_mask = numeric_series.notna() & ~np.isfinite(numeric_series)
        if nonfinite_mask.any():
            errors.append(
                f"Coordinate column {column} contains "
                f"{int(nonfinite_mask.sum())} non-finite values."
            )
        nonnumeric_count = int((numeric_series.isna() & series.notna()).sum())
        if nonnumeric_count:
            errors.append(
                f"Coordinate column {column} contains "
                f"{nonnumeric_count} non-numeric values."
            )
        if column in bounds and coordinate_ranges[column] is not None:
            minimum, maximum = bounds[column]
            observed_minimum, observed_maximum = coordinate_ranges[column]
            if observed_minimum < minimum or observed_maximum > maximum:
                errors.append(
                    f"Coordinate column {column} has observed range "
                    f"{coordinate_ranges[column]}, outside configured bounds "
                    f"{bounds[column]}."
                )

    category_values: dict[str, tuple[int | float, ...]] = {}
    for column in ("BUILDINGID", "FLOOR"):
        if column not in dataframe.columns:
            category_values[column] = ()
            continue
        if column in duplicate_column_names:
            category_values[column] = ()
            continue

        series = dataframe[column]
        numeric_series = pd.to_numeric(series, errors="coerce")
        values = numeric_series.dropna()
        unique_values = sorted(values.unique().tolist())
        category_values[column] = tuple(
            int(value) if float(value).is_integer() else float(value)
            for value in unique_values
        )
        if values.lt(0).any():
            errors.append(f"Column {column} contains negative values.")
        if not values.mod(1).eq(0).all():
            errors.append(f"Column {column} contains non-integer values.")

    return ValidationReport(
        dataset_name=dataset_name,
        shape=(int(dataframe.shape[0]), int(dataframe.shape[1])),
        column_names=column_names,
        rssi_columns_present=rssi_columns_present,
        missing_rssi_columns=missing_rssi_columns,
        target_columns_present=target_columns_present,
        missing_target_columns=missing_target_columns,
        duplicate_column_names=duplicate_column_names,
        dtype_counts=dtype_counts,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        rssi_value_range=rssi_value_range,
        rssi_missing_sentinel_values=rssi_missing_sentinel_values,
        invalid_rssi_values=invalid_rssi_values,
        coordinate_ranges=coordinate_ranges,
        category_values=category_values,
        errors=errors,
        warnings=warnings,
    )
