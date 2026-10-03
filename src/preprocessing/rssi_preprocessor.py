"""Reusable preprocessing for UJIIndoorLoc RSSI fingerprints."""

from __future__ import annotations

from math import isfinite

import pandas as pd

from src.config import RSSI_FEATURE_COLUMNS

RSSI_MIN_VALUE = -104
RSSI_MAX_VALUE = 0
UJIINDOORLOC_UNAVAILABLE_VALUE = 100
DEFAULT_UNAVAILABLE_REPLACEMENT = -110.0


class RSSIPreprocessor:
    """Select, validate, and replace unavailable values in WAP fingerprints.

    The fitted preprocessor stores the canonical WAP feature order so the same
    transformation can be reused for training and inference. Unavailable RSSI
    values are the UJIIndoorLoc sentinel ``100`` or null values.
    """

    def __init__(
        self,
        unavailable_value: float = UJIINDOORLOC_UNAVAILABLE_VALUE,
        replacement_value: float = DEFAULT_UNAVAILABLE_REPLACEMENT,
    ) -> None:
        """Initialize unavailable-value handling.

        Args:
            unavailable_value: Dataset value that represents an undetected AP.
            replacement_value: Finite replacement outside the valid RSSI range.

        Raises:
            ValueError: If either configured value is non-finite or the
                replacement collides with a valid RSSI or sentinel value.
        """
        if not isfinite(unavailable_value):
            raise ValueError("unavailable_value must be finite.")
        if not isfinite(replacement_value):
            raise ValueError("replacement_value must be finite.")
        if RSSI_MIN_VALUE <= replacement_value <= RSSI_MAX_VALUE:
            raise ValueError(
                f"replacement_value must be outside the valid RSSI range "
                f"[{RSSI_MIN_VALUE}, {RSSI_MAX_VALUE}]."
            )
        if replacement_value == unavailable_value:
            raise ValueError(
                "replacement_value must differ from unavailable_value."
            )

        self.unavailable_value = float(unavailable_value)
        self.replacement_value = float(replacement_value)
        self.feature_columns_: tuple[str, ...] | None = None

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame) -> None:
        """Ensure all expected WAP columns exist exactly once."""
        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()]
        if not duplicate_columns.empty:
            duplicates = sorted({str(column) for column in duplicate_columns})
            raise ValueError(f"Input contains duplicate column names: {duplicates}.")

        missing = [
            column for column in RSSI_FEATURE_COLUMNS if column not in dataframe.columns
        ]
        if missing:
            preview = ", ".join(missing[:10])
            remainder = len(missing) - min(len(missing), 10)
            suffix = f" (and {remainder} more)" if remainder else ""
            raise ValueError(
                f"Input is missing {len(missing)} required WAP features: "
                f"{preview}{suffix}."
            )

    def fit(self, dataframe: pd.DataFrame) -> RSSIPreprocessor:
        """Validate the expected WAP features and store their canonical order.

        Target and metadata columns, if present, are ignored.

        Args:
            dataframe: Fingerprint DataFrame containing all 520 WAP features.

        Returns:
            This fitted preprocessor.
        """
        self._validate_columns(dataframe)
        self.feature_columns_ = RSSI_FEATURE_COLUMNS
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return numeric WAP fingerprints with unavailable values replaced.

        Extra columns such as labels and metadata are ignored. The result
        contains only WAP001 through WAP520 in canonical order.

        Args:
            dataframe: Fingerprint DataFrame to transform.

        Returns:
            A numeric DataFrame containing exactly the ordered WAP features.

        Raises:
            RuntimeError: If ``fit`` has not been called.
            ValueError: If expected features are missing or RSSI values are
                non-numeric or outside the accepted UJIIndoorLoc range.
        """
        if self.feature_columns_ is None:
            raise RuntimeError("RSSIPreprocessor must be fitted before transform.")
        self._validate_columns(dataframe)

        selected = dataframe.loc[:, self.feature_columns_]
        numeric = selected.apply(pd.to_numeric, errors="coerce")
        nonnumeric_mask = selected.notna() & numeric.isna()
        if nonnumeric_mask.to_numpy().any():
            invalid_columns = nonnumeric_mask.columns[
                nonnumeric_mask.any()
            ].tolist()
            raise ValueError(
                "RSSI columns contain non-numeric values: "
                f"{invalid_columns[:10]}."
            )

        unavailable_mask = numeric.isna() | numeric.eq(self.unavailable_value)
        invalid_mask = (
            ~unavailable_mask
            & (
                ~numeric.ge(RSSI_MIN_VALUE)
                | ~numeric.le(RSSI_MAX_VALUE)
                | ~numeric.map(isfinite)
            )
        )
        if invalid_mask.to_numpy().any():
            invalid_columns = invalid_mask.columns[invalid_mask.any()].tolist()
            invalid_count = int(invalid_mask.to_numpy().sum())
            raise ValueError(
                f"Found {invalid_count} invalid RSSI value(s) outside "
                f"[{RSSI_MIN_VALUE}, {RSSI_MAX_VALUE}] and not equal to "
                f"{self.unavailable_value:g}; columns: {invalid_columns[:10]}."
            )

        transformed = numeric.mask(unavailable_mask, self.replacement_value)
        transformed = transformed.astype(float)
        transformed.columns = self.feature_columns_
        return transformed

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit feature ordering and transform the supplied fingerprints."""
        return self.fit(dataframe).transform(dataframe)

    def get_feature_names_out(self) -> tuple[str, ...]:
        """Return canonical WAP column names after fitting."""
        if self.feature_columns_ is None:
            raise RuntimeError("RSSIPreprocessor must be fitted first.")
        return self.feature_columns_
