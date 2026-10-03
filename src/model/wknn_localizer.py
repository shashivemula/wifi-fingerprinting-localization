"""Weighted K-Nearest Neighbors localization using WiFi RSSI fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import MODELS_DIR, RSSI_FEATURE_COLUMNS

TARGET_COLUMNS = ("LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR")
WeightingStrategy = Literal["inverse_distance", "uniform"]
FeatureInput = pd.DataFrame | np.ndarray


@dataclass(frozen=True)
class NeighborInfo:
    """A selected training fingerprint and its contribution to a prediction."""

    index: int
    distance: float
    weight: float
    building: int
    floor: int
    x: float
    y: float


@dataclass(frozen=True)
class LocalizationPrediction:
    """Predicted building, floor, coordinates, and contributing neighbors."""

    building: int
    floor: int
    x: float
    y: float
    neighbors: tuple[NeighborInfo, ...]


class WiFiWKNNLocalizer:
    """Estimate indoor position and labels from WAP001 through WAP520.

    Input DataFrames may also contain targets or other metadata; only canonical
    WAP columns are passed to the nearest-neighbor estimator. Features are
    expected to have already been transformed with the project's fitted
    :class:`src.preprocessing.rssi_preprocessor.RSSIPreprocessor`.
    """

    def __init__(
        self,
        k: int = 5,
        distance_metric: str = "euclidean",
        weighting_strategy: WeightingStrategy = "inverse_distance",
        models_dir: str | Path = MODELS_DIR,
    ) -> None:
        """Configure the neighbor search and weighting strategy.

        Args:
            k: Number of neighbors used per prediction.
            distance_metric: Metric accepted by scikit-learn's
                ``NearestNeighbors`` (for example ``euclidean`` or
                ``manhattan``).
            weighting_strategy: Either ``inverse_distance`` or ``uniform``.
            models_dir: Default directory used when saving by filename.

        Raises:
            ValueError: If ``k`` or the weighting strategy is invalid.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if weighting_strategy not in ("inverse_distance", "uniform"):
            raise ValueError(
                "weighting_strategy must be 'inverse_distance' or 'uniform'."
            )
        if not distance_metric:
            raise ValueError("distance_metric must be a non-empty metric name.")

        self.k = k
        self.distance_metric = distance_metric
        self.weighting_strategy = weighting_strategy
        self.models_dir = Path(models_dir)
        self._neighbor_search: NearestNeighbors | None = None
        self._building_ids: np.ndarray | None = None
        self._floors: np.ndarray | None = None
        self._coordinates: np.ndarray | None = None
        self.feature_names_in_: tuple[str, ...] = RSSI_FEATURE_COLUMNS
        self.n_features_in_: int = len(RSSI_FEATURE_COLUMNS)

    @staticmethod
    def _extract_feature_matrix(features: FeatureInput) -> np.ndarray:
        """Select and validate the 520 canonical WAP feature columns."""
        if isinstance(features, pd.DataFrame):
            duplicate_columns = features.columns[features.columns.duplicated()]
            if not duplicate_columns.empty:
                duplicate_names = sorted(
                    {str(column) for column in duplicate_columns}
                )
                raise ValueError(
                    f"Input contains duplicate column names: {duplicate_names}."
                )
            missing = [
                column
                for column in RSSI_FEATURE_COLUMNS
                if column not in features.columns
            ]
            if missing:
                raise ValueError(
                    f"Input is missing {len(missing)} required WAP features: "
                    f"{', '.join(missing[:10])}."
                )
            selected = features.loc[:, RSSI_FEATURE_COLUMNS]
            try:
                matrix = selected.to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("WAP features must contain numeric RSSI values.") from exc
        elif isinstance(features, np.ndarray):
            if features.ndim != 2:
                raise ValueError("RSSI array input must be a two-dimensional matrix.")
            if features.shape[1] != len(RSSI_FEATURE_COLUMNS):
                raise ValueError(
                    f"RSSI array must have exactly {len(RSSI_FEATURE_COLUMNS)} "
                    f"columns in canonical WAP order; got {features.shape[1]}."
                )
            try:
                matrix = features.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("WAP features must contain numeric RSSI values.") from exc
        else:
            raise TypeError("features must be a pandas DataFrame or NumPy array.")

        if matrix.shape[0] == 0:
            raise ValueError("RSSI input must contain at least one sample.")
        if not np.isfinite(matrix).all():
            raise ValueError(
                "RSSI features contain missing or non-finite values; preprocess "
                "the fingerprints before fitting or prediction."
            )
        return matrix

    @staticmethod
    def _extract_targets(
        features: pd.DataFrame,
        targets: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate target columns and return buildings, floors, and x/y."""
        if targets is None:
            target_frame = features
        else:
            target_frame = targets

        duplicate_columns = target_frame.columns[target_frame.columns.duplicated()]
        if not duplicate_columns.empty:
            duplicate_names = sorted({str(column) for column in duplicate_columns})
            raise ValueError(
                f"Target data contains duplicate column names: {duplicate_names}."
            )
        missing = [
            column for column in TARGET_COLUMNS if column not in target_frame.columns
        ]
        if missing:
            raise ValueError(f"Target data is missing required columns: {missing}.")
        if len(features) != len(target_frame):
            raise ValueError(
                "Feature and target data must contain the same number of samples."
            )

        target_values: dict[str, np.ndarray] = {}
        for column in TARGET_COLUMNS:
            try:
                values = pd.to_numeric(
                    target_frame[column],
                    errors="raise",
                ).to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Target column {column} must be numeric.") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Target column {column} contains non-finite values.")
            target_values[column] = values

        buildings = target_values["BUILDINGID"]
        floors = target_values["FLOOR"]
        for column, values in (("BUILDINGID", buildings), ("FLOOR", floors)):
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"Target column {column} must contain integers.")
            if (values < 0).any():
                raise ValueError(f"Target column {column} cannot contain negatives.")

        return (
            buildings.astype(np.int64),
            floors.astype(np.int64),
            np.column_stack(
                (target_values["LONGITUDE"], target_values["LATITUDE"])
            ),
        )

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame | None = None,
    ) -> WiFiWKNNLocalizer:
        """Fit neighbor search from RSSI fingerprints and localization targets.

        Pass either a single complete DataFrame, or a WAP feature DataFrame and
        a separate target DataFrame. When targets are supplied separately,
        their row order must correspond to the feature row order.

        Args:
            features: DataFrame containing the 520 WAP features and optionally
                all target columns.
            targets: Optional DataFrame containing LONGITUDE, LATITUDE,
                BUILDINGID, and FLOOR.

        Returns:
            This fitted localizer.

        Raises:
            ValueError: For missing features/targets, invalid values, or a
                neighbor count larger than the training sample count.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("fit features must be provided as a pandas DataFrame.")
        matrix = self._extract_feature_matrix(features)
        buildings, floors, coordinates = self._extract_targets(features, targets)
        if self.k > matrix.shape[0]:
            raise ValueError(
                f"k ({self.k}) cannot exceed the number of training samples "
                f"({matrix.shape[0]})."
            )

        neighbor_search = NearestNeighbors(
            n_neighbors=self.k,
            metric=self.distance_metric,
            algorithm="auto",
        )
        neighbor_search.fit(matrix)

        self._neighbor_search = neighbor_search
        self._building_ids = buildings
        self._floors = floors
        self._coordinates = coordinates
        return self

    def _require_fitted(
        self,
    ) -> tuple[NearestNeighbors, np.ndarray, np.ndarray, np.ndarray]:
        """Return fitted estimator state or raise a clear error."""
        if (
            self._neighbor_search is None
            or self._building_ids is None
            or self._floors is None
            or self._coordinates is None
        ):
            raise RuntimeError("WiFiWKNNLocalizer must be fitted before prediction.")
        return (
            self._neighbor_search,
            self._building_ids,
            self._floors,
            self._coordinates,
        )

    def _calculate_weights(self, distances: np.ndarray) -> np.ndarray:
        """Calculate normalized neighbor weights, safely handling exact matches."""
        if self.weighting_strategy == "uniform":
            weights = np.ones_like(distances, dtype=np.float64)
        else:
            zero_distance_mask = distances == 0
            if zero_distance_mask.any():
                weights = zero_distance_mask.astype(np.float64)
            else:
                weights = 1.0 / distances
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise RuntimeError("Neighbor distances produced invalid weights.")
        return weights / total_weight

    @staticmethod
    def _weighted_vote(
        labels: np.ndarray,
        weights: np.ndarray,
        distances: np.ndarray,
    ) -> int:
        """Choose the class with largest weight, resolving ties by closest row."""
        total_weights: dict[int, float] = {}
        closest_distances: dict[int, float] = {}
        for label, weight, distance in zip(labels, weights, distances):
            class_label = int(label)
            total_weights[class_label] = (
                total_weights.get(class_label, 0.0) + float(weight)
            )
            closest_distances[class_label] = min(
                closest_distances.get(class_label, float("inf")),
                float(distance),
            )
        return min(
            total_weights,
            key=lambda label: (
                -total_weights[label],
                closest_distances[label],
                label,
            ),
        )

    def predict(self, features: FeatureInput) -> list[LocalizationPrediction]:
        """Predict labels and indoor coordinates for one or more fingerprints.

        Args:
            features: DataFrame containing all WAP columns (extra metadata is
                ignored), or a two-dimensional NumPy array in canonical WAP
                order.

        Returns:
            One structured prediction per input row, including neighbor details.
        """
        neighbor_search, building_ids, floors, coordinates = self._require_fitted()
        matrix = self._extract_feature_matrix(features)
        distances_by_row, indices_by_row = neighbor_search.kneighbors(
            matrix,
            n_neighbors=self.k,
            return_distance=True,
        )

        predictions: list[LocalizationPrediction] = []
        for distances, indices in zip(distances_by_row, indices_by_row):
            weights = self._calculate_weights(distances)
            selected_buildings = building_ids[indices]
            selected_floors = floors[indices]
            selected_coordinates = coordinates[indices]
            predicted_coordinate = np.average(
                selected_coordinates,
                axis=0,
                weights=weights,
            )
            neighbors = tuple(
                NeighborInfo(
                    index=int(index),
                    distance=float(distance),
                    weight=float(weight),
                    building=int(building_ids[index]),
                    floor=int(floors[index]),
                    x=float(coordinates[index, 0]),
                    y=float(coordinates[index, 1]),
                )
                for index, distance, weight in zip(indices, distances, weights)
            )
            predictions.append(
                LocalizationPrediction(
                    building=self._weighted_vote(
                        selected_buildings,
                        weights,
                        distances,
                    ),
                    floor=self._weighted_vote(selected_floors, weights, distances),
                    x=float(predicted_coordinate[0]),
                    y=float(predicted_coordinate[1]),
                    neighbors=neighbors,
                )
            )
        return predictions

    def predict_single(self, features: pd.DataFrame | np.ndarray) -> LocalizationPrediction:
        """Predict one fingerprint and return one structured result.

        Args:
            features: A single-row WAP DataFrame or a one-dimensional 520-value
                NumPy array.
        """
        if isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)
        elif isinstance(features, pd.DataFrame) and len(features) != 1:
            raise ValueError("predict_single requires exactly one fingerprint row.")
        predictions = self.predict(features)
        if len(predictions) != 1:
            raise ValueError("predict_single requires exactly one fingerprint row.")
        return predictions[0]

    def save(self, path: str | Path) -> Path:
        """Serialize this fitted model to a joblib file.

        Args:
            path: Destination file. Relative filenames are placed under
                ``models_dir``.

        Returns:
            The resolved path written.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        destination = Path(path)
        if not destination.is_absolute():
            destination = self.models_dir / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> WiFiWKNNLocalizer:
        """Load a fitted model serialized with :meth:`save`.

        Args:
            path: Model file path.

        Returns:
            The fitted localizer instance.

        Raises:
            TypeError: If the serialized object is not a localizer.
        """
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Serialized object at {path} is not a {cls.__name__}."
            )
        return loaded
