"""Streamlit interface for WiFi WKNN indoor localization inference."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Geod

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RSSI_FEATURE_COLUMNS
from src.data.loader import DatasetLoadError, load_validation_data
from src.geo.coordinate_converter import projected_to_latlon
from src.model.wknn_localizer import LocalizationPrediction
from src.prediction.predictor import WiFiFingerprintPredictor
from src.preprocessing.rssi_preprocessor import (
    RSSI_MAX_VALUE,
    RSSI_MIN_VALUE,
    UJIINDOORLOC_UNAVAILABLE_VALUE,
)

MODEL_PATH = MODELS_DIR / "wknn_localizer.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "rssi_preprocessor.pkl"
GROUND_TRUTH_COLUMNS = ("LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR")
WGS84_GEOD = Geod(ellps="WGS84")
INPUT_MODES = (
    "UJIIndoorLoc Validation Sample",
    "Upload WiFi Fingerprint",
    "Manual RSSI",
)


@dataclass(frozen=True)
class InferenceResult:
    """Prediction and display metadata for one WAP-only fingerprint."""

    prediction: LocalizationPrediction
    latitude: float
    longitude: float
    detected_wap_count: int


@dataclass(frozen=True)
class BuildingFloorLocations:
    """Training locations and local-coordinate origin for one building/floor."""

    origin_x: float
    origin_y: float
    local_x: np.ndarray
    local_y: np.ndarray


@st.cache_resource(show_spinner="Loading saved WKNN model...")
def load_predictor(model_path: str, preprocessor_path: str) -> WiFiFingerprintPredictor:
    """Load saved inference artifacts once per Streamlit process."""
    return WiFiFingerprintPredictor(model_path, preprocessor_path)


@st.cache_data(show_spinner="Loading UJIIndoorLoc validation samples...")
def load_validation_samples() -> pd.DataFrame:
    """Load the official validation CSV for the explicit demonstration mode."""
    return load_validation_data()


def validate_upload(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate a CSV and return only its WAP columns in canonical order."""
    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no samples.")
    if dataframe.columns.duplicated().any():
        duplicate_columns = sorted(
            {str(column) for column in dataframe.columns[dataframe.columns.duplicated()]}
        )
        raise ValueError(f"The uploaded CSV has duplicate columns: {duplicate_columns}.")

    missing = [
        column for column in RSSI_FEATURE_COLUMNS if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"The uploaded CSV is missing {len(missing)} required WAP columns: "
            f"{', '.join(missing[:10])}."
        )

    return dataframe.loc[:, RSSI_FEATURE_COLUMNS]


def build_manual_fingerprint(entries: pd.DataFrame) -> pd.DataFrame:
    """Build a complete WAP row from only the access points the user detected."""
    required_columns = ("WAP ID", "RSSI")
    missing_columns = [
        column for column in required_columns if column not in entries.columns
    ]
    if missing_columns:
        raise ValueError(f"Manual input is missing columns: {missing_columns}.")

    rows = entries.loc[:, required_columns].dropna(how="all")
    if rows.empty:
        raise ValueError("Add at least one detected WAP and its RSSI value.")

    fingerprint = {
        column: UJIINDOORLOC_UNAVAILABLE_VALUE for column in RSSI_FEATURE_COLUMNS
    }
    seen_waps: set[str] = set()
    for row_number, (_, row) in enumerate(rows.iterrows(), start=1):
        wap_id = row["WAP ID"]
        rssi = row["RSSI"]
        if pd.isna(wap_id) or pd.isna(rssi):
            raise ValueError(
                f"Manual input row {row_number} must include both WAP ID and RSSI."
            )
        wap_id = str(wap_id)
        if wap_id not in RSSI_FEATURE_COLUMNS:
            raise ValueError(f"{wap_id!r} is not a valid WAP001-WAP520 feature.")
        if wap_id in seen_waps:
            raise ValueError(f"{wap_id} is listed more than once.")
        try:
            numeric_rssi = float(rssi)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RSSI for {wap_id} must be numeric.") from exc
        if (
            not np.isfinite(numeric_rssi)
            or numeric_rssi < RSSI_MIN_VALUE
            or numeric_rssi > RSSI_MAX_VALUE
        ):
            raise ValueError(
                f"RSSI for {wap_id} must be between {RSSI_MIN_VALUE} and "
                f"{RSSI_MAX_VALUE} dBm; 100 means not detected."
            )
        fingerprint[wap_id] = numeric_rssi
        seen_waps.add(wap_id)

    return pd.DataFrame([fingerprint], columns=RSSI_FEATURE_COLUMNS)


def _get_ground_truth(row: pd.Series) -> dict[str, int | float] | None:
    """Read complete numeric labels for display only, after inference."""
    if any(column not in row.index for column in GROUND_TRUTH_COLUMNS):
        return None
    try:
        values = {
            column: float(pd.to_numeric(row[column], errors="raise"))
            for column in GROUND_TRUTH_COLUMNS
        }
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in values.values()):
        return None
    if any(
        not values[column].is_integer()
        for column in ("BUILDINGID", "FLOOR")
    ):
        return None
    return {
        "x": values["LONGITUDE"],
        "y": values["LATITUDE"],
        "building": int(values["BUILDINGID"]),
        "floor": int(values["FLOOR"]),
    }


def _detected_wap_count(wap_frame: pd.DataFrame) -> int:
    """Count finite detected RSSI readings, excluding sentinel 100 and nulls."""
    numeric = wap_frame.loc[:, RSSI_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return int(
        (
            numeric.ge(RSSI_MIN_VALUE)
            & numeric.le(RSSI_MAX_VALUE)
            & np.isfinite(numeric)
        )
        .sum(axis=1)
        .iloc[0]
    )


def _predict_fingerprint(
    predictor: WiFiFingerprintPredictor,
    fingerprint: pd.DataFrame,
) -> InferenceResult:
    """Preprocess and predict using WAP features only."""
    wap_frame = validate_upload(fingerprint)
    transformed = predictor.preprocessor.transform(wap_frame)
    prediction = predictor.model.predict_single(transformed)
    latitude, longitude = projected_to_latlon(prediction.x, prediction.y)
    return InferenceResult(
        prediction=prediction,
        latitude=float(latitude),
        longitude=float(longitude),
        detected_wap_count=_detected_wap_count(wap_frame),
    )


def _building_floor_locations(
    predictor: WiFiFingerprintPredictor,
    building: int,
    floor: int,
) -> BuildingFloorLocations:
    """Get saved training locations for a building/floor and localize origin."""
    model = predictor.model
    coordinates = model._coordinates
    buildings = model._building_ids
    floors = model._floors
    if coordinates is None or buildings is None or floors is None:
        raise RuntimeError("Saved WKNN model is missing its fitted location data.")

    mask = (buildings == building) & (floors == floor)
    locations = coordinates[mask]
    if locations.size == 0:
        raise RuntimeError(
            f"The saved model contains no training locations for building "
            f"{building}, floor {floor}."
        )
    origin_x = float(locations[:, 0].min())
    origin_y = float(locations[:, 1].min())
    local_locations = locations - np.array([origin_x, origin_y])
    return BuildingFloorLocations(
        origin_x=origin_x,
        origin_y=origin_y,
        local_x=local_locations[:, 0],
        local_y=local_locations[:, 1],
    )


def _local_coordinates(
    x: float,
    y: float,
    locations: BuildingFloorLocations,
) -> tuple[float, float]:
    """Convert projected model coordinates to local floor-relative offsets."""
    return x - locations.origin_x, y - locations.origin_y


def _distance_meters(
    actual_x: float,
    actual_y: float,
    predicted_x: float,
    predicted_y: float,
) -> float:
    """Calculate WGS84 geodesic distance for two projected model positions."""
    actual_latitude, actual_longitude = projected_to_latlon(actual_x, actual_y)
    predicted_latitude, predicted_longitude = projected_to_latlon(
        predicted_x,
        predicted_y,
    )
    _, _, distance = WGS84_GEOD.inv(
        actual_longitude,
        actual_latitude,
        predicted_longitude,
        predicted_latitude,
    )
    return float(distance)


def _make_location_chart(
    result: InferenceResult,
    locations: BuildingFloorLocations,
    ground_truth: dict[str, int | float] | None,
    actual_local: tuple[float, float] | None,
    localization_error: float | None,
) -> plt.Figure:
    """Plot prediction in local building/floor coordinates, never projected axes."""
    prediction = result.prediction
    predicted_local = _local_coordinates(prediction.x, prediction.y, locations)
    neighbor_local = [
        _local_coordinates(neighbor.x, neighbor.y, locations)
        for neighbor in prediction.neighbors
    ]
    figure, axis = plt.subplots(figsize=(9, 7))

    axis.scatter(
        locations.local_x,
        locations.local_y,
        color="#cbd5e1",
        s=9,
        alpha=0.35,
        label="Training locations on this building/floor",
        zorder=1,
    )
    if neighbor_local:
        axis.scatter(
            [point[0] for point in neighbor_local],
            [point[1] for point in neighbor_local],
            color="#475569",
            s=34,
            marker="^",
            label="Nearest training fingerprints",
            zorder=2,
        )

    axis.scatter(
        predicted_local[0],
        predicted_local[1],
        color="#dc2626",
        marker="*",
        s=280,
        edgecolor="black",
        linewidth=0.7,
        label="Predicted location",
        zorder=5,
    )
    if ground_truth is not None and actual_local is not None:
        axis.scatter(
            actual_local[0],
            actual_local[1],
            color="#2563eb",
            marker="o",
            s=115,
            edgecolor="white",
            linewidth=1.0,
            label="Ground truth",
            zorder=4,
        )
        axis.plot(
            [actual_local[0], predicted_local[0]],
            [actual_local[1], predicted_local[1]],
            color="#334155",
            linestyle="--",
            linewidth=1.5,
            label="Ground truth to prediction",
            zorder=3,
        )
        if localization_error is not None:
            axis.annotate(
                f"{localization_error:.2f} m",
                xy=predicted_local,
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

    axis.set(
        title=f"Building {prediction.building} — Floor {prediction.floor}",
        xlabel="Indoor X (meters)",
        ylabel="Indoor Y (meters)",
    )
    x_extent = list(locations.local_x)
    y_extent = list(locations.local_y)
    x_extent.extend((predicted_local[0],))
    y_extent.extend((predicted_local[1],))
    if actual_local is not None:
        x_extent.append(actual_local[0])
        y_extent.append(actual_local[1])
    x_min = min(x_extent)
    x_max = max(x_extent)
    y_min = min(y_extent)
    y_max = max(y_extent)
    x_span = max(float(np.ptp(x_extent)), 1.0)
    y_span = max(float(np.ptp(y_extent)), 1.0)
    x_padding = max(x_span * 0.06, 1.0)
    y_padding = max(y_span * 0.06, 1.0)
    axis.set_xlim(x_min - x_padding, x_max + x_padding)
    axis.set_ylim(y_min - y_padding, y_max + y_padding)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def _render_prediction(
    predictor: WiFiFingerprintPredictor,
    result: InferenceResult,
    ground_truth: dict[str, int | float] | None = None,
) -> None:
    """Display geographic prediction and local indoor view, with optional truth."""
    prediction = result.prediction
    locations = _building_floor_locations(
        predictor,
        prediction.building,
        prediction.floor,
    )
    predicted_local = _local_coordinates(prediction.x, prediction.y, locations)

    st.markdown("### Prediction")
    st.caption(
        "Latitude/longitude are geographic WGS84 coordinates. Indoor X/Y below "
        "are local visualization offsets for this building and floor. Projected "
        "model coordinates are retained internally and are not GPS coordinates."
    )

    if ground_truth is not None:
        actual_latitude, actual_longitude = projected_to_latlon(
            float(ground_truth["x"]),
            float(ground_truth["y"]),
        )
        actual_local = _local_coordinates(
            float(ground_truth["x"]),
            float(ground_truth["y"]),
            locations,
        )
        localization_error = _distance_meters(
            float(ground_truth["x"]),
            float(ground_truth["y"]),
            prediction.x,
            prediction.y,
        )

        ground_truth_column, prediction_column = st.columns(2)
        with ground_truth_column:
            st.markdown("#### Ground Truth")
            st.metric("Building", str(ground_truth["building"]))
            st.metric("Floor", str(ground_truth["floor"]))
            st.metric("Latitude", f"{actual_latitude:.8f}°")
            st.metric("Longitude", f"{actual_longitude:.8f}°")
            st.caption(
                f"Local indoor position: X {actual_local[0]:.2f} m, "
                f"Y {actual_local[1]:.2f} m"
            )
        with prediction_column:
            st.markdown("#### Prediction")
            st.metric("Building", str(prediction.building))
            st.metric("Floor", str(prediction.floor))
            st.metric("Latitude", f"{result.latitude:.8f}°")
            st.metric("Longitude", f"{result.longitude:.8f}°")
            st.caption(
                f"Local indoor position: X {predicted_local[0]:.2f} m, "
                f"Y {predicted_local[1]:.2f} m"
            )
        st.metric("Localization Error", f"{localization_error:.3f} m")
    else:
        actual_local = None
        localization_error = None
        prediction_columns = st.columns(4)
        prediction_columns[0].metric("Building", str(prediction.building))
        prediction_columns[1].metric("Floor", str(prediction.floor))
        prediction_columns[2].metric("Latitude (WGS84)", f"{result.latitude:.8f}°")
        prediction_columns[3].metric("Longitude (WGS84)", f"{result.longitude:.8f}°")
        indoor_columns = st.columns(2)
        indoor_columns[0].metric("Indoor X", f"{predicted_local[0]:.2f} m")
        indoor_columns[1].metric("Indoor Y", f"{predicted_local[1]:.2f} m")

    st.metric("Detected WAPs", f"{result.detected_wap_count} / 520")
    st.info(
        "WKNN compares this WiFi fingerprint with fingerprints collected at "
        "known locations in the UJIIndoorLoc database. The predicted position "
        "is derived from the nearest fingerprints using distance-based weighting."
    )

    with st.expander("Nearest training fingerprints"):
        st.caption(
            "RSSI-space distances and distance-based neighbor weights are not "
            "physical distances or calibrated confidence probabilities."
        )
        neighbor_rows = []
        for neighbor in prediction.neighbors:
            local_x, local_y = _local_coordinates(neighbor.x, neighbor.y, locations)
            neighbor_rows.append(
                {
                    "Training row": neighbor.index,
                    "Building": neighbor.building,
                    "Floor": neighbor.floor,
                    "Indoor X (m)": local_x,
                    "Indoor Y (m)": local_y,
                    "RSSI-space distance": neighbor.distance,
                    "Distance-based weight": neighbor.weight,
                }
            )
        st.dataframe(
            pd.DataFrame(neighbor_rows),
            width="stretch",
            hide_index=True,
        )

    chart = _make_location_chart(
        result,
        locations,
        ground_truth,
        actual_local,
        localization_error,
    )
    st.pyplot(chart, clear_figure=True, width="stretch")
    plt.close(chart)


def _validation_sample_mode(predictor: WiFiFingerprintPredictor) -> None:
    """Predict a validation row from WAP-only features, then display its labels."""
    try:
        validation_data = load_validation_samples()
    except DatasetLoadError as exc:
        st.error(f"Could not load UJIIndoorLoc validation samples: {exc}")
        return
    if validation_data.empty:
        st.error("The UJIIndoorLoc validation CSV contains no samples.")
        return

    sample_index = st.number_input(
        "Validation sample row / index",
        min_value=0,
        max_value=len(validation_data) - 1,
        value=0,
        step=1,
        help=f"Choose a row from 0 to {len(validation_data) - 1}.",
        key="validation_sample_index",
    )
    st.caption(
        f"{len(validation_data):,} official validation samples are available. "
        "Only WAP001-WAP520 will be passed to the saved model."
    )
    if not st.button("Run Prediction", type="primary", key="run_validation_prediction"):
        return

    row = validation_data.iloc[int(sample_index)]
    try:
        wap_frame = validate_upload(validation_data.iloc[[int(sample_index)]])
        result = _predict_fingerprint(predictor, wap_frame)
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(f"Could not predict this validation sample: {exc}")
        return

    ground_truth = _get_ground_truth(row)
    if ground_truth is None:
        st.error("This validation row does not have complete numeric ground truth.")
        return
    _render_prediction(predictor, result, ground_truth)


def _csv_input(predictor: WiFiFingerprintPredictor) -> None:
    """Run inference for every row in an uploaded WAP CSV."""
    uploaded_file = st.file_uploader(
        "Upload a CSV containing WAP001-WAP520",
        type=["csv"],
        key="fingerprint_csv",
    )
    if uploaded_file is None:
        st.caption(
            "A CSV may contain one or many fingerprints. Ground-truth columns "
            "are optional and are excluded from model input."
        )
        return

    try:
        uploaded = pd.read_csv(BytesIO(uploaded_file.getvalue()), low_memory=False)
        wap_frame = validate_upload(uploaded)
        predictions = predictor.predict_batch(wap_frame)
    except (ValueError, RuntimeError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Invalid fingerprint CSV or prediction input: {exc}")
        return

    st.success(f"Predicted {len(predictions):,} fingerprint(s).")
    st.dataframe(pd.DataFrame(predictions), width="stretch", hide_index=True)
    selected_index = st.number_input(
        "Sample to visualize",
        min_value=0,
        max_value=len(wap_frame) - 1,
        value=0,
        step=1,
        key="csv_sample_index",
    )
    row_number = int(selected_index)
    try:
        result = _predict_fingerprint(predictor, wap_frame.iloc[[row_number]])
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(f"Could not display this fingerprint: {exc}")
        return

    _render_prediction(predictor, result)


def _manual_input(predictor: WiFiFingerprintPredictor) -> None:
    """Collect RSSI only for detected WAPs and mark all others unavailable."""
    if "manual_wap_rows" not in st.session_state:
        st.session_state.manual_wap_rows = [0]
        st.session_state.manual_wap_row_count = 1

    st.caption(
        "Add one row per detected access point. Unlisted WAPs are set to the "
        "UJIIndoorLoc not-detected value (100)."
    )
    if st.button("Add detected WAP", key="add_manual_wap"):
        row_id = st.session_state.manual_wap_row_count
        st.session_state.manual_wap_rows.append(row_id)
        st.session_state.manual_wap_row_count += 1
        st.rerun()

    with st.form("manual_fingerprint"):
        entries: list[dict[str, object]] = []
        for row_id in st.session_state.manual_wap_rows:
            wap_id, rssi = st.columns((2, 1))
            with wap_id:
                selected_wap = st.selectbox(
                    f"WAP ID — row {row_id + 1}",
                    options=(None, *RSSI_FEATURE_COLUMNS),
                    format_func=lambda value: value or "Select detected WAP",
                    key=f"manual_wap_{row_id}",
                )
            with rssi:
                selected_rssi = st.number_input(
                    f"RSSI (dBm) — row {row_id + 1}",
                    min_value=RSSI_MIN_VALUE,
                    max_value=RSSI_MAX_VALUE,
                    value=None,
                    step=1,
                    key=f"manual_rssi_{row_id}",
                    help=f"Valid detected signal: {RSSI_MIN_VALUE} to "
                    f"{RSSI_MAX_VALUE} dBm. 100 means not detected.",
                )
            entries.append({"WAP ID": selected_wap, "RSSI": selected_rssi})
        submitted = st.form_submit_button("Run Prediction", type="primary")

    if not submitted:
        return
    try:
        fingerprint = build_manual_fingerprint(pd.DataFrame(entries))
        result = _predict_fingerprint(predictor, fingerprint)
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(f"Invalid manual fingerprint: {exc}")
        return
    _render_prediction(predictor, result)


def main() -> None:
    """Render the three-mode, inference-only WiFi fingerprint application."""
    st.set_page_config(
        page_title="WiFi Fingerprinting Indoor Localization",
        page_icon="📍",
        layout="wide",
    )
    st.title("WiFi Fingerprinting Indoor Localization")
    st.write(
        "Estimate building, floor, and indoor position from WiFi RSSI "
        "fingerprints using the saved Weighted K-Nearest Neighbors model."
    )
    st.info(
        "Inference only: the app loads the saved WKNN model and fitted "
        "preprocessor; it does not retrain or scan WiFi hardware."
    )

    try:
        predictor = load_predictor(str(MODEL_PATH), str(PREPROCESSOR_PATH))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        st.error(
            "Could not load the saved WKNN model and preprocessing configuration. "
            "Run `python -m src.training.train` first. "
            f"Details: {exc}"
        )
        st.stop()

    input_mode = st.radio(
        "Fingerprint input mode",
        INPUT_MODES,
        horizontal=True,
        key="fingerprint_input_mode",
    )
    if input_mode == INPUT_MODES[0]:
        _validation_sample_mode(predictor)
    elif input_mode == INPUT_MODES[1]:
        _csv_input(predictor)
    else:
        _manual_input(predictor)


if __name__ == "__main__":
    main()
