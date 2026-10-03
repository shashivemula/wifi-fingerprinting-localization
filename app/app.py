"""Streamlit interface for inference with the saved WiFi WKNN model."""

from __future__ import annotations

import sys
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
from src.geo.coordinate_converter import projected_to_latlon
from src.model.wknn_localizer import LocalizationPrediction
from src.prediction.predictor import WiFiFingerprintPredictor

MODEL_PATH = MODELS_DIR / "wknn_localizer.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "rssi_preprocessor.pkl"
GROUND_TRUTH_COLUMNS = ("LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR")
RSSI_MISSING_VALUE = 100
WGS84_GEOD = Geod(ellps="WGS84")


@st.cache_resource(show_spinner="Loading saved WKNN model...")
def load_predictor(model_path: str, preprocessor_path: str) -> WiFiFingerprintPredictor:
    """Load saved inference artifacts once per Streamlit process."""
    return WiFiFingerprintPredictor(model_path, preprocessor_path)


def validate_upload(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate a CSV and return only its WAP columns for inference."""
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


def _get_ground_truth(row: pd.Series) -> dict[str, int | float] | None:
    """Return valid actual labels separately, if all expected labels exist."""
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
    return {
        "x": values["LONGITUDE"],
        "y": values["LATITUDE"],
        "building": int(values["BUILDINGID"]),
        "floor": int(values["FLOOR"]),
    }


def _distance_meters(
    actual_x: float,
    actual_y: float,
    predicted_x: float,
    predicted_y: float,
) -> float:
    """Calculate WGS84 geodesic distance from dataset projected coordinates."""
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
    prediction: LocalizationPrediction,
    ground_truth: dict[str, int | float] | None,
) -> plt.Figure:
    """Plot the prediction, neighbors, and actual point only when labeled."""
    figure, axis = plt.subplots(figsize=(8, 6))
    neighbors = prediction.neighbors
    if neighbors:
        axis.scatter(
            [neighbor.x for neighbor in neighbors],
            [neighbor.y for neighbor in neighbors],
            c=[neighbor.distance for neighbor in neighbors],
            cmap="Blues",
            s=70,
            marker="^",
            label="Nearest training fingerprints",
        )
    axis.scatter(
        prediction.x,
        prediction.y,
        color="#d62728",
        marker="*",
        s=220,
        edgecolor="black",
        linewidth=0.6,
        label="Predicted position",
        zorder=3,
    )

    title = "Predicted indoor position"
    if ground_truth is not None:
        axis.scatter(
            float(ground_truth["x"]),
            float(ground_truth["y"]),
            color="#2ca02c",
            marker="o",
            s=100,
            edgecolor="black",
            linewidth=0.6,
            label="Actual position",
            zorder=4,
        )
        axis.plot(
            [float(ground_truth["x"]), prediction.x],
            [float(ground_truth["y"]), prediction.y],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Actual-to-predicted error",
            zorder=1,
        )
        title = "Actual and predicted indoor position"

    axis.set(
        title=title,
        xlabel="X (EPSG:3857 meters)",
        ylabel="Y (EPSG:3857 meters)",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def _display_prediction(
    predictor: WiFiFingerprintPredictor,
    wap_frame: pd.DataFrame,
    ground_truth: dict[str, int | float] | None,
) -> None:
    """Run inference and display predicted location, neighbors, and optional truth."""
    try:
        transformed = predictor.preprocessor.transform(wap_frame)
        model_prediction = predictor.model.predict_single(transformed)
        prediction = {
            "building": model_prediction.building,
            "floor": model_prediction.floor,
            "x": model_prediction.x,
            "y": model_prediction.y,
        }
        latitude, longitude = projected_to_latlon(prediction["x"], prediction["y"])
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(f"Could not generate a prediction: {exc}")
        return

    detected_count = int(
        transformed.ne(predictor.preprocessor.replacement_value).sum(axis=1).iloc[0]
    )
    st.subheader("Prediction")
    result_columns = st.columns(4)
    result_columns[0].metric("Predicted Building", str(prediction["building"]))
    result_columns[1].metric("Predicted Floor", str(prediction["floor"]))
    result_columns[2].metric("Indoor X", f"{prediction['x']:.3f} m")
    result_columns[3].metric("Indoor Y", f"{prediction['y']:.3f} m")
    geographic_columns = st.columns(2)
    geographic_columns[0].metric("Latitude (WGS84)", f"{latitude:.8f}°")
    geographic_columns[1].metric("Longitude (WGS84)", f"{longitude:.8f}°")
    st.metric("Detected WAP count", f"{detected_count} / {len(RSSI_FEATURE_COLUMNS)}")

    if ground_truth is not None:
        actual_latitude, actual_longitude = projected_to_latlon(
            float(ground_truth["x"]),
            float(ground_truth["y"]),
        )
        localization_error = _distance_meters(
            float(ground_truth["x"]),
            float(ground_truth["y"]),
            prediction["x"],
            prediction["y"],
        )
        st.subheader("Actual vs predicted (uploaded labels)")
        st.write(
            {
                "Actual building": ground_truth["building"],
                "Actual floor": ground_truth["floor"],
                "Actual X": float(ground_truth["x"]),
                "Actual Y": float(ground_truth["y"]),
                "Actual latitude": actual_latitude,
                "Actual longitude": actual_longitude,
                "Localization error (meters)": localization_error,
            }
        )
    else:
        st.info(
            "No complete ground-truth labels were supplied. Only the predicted "
            "position is shown; actual location and localization error are not "
            "available."
        )

    with st.expander("Nearest training fingerprints and weighting"):
        st.caption(
            "RSSI-space distance is measured over the 520 preprocessed WAP "
            "features; it is not a physical distance in meters. WKNN weights "
            "the coordinate estimate by inverse RSSI-space distance. These "
            "weights are not calibrated probabilities or confidence scores."
        )
        neighbors = pd.DataFrame(
            [
                {
                    "Training row": neighbor.index,
                    "RSSI-space distance": neighbor.distance,
                    "WKNN weight": neighbor.weight,
                    "Building": neighbor.building,
                    "Floor": neighbor.floor,
                    "X": neighbor.x,
                    "Y": neighbor.y,
                }
                for neighbor in model_prediction.neighbors
            ]
        )
        st.dataframe(neighbors, width="stretch", hide_index=True)

    chart = _make_location_chart(model_prediction, ground_truth)
    st.pyplot(chart, clear_figure=True, width="stretch")
    plt.close(chart)


def _csv_input(predictor: WiFiFingerprintPredictor) -> None:
    """Upload a WAP CSV, predict each sample, and display a selected row."""
    uploaded_file = st.file_uploader(
        "Upload a CSV with WAP001-WAP520 columns",
        type=["csv"],
        key="fingerprint_csv",
    )
    if uploaded_file is None:
        st.caption("The CSV may contain one or more samples; extra columns are ignored.")
        return

    try:
        uploaded = pd.read_csv(BytesIO(uploaded_file.getvalue()), low_memory=False)
        wap_frame = validate_upload(uploaded)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Invalid fingerprint CSV: {exc}")
        return

    try:
        predictions = predictor.predict_batch(wap_frame)
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(f"Could not predict uploaded fingerprints: {exc}")
        return

    st.success(f"Validated {len(wap_frame):,} fingerprint(s).")
    st.dataframe(
        pd.DataFrame(predictions, index=range(len(predictions))).rename_axis("Sample"),
        width="stretch",
    )
    selected_index = st.number_input(
        "Sample to visualize",
        min_value=0,
        max_value=len(wap_frame) - 1,
        value=0,
        step=1,
        key="csv_sample_index",
    )
    row_number = int(selected_index)
    ground_truth = _get_ground_truth(uploaded.iloc[row_number])
    if ground_truth is None and any(
        column in uploaded.columns for column in GROUND_TRUTH_COLUMNS
    ):
        st.warning(
            "Some ground-truth columns were found, but the complete numeric "
            "LONGITUDE, LATITUDE, BUILDINGID, and FLOOR targets are required "
            "to compare positions."
        )
    _display_prediction(
        predictor,
        wap_frame.iloc[[row_number]],
        ground_truth,
    )


def _manual_input(predictor: WiFiFingerprintPredictor) -> None:
    """Collect manually entered RSSI values for detected WAPs."""
    selected_waps = st.multiselect(
        "Select detected access points",
        options=RSSI_FEATURE_COLUMNS,
        help=(
            "Leave WAPs unselected when they were not detected. Unselected "
            "features use the UJIIndoorLoc unavailable-signal value 100."
        ),
        key="manual_selected_waps",
    )
    with st.form("manual_fingerprint"):
        values: dict[str, int] = {}
        columns = st.columns(3)
        for index, wap_name in enumerate(selected_waps):
            with columns[index % len(columns)]:
                values[wap_name] = st.number_input(
                    wap_name,
                    min_value=-104,
                    max_value=0,
                    value=-70,
                    step=1,
                    help="RSSI in dBm; accepted range is -104 to 0.",
                )
        submitted = st.form_submit_button("Predict location")

    if submitted:
        if not selected_waps:
            st.error("Select at least one detected WAP and enter its RSSI value.")
            return
        fingerprint = {column: RSSI_MISSING_VALUE for column in RSSI_FEATURE_COLUMNS}
        fingerprint.update(values)
        try:
            wap_frame = validate_upload(pd.DataFrame([fingerprint]))
        except ValueError as exc:
            st.error(f"Invalid manual fingerprint: {exc}")
            return
        _display_prediction(predictor, wap_frame, None)


def main() -> None:
    """Render the inference-only WiFi fingerprint application."""
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
        "This app performs inference only. It does not train or modify the model."
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
        "Fingerprint input method",
        ("Upload CSV", "Enter RSSI values manually"),
        horizontal=True,
    )
    if input_mode == "Upload CSV":
        _csv_input(predictor)
    else:
        _manual_input(predictor)


if __name__ == "__main__":
    main()
