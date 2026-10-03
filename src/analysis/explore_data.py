"""Exploratory analysis and plots for the UJIIndoorLoc dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, TRAINING_DATA_PATH, VALIDATION_DATA_PATH
from src.data.loader import PathLike, load_datasets
from src.data.validator import ValidationReport, validate_dataframe
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor

_UNAVAILABLE_REPLACEMENT = -110.0


def _unique_space_count(dataframe: pd.DataFrame) -> int:
    """Count unique physical-space labels by building, floor, and space ID."""
    space_columns = ["BUILDINGID", "FLOOR", "SPACEID"]
    return int(dataframe[space_columns].drop_duplicates().shape[0])


def _print_dataset_summary(
    name: str,
    dataframe: pd.DataFrame,
    validation_report: ValidationReport,
    preprocessor: RSSIPreprocessor,
) -> tuple[np.ndarray, pd.Series]:
    """Print dataset and RSSI summaries; return signal values and AP counts."""
    print(f"\n=== {name.title()} dataset ===")
    print(f"Dataset size: {len(dataframe):,} samples x {dataframe.shape[1]:,} columns")
    print(f"WAP features: {len(preprocessor.get_feature_names_out()):,}")
    print(
        "Buildings: "
        f"{dataframe['BUILDINGID'].nunique(dropna=True):,} "
        f"(values: {sorted(dataframe['BUILDINGID'].dropna().unique().tolist())})"
    )
    print(
        "Floors: "
        f"{dataframe['FLOOR'].nunique(dropna=True):,} "
        f"(values: {sorted(dataframe['FLOOR'].dropna().unique().tolist())})"
    )
    print(f"Unique spaces (building/floor/SPACEID): {_unique_space_count(dataframe):,}")
    print(f"Users: {dataframe['USERID'].nunique(dropna=True):,}")
    print(f"Phones: {dataframe['PHONEID'].nunique(dropna=True):,}")
    for coordinate, bounds in validation_report.coordinate_ranges.items():
        print(f"{coordinate} range (source units): {bounds}")

    processed = preprocessor.transform(dataframe)
    detected_mask = processed.ne(_UNAVAILABLE_REPLACEMENT)
    detected_counts = detected_mask.sum(axis=1)
    signal_values = processed.to_numpy(copy=False)[
        detected_mask.to_numpy(copy=False)
    ]

    print("Detected RSSI statistics (sentinels excluded):")
    if signal_values.size:
        print(
            f"  count={signal_values.size:,}, "
            f"min={np.min(signal_values):.0f}, "
            f"max={np.max(signal_values):.0f}, "
            f"mean={np.mean(signal_values):.2f}, "
            f"median={np.median(signal_values):.2f}, "
            f"std={np.std(signal_values):.2f}"
        )
    else:
        print("  No detected RSSI values.")
    print(
        "Detected WAPs per sample: "
        f"min={int(detected_counts.min())}, "
        f"mean={detected_counts.mean():.2f}, "
        f"median={detected_counts.median():.0f}, "
        f"max={int(detected_counts.max())}"
    )
    detected_distribution = Counter(int(value) for value in detected_counts)
    print(
        "Detected WAP count distribution (detected count: samples): "
        f"{dict(sorted(detected_distribution.items()))}"
    )
    print("Most frequently detected WAPs:")
    wap_counts = detected_mask.sum(axis=0).sort_values(ascending=False, kind="stable")
    for wap_name, sample_count in wap_counts.head(10).items():
        print(
            f"  {wap_name}: {int(sample_count):,} samples "
            f"({sample_count / len(dataframe) * 100:.2f}%)"
        )
    print(validation_report.to_text())
    return signal_values, detected_counts


def _plot_split_distribution(
    training: pd.Series,
    validation: pd.Series,
    title: str,
    xlabel: str,
    destination: Path,
) -> None:
    """Save a grouped bar chart comparing category counts by data split."""
    categories = sorted(set(training.index).union(validation.index))
    positions = np.arange(len(categories), dtype=float)
    width = 0.4
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        positions - width / 2,
        training.reindex(categories, fill_value=0).to_numpy(),
        width,
        label="Training",
    )
    axis.bar(
        positions + width / 2,
        validation.reindex(categories, fill_value=0).to_numpy(),
        width,
        label="Validation",
    )
    axis.set(
        title=title,
        xlabel=xlabel,
        ylabel="Samples",
        xticks=positions,
        xticklabels=[str(category) for category in categories],
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_detected_wap_counts(
    counts_by_split: dict[str, pd.Series],
    feature_count: int,
    destination: Path,
) -> None:
    """Save the per-sample detected WAP count distribution by data split."""
    figure, axis = plt.subplots(figsize=(10, 5))
    bins = np.arange(-0.5, feature_count + 1.5, 1)
    for split, counts in counts_by_split.items():
        axis.hist(counts.to_numpy(), bins=bins, alpha=0.55, label=split.title())
    axis.set(
        title="Detected WAPs per Sample",
        xlabel="Detected WAP count",
        ylabel="Samples",
        xlim=(-0.5, feature_count + 0.5),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_rssi_distribution(
    signals_by_split: dict[str, np.ndarray],
    destination: Path,
) -> None:
    """Save the RSSI signal distribution, excluding unavailable sentinels."""
    figure, axis = plt.subplots(figsize=(10, 5))
    bins = np.arange(-104.5, 1.5, 1)
    for split, signal_values in signals_by_split.items():
        axis.hist(
            signal_values,
            bins=bins,
            alpha=0.55,
            label=split.title(),
            density=True,
        )
    axis.set(
        title="RSSI Distribution for Detected Access Points",
        xlabel="RSSI (dBm)",
        ylabel="Density",
        xlim=(-105, 1),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_indoor_coordinates(
    combined: pd.DataFrame,
    destination: Path,
) -> None:
    """Save indoor coordinate scatter plots, grouped by building and floor."""
    buildings = sorted(combined["BUILDINGID"].dropna().unique())
    if not buildings:
        raise ValueError("Cannot plot indoor coordinates without building labels.")

    figure, axis = plt.subplots(figsize=(9, 7))
    for building in buildings:
        building_rows = combined.loc[combined["BUILDINGID"] == building]
        axis.scatter(
            building_rows["LONGITUDE"],
            building_rows["LATITUDE"],
            s=7,
            alpha=0.4,
            label=f"Building {building}",
        )
    axis.set(
        title="Indoor Coordinates by Building (Training and Validation)",
        xlabel="LONGITUDE (source coordinate units)",
        ylabel="LATITUDE (source coordinate units)",
    )
    axis.legend(markerscale=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_building_floor_locations(
    combined: pd.DataFrame,
    destination: Path,
) -> None:
    """Save one coordinate scatter subplot per building, colored by floor."""
    buildings = sorted(combined["BUILDINGID"].dropna().unique())
    if not buildings:
        raise ValueError("Cannot plot building/floor locations without building labels.")

    figure, axes = plt.subplots(
        len(buildings),
        1,
        figsize=(10, max(4, 4 * len(buildings))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    floor_values = sorted(combined["FLOOR"].dropna().unique())
    colors = plt.get_cmap("tab10")
    for axis, building in zip(axes[:, 0], buildings):
        building_rows = combined.loc[combined["BUILDINGID"] == building]
        for color_index, floor in enumerate(floor_values):
            floor_rows = building_rows.loc[building_rows["FLOOR"] == floor]
            if floor_rows.empty:
                continue
            axis.scatter(
                floor_rows["LONGITUDE"],
                floor_rows["LATITUDE"],
                s=7,
                alpha=0.4,
                color=colors(color_index % 10),
                label=f"Floor {floor}",
            )
        axis.set_title(f"Building {building}")
        axis.set_ylabel("LATITUDE (source units)")
        axis.grid(alpha=0.2)
        axis.legend(markerscale=2, fontsize="small")
    axes[-1, 0].set_xlabel("LONGITUDE (source units)")
    figure.suptitle("Indoor Location Distribution by Building and Floor")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _print_combined_counts(training: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Report distinct buildings, floors, spaces, users, and phones overall."""
    combined = pd.concat(
        [
            training[["BUILDINGID", "FLOOR", "SPACEID", "USERID", "PHONEID"]],
            validation[["BUILDINGID", "FLOOR", "SPACEID", "USERID", "PHONEID"]],
        ],
        ignore_index=True,
    )
    print("\n=== Combined dataset label counts ===")
    print(f"Buildings: {combined['BUILDINGID'].nunique(dropna=True):,}")
    print(f"Floors: {combined['FLOOR'].nunique(dropna=True):,}")
    print(f"Unique spaces (building/floor/SPACEID): {_unique_space_count(combined):,}")
    print(f"Users: {combined['USERID'].nunique(dropna=True):,}")
    print(f"Phones: {combined['PHONEID'].nunique(dropna=True):,}")


def explore_dataset(
    training_path: PathLike = TRAINING_DATA_PATH,
    validation_path: PathLike = VALIDATION_DATA_PATH,
    reports_dir: PathLike = REPORTS_DIR,
) -> dict[str, Path]:
    """Summarize both dataset splits and generate the required PNG plots.

    Plots and RSSI summaries use only WAP features from the preprocessor;
    labels and coordinates are used separately for descriptive summaries.

    Args:
        training_path: Training CSV path or directory containing its CSV.
        validation_path: Validation CSV path or directory containing its CSV.
        reports_dir: Directory in which PNG plots are saved.

    Returns:
        A mapping from plot names to generated file paths.
    """
    training, validation = load_datasets(
        training_path=training_path,
        validation_path=validation_path,
    )
    training_report = validate_dataframe(training, dataset_name="training")
    validation_report = validate_dataframe(validation, dataset_name="validation")
    for report in (training_report, validation_report):
        if not report.is_valid:
            raise ValueError(report.to_text())

    preprocessor = RSSIPreprocessor().fit(training)
    signals_by_split: dict[str, np.ndarray] = {}
    counts_by_split: dict[str, pd.Series] = {}
    for name, dataframe, report in (
        ("training", training, training_report),
        ("validation", validation, validation_report),
    ):
        signals, detected_counts = _print_dataset_summary(
            name,
            dataframe,
            report,
            preprocessor,
        )
        signals_by_split[name] = signals
        counts_by_split[name] = detected_counts

    _print_combined_counts(training, validation)
    combined_locations = pd.concat(
        [
            training[["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]],
            validation[["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]],
        ],
        ignore_index=True,
    )

    output_directory = Path(reports_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "building_distribution": output_directory / "building_distribution.png",
        "floor_distribution": output_directory / "floor_distribution.png",
        "detected_wap_count_distribution": (
            output_directory / "detected_wap_count_distribution.png"
        ),
        "rssi_distribution": output_directory / "rssi_distribution.png",
        "indoor_coordinate_scatter": (
            output_directory / "indoor_coordinate_scatter.png"
        ),
        "building_floor_location_distribution": (
            output_directory / "building_floor_location_distribution.png"
        ),
    }
    _plot_split_distribution(
        training["BUILDINGID"].value_counts().sort_index(),
        validation["BUILDINGID"].value_counts().sort_index(),
        "Building Distribution by Dataset Split",
        "Building ID",
        plot_paths["building_distribution"],
    )
    _plot_split_distribution(
        training["FLOOR"].value_counts().sort_index(),
        validation["FLOOR"].value_counts().sort_index(),
        "Floor Distribution by Dataset Split",
        "Floor",
        plot_paths["floor_distribution"],
    )
    _plot_detected_wap_counts(
        counts_by_split,
        len(preprocessor.get_feature_names_out()),
        plot_paths["detected_wap_count_distribution"],
    )
    _plot_rssi_distribution(
        signals_by_split,
        plot_paths["rssi_distribution"],
    )
    _plot_indoor_coordinates(
        combined_locations,
        plot_paths["indoor_coordinate_scatter"],
    )
    _plot_building_floor_locations(
        combined_locations,
        plot_paths["building_floor_location_distribution"],
    )

    print("\n=== Generated plots ===")
    for plot_name, plot_path in plot_paths.items():
        print(f"{plot_name}: {plot_path}")
    return plot_paths


def main(arguments: Sequence[str] | None = None) -> None:
    """Run exploratory analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Explore the UJIIndoorLoc training and validation datasets."
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=TRAINING_DATA_PATH,
        help="Training CSV path or directory containing trainingData.csv.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=VALIDATION_DATA_PATH,
        help="Validation CSV path or directory containing validationData.csv.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for generated PNG plots.",
    )
    options = parser.parse_args(arguments)
    explore_dataset(options.training, options.validation, options.reports_dir)


if __name__ == "__main__":
    main()
