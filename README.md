# WiFi Fingerprinting Based Indoor Localization Using Weighted K-Nearest Neighbors

An academic Python project scaffold for indoor localization using WiFi RSSI
fingerprints from the UJIIndoorLoc dataset. The planned estimator is Weighted
K-Nearest Neighbors (WKNN) only.

## Project status

Phase 1 established the package structure and basic configuration. Phase 2
provides CSV loading and schema/data-quality validation. Phase 3 provides RSSI
preprocessing as a reusable fitted component. Phase 4 adds exploratory
summaries and plots. WKNN estimation, geographic conversion, model evaluation,
and application behavior have not been implemented yet.

## Scope

- Use only `WAP001` through `WAP520` as model input features.
- Treat building, floor, and indoor coordinates as prediction outputs.
  Geographic coordinates should be derived only after a verified coordinate
  transformation.
- Do not compare WKNN with other algorithms.
- Keep raw dataset files local; they are ignored by Git by default.
- Do not store fabricated metrics or results.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`

## Setup

From the project root in the VS Code terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place `trainingData.csv` and `validationData.csv` in `data/raw/`. The loader
checks the expected names and reports which CSV files are available if one is
missing. Callers can also pass custom file or directory paths.

## Load and validate datasets

```python
from src.data.loader import load_datasets
from src.data.validator import validate_dataframe

training, validation = load_datasets()
for name, dataframe in (("training", training), ("validation", validation)):
    report = validate_dataframe(dataframe, dataset_name=name)
    print(report.to_text())
```

Validation requires all `WAP001`–`WAP520` features and the configured target
columns. RSSI values from -104 through 0 and the UJIIndoorLoc missing-signal
sentinel 100 are recognized. Coordinate ranges are reported in the CSV's
source units; geographic bounds are not assumed without a verified coordinate
reference system. Validation does not alter or preprocess the DataFrame.

## RSSI preprocessing and exploratory analysis

`RSSIPreprocessor` selects only `WAP001` through `WAP520`, converts numeric
values, replaces the UJIIndoorLoc unavailable sentinel (`100`) and nulls with
the configurable default `-110`, and returns the features in canonical WAP
order. Fit it once on the training DataFrame and reuse the fitted, pickleable
instance for inference:

```python
from src.data.loader import load_datasets
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor

training, validation = load_datasets()
preprocessor = RSSIPreprocessor()
training_features = preprocessor.fit_transform(training)
validation_features = preprocessor.transform(validation)
```

Run the full exploratory analysis on the training and validation CSVs. The
script reports dataset sizes, building/floor/space/user/phone counts,
coordinate ranges, RSSI statistics, detected access points per sample, and the
most frequently detected WAPs. Validation and training statistics are printed
separately; location/category plots include both splits.

```powershell
python -m src.analysis.explore_data
```

The script saves building, floor, detected-WAP-count, RSSI, indoor-coordinate,
and building/floor location plots under `reports/`. Use `--training`,
`--validation`, and `--reports-dir` to select alternate paths.

## Run

The current entry point only confirms that the scaffold is ready:

```powershell
python main.py
```

## Tests

Run the project test suite with:

```powershell
python -m pytest
```

## Project layout

```text
data/
    raw/
    processed/
src/
    analysis/
    data/
    evaluation/
    geo/
    model/
    prediction/
    preprocessing/
    training/
    visualization/
    config.py
models/
reports/
tests/
app/
main.py
requirements.txt
```
