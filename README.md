# WiFi Fingerprinting Based Indoor Localization Using Weighted K-Nearest Neighbors

An academic Python project scaffold for indoor localization using WiFi RSSI
fingerprints from the UJIIndoorLoc dataset. The planned estimator is Weighted
K-Nearest Neighbors (WKNN) only.

## Project status

Phase 1 established the package structure and basic configuration. Phase 2
provides CSV loading and schema/data-quality validation. Phase 3 provides RSSI
preprocessing as a reusable fitted component. Phase 4 adds exploratory
summaries and plots. Phase 5 adds the WKNN localization engine. Phase 6 adds
training and model persistence. Phase 7 adds saved-artifact inference.
Phase 8 adds verified coordinate conversion and official-validation
evaluation. Application behavior has not been implemented yet.

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

## WKNN localization

Fit from the complete training DataFrame, or pass the WAP feature DataFrame and
the four target columns separately. When making predictions, pass
`WAP001`–`WAP520` in a DataFrame; additional label or metadata columns are
ignored. Apply the same fitted `RSSIPreprocessor` to training and inference
fingerprints before fitting or prediction:

```python
from src.data.loader import load_training_data
from src.model.wknn_localizer import WiFiWKNNLocalizer
from src.preprocessing.rssi_preprocessor import RSSIPreprocessor

training = load_training_data()
preprocessor = RSSIPreprocessor()
training_features = preprocessor.fit_transform(training)
localizer = WiFiWKNNLocalizer(k=5).fit(training_features.join(training[
    ["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]
]))
prediction = localizer.predict_single(training_features.iloc[[0]])
print(prediction)
```

The localizer supports configurable scikit-learn distance metrics and
`inverse_distance` or `uniform` neighbor weighting. Fitted models can be
serialized with `save()` and restored with `WiFiWKNNLocalizer.load(path)`.
Coordinates remain in the dataset's source coordinate system; no geographic
transformation is performed.

## Inference

Load the saved artifacts and predict one fingerprint or a batch. Prediction
accepts a DataFrame or mappings containing all 520 WAP fields. Additional
metadata columns are ignored; only WAP features enter preprocessing and the
model.

```python
from src.prediction.predictor import WiFiFingerprintPredictor

predictor = WiFiFingerprintPredictor()
one_prediction = predictor.predict_single(wap_fingerprint)
batch_predictions = predictor.predict_batch(wap_fingerprint_dataframe)
```

Run the validation-sample demonstration from the VS Code terminal:

```powershell
python -m src.prediction.predictor
```

It prints the selected validation row's actual labels alongside predictions.
Only that row's WAP columns are passed to inference; actual coordinates,
building, and floor are used only for display.

## Coordinate reference system and evaluation

The UCI UJIIndoorLoc description documents the coordinate and RSSI columns,
but does not state an EPSG code. The source coordinates are verified here as
**WGS 84 / Pseudo-Mercator (EPSG:3857)**, in meters, rather than decimal-degree
GPS values: the sample coordinate `(-7541.2643, 4864920.7782)` converts to
approximately `39.9929702, -0.0677443`, which falls within the mapped
Universitat Jaume I campus boundary in Castellon. The EPSG:3395 alternative
puts the same point at approximately 40.1825° N, outside the campus. CRS
conversion uses `pyproj` and explicit x/y axis order; geographic outputs are
returned as `(latitude, longitude)` in WGS84.

Sources used to verify the CRS:

- [UCI UJIIndoorLoc dataset documentation](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) — data description and projected coordinate values.
- [Universitat Jaume I campus boundary (OpenStreetMap)](https://www.openstreetmap.org/way/46835092) — known geographic reference extent.
- [EPSG:3857 definition](https://epsg.io/3857) — projected CRS definition and meter units.

Evaluate the saved model using the official validation set without fitting on
it:

```powershell
python -m src.evaluation.metrics
```

The predictor receives only WAP001–WAP520. Geographic localization distances
are computed with the WGS84 ellipsoidal geodesic in meters. Results and plots
are written to `reports/evaluation_results.csv`,
`reports/localization_error_distribution.png`, and
`reports/actual_vs_predicted_coordinates.png`.

## Training

Train and save the WKNN localizer and fitted RSSI preprocessing configuration:

```powershell
python -m src.training.train
```

The pipeline validates and fits only `trainingData.csv`. Its optional
reproducible holdout evaluation is drawn from the training CSV, keeping
identical WAP fingerprints in the same split to reduce duplicate-fingerprint
leakage. It does not load or fit on the official validation dataset. By
default, artifacts are saved to `models/wknn_localizer.pkl` and
`models/rssi_preprocessor.pkl`.
Training options, including `--k`, `--distance-metric`, and alternate output
paths, are available through command-line arguments.

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
