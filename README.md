# WiFi Fingerprinting Based Indoor Localization Using Weighted K-Nearest Neighbors

An academic Python project for indoor localization using WiFi RSSI
fingerprints from the UJIIndoorLoc dataset. The estimator is Weighted
K-Nearest Neighbors (WKNN) only.

## Scope

- Only `WAP001` through `WAP520` are model input features.
- `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`,
  `RELATIVEPOSITION`, `USERID`, `PHONEID`, and `TIMESTAMP` are targets or
  metadata, never model input features.
- No other ML algorithms are implemented or compared.
- Raw dataset files remain local and are ignored by Git by default.

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`

## Reproducible setup and execution

From the repository root in the VS Code terminal, create a virtual
environment, install requirements, and place the original
`trainingData.csv` and `validationData.csv` files in `data/raw/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train using only the training CSV, evaluate using the official validation CSV,
or launch the inference-only application:

```powershell
python -m src.training.train
python -m src.evaluation.metrics
streamlit run app/app.py
```

Run the test suite with `python -m pytest -q`. Model artifacts are saved under
`models/`; evaluation and visualization outputs are saved under `reports/`.

## Dataset placement and loading

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
values, treats RSSI `100` as UJIIndoorLoc's unavailable/not-detected
representation, replaces it and nulls with the configurable default `-110`,
and returns features in canonical WAP order. The training pipeline saves the
fitted, pickleable preprocessor and inference reuses that artifact. The
pipeline reads the raw CSVs; it does not modify them.

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

Training defaults are **K=5**, **Euclidean distance**, and
**inverse-distance weighting**. Exact zero-distance neighbors are handled by
assigning weight only to zero-distance neighbors before normalization. Indoor
coordinates are a distance-weighted average; building and floor are predicted
by weighted neighbor voting. These are distance-based neighbor weights, not
probabilities or calibrated confidence scores. Fitted models can be serialized
with `save()` and restored with `WiFiWKNNLocalizer.load(path)`.

The checked training dataset contains **637 exact duplicate rows**. They are
reported and are not silently removed. The optional holdout hashes complete
WAP fingerprints and uses those hashes as split groups, so identical
fingerprints cannot fall in both holdout and fit partitions. Duplicates remain
in the final fit on all training rows.

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

The optional single-sample demonstration reads one validation row for a
post-fit inference/display check. It passes only that row's WAP columns to the
predictor; the actual labels are displayed after prediction and are not used
for fitting or tuning. Run it with:

```powershell
python -m src.prediction.predictor
```

It prints the selected validation row's actual labels alongside predictions.
Only that row's WAP columns are passed to inference; actual coordinates,
building, and floor are used only for display.

## Coordinate reference system and evaluation

UJIIndoorLoc stores location coordinates in projected metric units. This
implementation uses **EPSG:3857** for projected-to-WGS84 conversion based on
empirical verification and supporting geographic reference material; the
original dataset paper does not explicitly specify an EPSG code. For example,
the sample coordinate `(-7541.2643, 4864920.7782)` converts to approximately
`39.9929702, -0.0677443`, within the mapped Universitat Jaume I campus
boundary in Castellon. This is an implementation assumption, not a CRS
designation attributed to the dataset provider. Conversion uses `pyproj` with
explicit x/y axis order; geographic outputs are returned as
`(latitude, longitude)` in WGS84.

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

## Streamlit application

Run the inference-only application from the repository root:

```powershell
streamlit run app/app.py
```

It loads `models/wknn_localizer.pkl` and
`models/rssi_preprocessor.pkl`; it does not retrain the model. Provide a CSV
with all WAP001–WAP520 columns or enter RSSI values for detected WAPs manually.
Other CSV columns are ignored by inference. Actual-position comparisons appear
only when the uploaded sample includes valid LONGITUDE, LATITUDE, BUILDINGID,
and FLOOR labels. Ground truth is not required for normal inference. The app
does not directly scan WiFi hardware; RSSI values must be supplied through a
CSV or manual entry. Predictions include building, floor, indoor X/Y, and
derived WGS84 latitude/longitude.

## Prediction visualizations

Generate actual/predicted indoor plots, geographic plots, a localization error
distribution, and an interactive Folium map for one validation sample:

```powershell
python -m src.visualization.map_visualization
```

The individual HTML map distinguishes the actual (blue) and predicted (red)
markers, connects them with a line, and displays the WGS84 geodesic error in
meters. Static PNGs and the HTML map are saved under `reports/`. Use
`--sample-index` to select the validation sample for the individual map.

## Training

Train and save the WKNN localizer and fitted RSSI preprocessing configuration:

```powershell
python -m src.training.train
```

The pipeline validates and fits only `trainingData.csv`. Its optional
reproducible holdout evaluation is drawn from the training CSV, keeping
identical WAP fingerprints in the same split to reduce duplicate-fingerprint
leakage. It does not load or fit on the official validation dataset. Final
validation metrics are calculated separately by `src.evaluation.metrics`.
By default, artifacts are saved to `models/wknn_localizer.pkl` and
`models/rssi_preprocessor.pkl`.
Training options, including `--k`, `--distance-metric`, and alternate output
paths, are available through command-line arguments.

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
