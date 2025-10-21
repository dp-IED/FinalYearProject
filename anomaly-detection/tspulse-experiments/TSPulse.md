# TSPulse Observations (KIT dataset)

## Summary

After experimenting with different prediction modes to detect anomalies in Engine Coolant temperatures, I suspect that the model has an internal calculation which results in a baseline 5% anomaly percentage.

To solve this I take the raw anomaly scores assigned by the model to the datapoints, and consider anomalies only those with a 0.5 score or higher.

This method is not ideal, I think this should be reworked to depend on the distribution of anomaly scores.

Best configuration according to this measuring method:

KIT: Config #3 (0.11% anomalies)
carOBD: Config #3 (0.01% anomalies)

The higher KIT anomaly % reflects a heavier high‑score tail caused by dataset/context differences (auxiliary features, transient driving patterns, sampling/normalization), while carOBD is steadier, yielding fewer >0.5 spikes.

**Checks to run**
- **Score distribution**: Compare CDFs/histograms; KS test between KIT vs carOBD anomaly scores.
- **Residuals**: Per-feature residual variance and spike counts; separate warm‑up vs steady‑state.
- **Temporal structure**: Resample to a common rate; check timestamp jitter/missing gaps.
- **Feature alignment**: Use matched auxiliaries (e.g., replace `ENGINE_LOAD` vs `Air Flow Rate`) and re-run.
- **Normalization**: Per‑trip robust scaling; verify units/ranges and leakage-free preprocessing.
- **Sensitivity**: Sweep `smoothing_length`, `aggregation_function` (mean vs median), and percentile-based thresholds.
## Config #1: Recipe defaults trained on one KIT file (2018-03-26_Seat_Leon_RT_S_Normal.csv)

Attempting to set a baseline anomaly score for coolant temperature. Should achieve close to 0% anomalies.

```python
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model,
    timestamp_column="Time",  # KIT dataset time column
    target_columns=[
        "Engine Coolant Temperature [°C]",  # Primary target for coolant anomalies
        "Engine RPM [RPM]",
        "Vehicle Speed Sensor [km/h]",
        "Absolute Throttle Position [%]",
        "Intake Manifold Absolute Pressure [kPa]",
        "Air Flow Rate from Mass Flow Sensor [g/s]"
    ],
    prediction_mode=[
        AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
        AnomalyScoreMethods.PREDICTIVE.value
    ],
    aggregation_length=32,
    aggr_function="max",
    smoothing_length=1,  # No smoothing, we're looking for 1 off events
    least_significant_scale=0.01,
    least_significant_score=0.1,
)
```

### TSPulse Anomaly Detection Results

- **Total points**: 33563 points
- **Calculated threshold (95th percentile)**: 0.0471

- **Max anomaly score**: 1.0000
- **Min anomaly score**: 0.0000
- **Mean anomaly score**: 0.0143

### Coolant Temperature Anomaly Analysis

- **Coolant anomalies detected**: 1679
- **Coolant anomaly percentage**: 5.00%

## Config #2: Tweaked recipe defaults to reduce false positives

Same goal as Config #1, attempting to set a baseline anomaly score for coolant temperature. Should achieve close to 0% anomalies.

```python
config = {
    "prediction_mode": [AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value, AnomalyScoreMethods.PREDICTIVE.value],
    "aggregation_length": 16, # reduced from 32: Smaller windows = detect point anomalies (single data points), Larger windows = detect collective anomalies (sequences of abnormal behavior)
    "aggregation_function": "mean", # changed from max since we're experiencing lots of false positives
    "smoothing_length": 1, # No smoothing, we're looking for isolated events
    "least_significant_scale": 0.02, # increased to 0.02 (Lower values (0.001-0.01) = more sensitive (more anomalies detected), Higher values (0.05-0.1) = less sensitive (fewer anomalies))
    "least_significant_score": 0.2 # Increased to 0.2, minimum anomaly threshold, increased to reduce false positives

}

# Configure anomaly detection pipeline for KIT dataset
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model,
    timestamp_column="Time",  # KIT dataset time column
    target_columns=[
        "Engine Coolant Temperature [°C]",  # Primary target for coolant anomalies
        "Engine RPM [RPM]",
        "Vehicle Speed Sensor [km/h]",
        "Absolute Throttle Position [%]",
        "Intake Manifold Absolute Pressure [kPa]",
        "Air Flow Rate from Mass Flow Sensor [g/s]"
    ],
    prediction_mode=config["prediction_mode"],
    aggregation_length=config["aggregation_length"],
    aggr_function=config["aggregation_function"],
    smoothing_length=config["smoothing_length"],
    least_significant_scale=config["least_significant_scale"],
    least_significant_score=config["least_significant_score"],
)
```

### TSPulse Anomaly Detection Results

- **Total points**: 33563 points
- **Calculated threshold (95th percentile)**: 0.0093 (lower than Config #1)

- **Max anomaly score**: 0.9581 (lower than Config #1)
- **Min anomaly score**: 0.0000
- **Mean anomaly score**: 0.0051 (lower than Config #1)

### Coolant Temperature Anomaly Analysis

- **Coolant anomalies detected**: 1679
- **Coolant anomaly percentage**: 5.00%

Same results as Config #1, same number of anomalies detected, except lower mean anomaly score and lower max anomaly score.

There may be an internal reranking which is preventing us to get <5% anomalies.

## Config #3: Removed frequency reconstruction mode

Removed AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value from the prediction mode.

```python
config = {
    "prediction_mode": [AnomalyScoreMethods.PREDICTIVE.value], # removed AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value
    "aggregation_length": 16,
    "aggregation_function": "mean",
    "smoothing_length": 1,
    "least_significant_scale": 0.02,
    "least_significant_score": 0.2

}
# Configure anomaly detection pipeline for KIT dataset
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model,
    timestamp_column="Time",  # KIT dataset time column
    target_columns=[
        "Engine Coolant Temperature [°C]",  # Primary target for coolant anomalies
        "Engine RPM [RPM]",
        "Vehicle Speed Sensor [km/h]",
        "Absolute Throttle Position [%]",
        "Intake Manifold Absolute Pressure [kPa]",
        "Air Flow Rate from Mass Flow Sensor [g/s]"
    ],
    prediction_mode=config["prediction_mode"],
    aggregation_length=config["aggregation_length"],
    aggr_function=config["aggregation_function"],
    smoothing_length=config["smoothing_length"],
    least_significant_scale=config["least_significant_scale"],
    least_significant_score=config["least_significant_score"],
)
```

### TSPulse Anomaly Detection Results

- **Total points**: 33563 points
- **Calculated threshold (95th percentile)**: 0.0013 (lower than Config #2 and Config #1)
- **Max anomaly score**: 1.0000
- **Min anomaly score**: 0.0000
- **Mean anomaly score**: 0.0017 (lower than Config #2 and Config #1)

### Coolant Temperature Anomaly Analysis

- **Coolant anomalies detected**: 1679 (same as Config #2 and Config #1)
- **Coolant anomaly percentage**: 5.00% (same as Config #2 and Config #1)

With a 0.5 anomaly score threshold, we get 37 anomalies, a 0.11% anomaly rate. This is a good result, we can consider that we have achieved a baseline on the KIT dataset.

_This is the best result so far, but still not close to 0% anomalies._

## Config #4: Tried frequency reconstruction mode only

Removed AnomalyScoreMethods.PREDICTIVE.value from the prediction mode.

```python
config = {
    "prediction_mode": [AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value], # removed AnomalyScoreMethods.Predictive
    "aggregation_length": 16,
    "aggregation_function": "mean",
    "smoothing_length": 1,
    "least_significant_scale": 0.02,
    "least_significant_score": 0.2
}

# Configure anomaly detection pipeline for KIT dataset
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model,
    timestamp_column="Time",  # KIT dataset time column
    target_columns=[
        "Engine Coolant Temperature [°C]",  # Primary target for coolant anomalies
        "Engine RPM [RPM]",
        "Vehicle Speed Sensor [km/h]",
        "Absolute Throttle Position [%]",
        "Intake Manifold Absolute Pressure [kPa]",
        "Air Flow Rate from Mass Flow Sensor [g/s]"
    ],
    prediction_mode=config["prediction_mode"],
    aggregation_length=config["aggregation_length"],
    aggr_function=config["aggregation_function"],
    smoothing_length=config["smoothing_length"],
    least_significant_scale=config["least_significant_scale"],
    least_significant_score=config["least_significant_score"],
)
```

### TSPulse Anomaly Detection Results

- **Total points**: 33563 points
- **Calculated threshold (95th percentile)**: 0.0153
- **Max anomaly score**: 1.0000
- **Min anomaly score**: 0.0000
- **Mean anomaly score**: 0.0084 (higher than Config #3 and Config #2)

### Coolant Temperature Anomaly Analysis

- **Coolant anomalies detected**: 1679 (same as Config #3 and Config #2)
- **Coolant anomaly percentage**: 5.00% (same as Config #3 and Config #2)

Same results as Config #3 and Config #2, same number of anomalies detected, except higher mean anomaly score.

## Observations

### There seems to be an internal 5% default detection rate

### Solution: Take only those that have a .5 anomaly score or higher. This is a scuffed method, but it works for now.

### We could have multiple thresholds for different levels of severity (e.g. 0.5, 0.75, 1.0)

# TSPulse Observations (CarOBD dataset)

## Config #1: Using Config #3

Using predictive mode only, same as Config #3 for KIT dataset, since it seems to be the best performing mode in terms of reducing the mean anomaly score.

```python
config = {
    "prediction_mode": [AnomalyScoreMethods.PREDICTIVE.value], # removed AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION
    "aggregation_length": 16,
    "aggregation_function": "mean",
    "smoothing_length": 1,
    "least_significant_scale": 0.02,
    "least_significant_score": 0.2
}

# Configure anomaly detection pipeline for CarOBD dataset
pipeline = TimeSeriesAnomalyDetectionPipeline(
    model,
    timestamp_column="Time",  # CarOBD dataset time column
    target_columns=[
        "COOLANT_TEMPERATURE",  # Primary target for coolant anomalies
        "ENGINE_RPM",
        "VEHICLE_SPEED",
        "THROTTLE",
        "INTAKE_MANIFOLD_PRESSURE",
        "ENGINE_LOAD"
    ],
    prediction_mode=config["prediction_mode"],
    aggregation_length=config["aggregation_length"],
    aggr_function=config["aggregation_function"],
    smoothing_length=config["smoothing_length"],
    least_significant_scale=config["least_significant_scale"],
    least_significant_score=config["least_significant_score"],
)
```

### TSPulse Anomaly Detection Results

- **Total points**: 34684 points
- **Calculated threshold (95th percentile)**: 0.0008
- **Max anomaly score**: 1.0000
- **Min anomaly score**: 0.0000
- **Mean anomaly score**: 0.0009

### Coolant Temperature Anomaly Analysis

- **Coolant anomalies detected**: 1735
- **Coolant anomaly percentage**: 5.00%

With a 0.5 anomaly score threshold, we get only 5 anomalies, a 0.01% anomaly rate. This is a good result, we can consider that we have achieved a baseline on both datasets.
