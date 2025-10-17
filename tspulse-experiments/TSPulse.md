# TSPulse Observations

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

Results columns: ['Time', 'Engine Coolant Temperature [°C]', 'Engine RPM [RPM]', 'Vehicle Speed Sensor [km/h]', 'Absolute Throttle Position [%]', 'Intake Manifold Absolute Pressure [kPa]', 'Air Flow Rate from Mass Flow Sensor [g/s]', 'anomaly_score']

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

Results columns: ['Time', 'Engine Coolant Temperature [°C]', 'Engine RPM [RPM]', 'Vehicle Speed Sensor [km/h]', 'Absolute Throttle Position [%]', 'Intake Manifold Absolute Pressure [kPa]', 'Air Flow Rate from Mass Flow Sensor [g/s]', 'anomaly_score']

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

*This is the best result so far, but still not close to 0% anomalies.*

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

