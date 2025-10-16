# TSPulse Observations

### Config #1: Recipe defaults

Attempting to set a baseline anomaly score for coolant temperature. Should achieve close to 0% anomalies.

Current results:
- Total data points analyzed: 10,000
- Total anomalies detected: 500
- Overall anomaly rate: 5.00%

Coolant Temperature Sensor Results:
- Coolant anomalies detected: 500
- Coolant anomaly rate: 5.00%
- Threshold value: 0.3002

image.png

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