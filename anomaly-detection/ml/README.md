# ML Anomaly Detection for carOBD Data

A production-ready machine learning system for detecting anomalies in automotive engine sensor data, specifically designed for carOBD datasets.

## Overview

This project addresses overfitting issues in ML anomaly detection and provides realistic performance evaluation for automotive sensor fault detection. The system uses a unified approach that works for both idle and motion engine data.

## Key Features

- **Unified Model**: Single model for both idle and motion data
- **Realistic Fault Injection**: Automotive-specific sensor failure patterns
- **Model Persistence**: Save/load functionality to avoid retraining
- **Comprehensive Evaluation**: Multiple metrics including F1-score, precision, recall
- **Production Ready**: Clean, well-documented code

## Performance Results

| Algorithm            | F1-Score | Precision | Recall | Accuracy |
| -------------------- | -------- | --------- | ------ | -------- |
| **Isolation Forest** | 0.448    | 0.527     | 0.390  | 0.808    |
| **One-Class SVM**    | 0.325    | 0.788     | 0.205  | 0.830    |

## Project Structure

```
anomaly-detection/ml/
├── ml_anomaly_detector.py    # Main production module
├── test_unified_model.py     # Test script for existing model
├── models/                   # Saved models directory
│   ├── README.md            # Model documentation
│   └── arima_models/        # Legacy ARIMA models
└── README.md                # This file

anomaly-detection/models/
└── unified_ml_detector.pkl   # Current working ML model
```

## Installation

1. Ensure you have the required dependencies:

```bash
pip install numpy pandas scikit-learn
```

2. Place your carOBD data in `data/carOBD/obdiidata/` directory

## Usage

### Training the Model (First Time)

```bash
python anomaly-detection/ml/ml_anomaly_detector.py
```

This will:

- Load all idle and motion data
- Extract features with built-in realistic fault injection
- Train both Isolation Forest and One-Class SVM models
- Save the trained model to `models/ml_anomaly_detector.pkl`

### Testing the Model

```bash
python anomaly-detection/ml/test_model.py
```

This will:

- Load the saved model
- Test on a sample of data
- Show performance metrics and example predictions

## Data Requirements

The system expects carOBD data files in the following format:

- **Idle files**: `idle*.csv` (e.g., idle1.csv, idle2.csv, ...)
- **Motion files**: `drive*.csv`, `live*.csv`, `long*.csv`, `ufpe*.csv`

Required sensor columns:

- `COOLANT_TEMPERATURE ()`
- `ENGINE_RPM ()`
- `VEHICLE_SPEED ()`
- `THROTTLE ()`
- `ENGINE_LOAD ()`
- `INTAKE_MANIFOLD_PRESSURE ()`
- `INTAKE_AIR_TEMP ()`

## Features Extracted

The system extracts the following features:

### Basic Sensor Features

- All 7 sensor readings with calibration applied

### Derived Features

- **Temperature-RPM Ratio**: Thermal efficiency indicator
- **Throttle Efficiency**: Speed-to-throttle ratio

### Statistical Features

- **Rolling Statistics**: Mean and standard deviation (3-point window)
- **Temporal Patterns**: Captures sensor behavior over time

## Fault Injection Strategy

The system creates realistic automotive sensor faults:

1. **Coolant Bias**: Systematic temperature offset (10-30°C)
2. **Coolant Drift**: Gradual sensor aging (20% drift)
3. **Stuck Sensor**: Fixed unrealistic temperature values
4. **RPM Bias**: Engine speed sensor offset (100-500 RPM)
5. **Multi-Sensor**: Correlated degradation across sensors

## Model Parameters

### Isolation Forest

- `contamination`: 0.1 (expected anomaly proportion)
- `n_estimators`: 200 (number of trees)
- `max_samples`: 0.8 (bootstrap sampling)
- `max_features`: 0.8 (feature subsampling)

### One-Class SVM

- `nu`: 0.1 (proportion of outliers)
- `kernel`: 'rbf' (radial basis function)
- `gamma`: 'scale' (automatic scaling)

## Evaluation Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Correct predictions / Total predictions
- **Balanced Accuracy**: (Recall + Specificity) / 2

## Troubleshooting

### Common Issues

1. **Model not found**: Run `ml_anomaly_detector.py` first to train and save the model
2. **Data loading errors**: Ensure carOBD data files are in the correct directory
3. **Memory issues**: Reduce the number of files loaded or use data sampling

### Performance Issues

- **Low Recall**: Increase contamination parameter or improve built-in fault injection
- **High False Positives**: Decrease contamination parameter or add more training data
- **Poor Performance**: Check data quality and feature engineering

## Technical Details

### Overfitting Prevention

The system addresses overfitting through:

- **Proper Validation**: Train/validation split (80/20)
- **Realistic Fault Injection**: Automotive-specific failure patterns
- **Robust Parameters**: Conservative model settings
- **Feature Engineering**: Statistical features that generalize better

### Data Preprocessing

- **Calibration**: Coolant temperature conversion using `(raw + 40) * 2`
- **Missing Values**: Forward fill, backward fill, then median imputation
- **Scaling**: RobustScaler for outlier-resistant normalization

## Future Improvements

1. **Ensemble Methods**: Combine multiple algorithms for better performance
2. **Deep Learning**: Neural networks for complex temporal patterns
3. **Online Learning**: Adaptive models that learn from new data
4. **Feature Selection**: Automatic selection of most important features

## Contributing

When modifying the system:

1. Maintain the unified model approach
2. Keep realistic built-in fault injection patterns
3. Update documentation for any parameter changes
4. Test thoroughly with different datasets

## License

This project is part of a Final Year Project (FYP) for automotive anomaly detection research.
