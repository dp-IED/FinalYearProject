# Models Directory

This directory contains the trained models for anomaly detection.

## Current Models

### ML Anomaly Detection Model

- **File**: `../models/unified_ml_detector.pkl`
- **Description**: Unified ML model for both idle and motion data using Isolation Forest and One-Class SVM
- **Performance**:
  - Isolation Forest: F1=0.448, Recall=0.390, Precision=0.527
  - One-Class SVM: F1=0.321, Recall=0.200, Precision=0.788
- **Usage**: Load with `MLAnomalyDetector.load_model("anomaly-detection/models/unified_ml_detector")`

### ARIMA Models (Legacy)

- **Directory**: `arima_models/`
- **Description**: ARIMA-based anomaly detection models for idle and motion data
- **Files**:
  - `arima_idle_model.pkl` - ARIMA model for idle data
  - `arima_motion_model.pkl` - ARIMA model for motion data
  - `arima_*_model_metadata.json` - Model metadata and parameters
- **Status**: Legacy models, kept for comparison purposes
- **Performance**: Lower performance compared to ML models (F1 ≈ 0.005-0.011)

## Model Training

To retrain the unified ML model:

```bash
python anomaly-detection/ml/ml_anomaly_detector.py
```

## File Sizes

- `unified_ml_detector.pkl`: ~186 MB
- `arima_idle_model.pkl`: ~478 MB
- `arima_motion_model.pkl`: ~1.5 GB

## Notes

- Models are excluded from git via `.gitignore` due to large file sizes
- Models can be regenerated from code using the training scripts
- ARIMA models are kept for historical comparison but are not actively used
