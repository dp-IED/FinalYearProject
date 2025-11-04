# Deep Learning Anomaly Detection

This directory will contain PyTorch-based deep learning models for anomaly detection in automotive sensor data.

## Planned Models

### 1. LSTM (Long Short-Term Memory)
- **Purpose**: Capture temporal dependencies in sensor time series
- **Architecture**: LSTM layers for sequence modeling
- **Use Case**: Detect anomalies based on temporal patterns

### 2. CNN-LSTM Hybrid
- **Purpose**: Combine spatial feature extraction (CNN) with temporal modeling (LSTM)
- **Architecture**: 1D CNN layers followed by LSTM layers
- **Use Case**: Detect complex spatiotemporal anomalies

### 3. Autoencoder
- **Purpose**: Learn normal patterns and flag reconstruction errors
- **Architecture**: Encoder-decoder with bottleneck
- **Use Case**: Unsupervised anomaly detection

## Research Findings

Based on recent research:
- **LSTM/CNN models** often outperform traditional ML (Isolation Forest, One-Class SVM) for sensor anomaly detection
- **1D-CNN** achieved F1=73% vs SVM/IF in railway sensor data
- **CNN-LSTM hybrid** with attention reached 98.2% accuracy in cloud data
- **IoT networks**: CNN achieved 91.2% precision vs 85.4% for SVM

## Implementation Status

🚧 **Under Development** - Models will be implemented using PyTorch

## Future Structure

```
deep-learning/
├── README.md (this file)
├── models/
│   ├── lstm.py          # LSTM model implementation
│   ├── cnn_lstm.py      # CNN-LSTM hybrid model
│   └── autoencoder.py   # Autoencoder model
├── training/
│   ├── train_lstm.py    # LSTM training script
│   ├── train_cnn_lstm.py # CNN-LSTM training script
│   └── train_autoencoder.py # Autoencoder training script
├── utils/
│   ├── data_loader.py   # PyTorch DataLoader for sequences
│   └── evaluation.py    # Evaluation metrics for deep learning models
└── notebooks/
    └── experiments.ipynb # Experimentation notebook
```

## Comparison with Traditional ML

| Model Type | Expected F1 | Training Time | Data Requirements |
|-----------|------------|---------------|-------------------|
| Isolation Forest | 0.38-0.53 | < 1 min | Low |
| One-Class SVM | 0.25-0.40 | < 1 min | Low |
| **LSTM** | 0.50-0.70 | 5-30 min | Medium |
| **CNN-LSTM** | 0.60-0.80 | 10-60 min | Medium-High |
| **Autoencoder** | 0.45-0.65 | 5-20 min | Medium |

## Integration

These models will:
- Use the same data pipeline as traditional ML models
- Support auto-detection (threshold optimization)
- Be evaluated using the same metrics (F1, Precision, Recall, AUC)
- Support ensemble methods with traditional ML models

