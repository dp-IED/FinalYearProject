# Evaluation Module: LLM-only vs GDN->KG Comparison

This module provides tools for creating a shared evaluation dataset and comparing the diagnostic performance of two methods:

1. **LLM-only baseline**: Raw unnormalized OBD logs processed directly by LLM
2. **GDN->KG method**: Normalized windows processed through GDN and Knowledge Graph

## Overview

Both methods evaluate on the **same windows** with the **same ground truth labels** to ensure fair comparison.

## Dataset Structure

The shared evaluation dataset contains:

- **Normalized windows**: `(N, 300, 8)` array for GDN->KG method
- **Unnormalized windows**: `(N, 300, 8)` array for LLM-only method
- **Ground truth labels**: Sensor-level and window-level binary labels
- **Metadata**: Statistical features, window IDs, fault types, etc.

## Files

- `create_shared_dataset.py` - Creates shared evaluation dataset from gdn.ipynb outputs
- `evaluate_llm_baseline.py` - Evaluates LLM-only method
- `evaluate_gdn_kg.py` - Evaluates GDN->KG method
- `compare_methods.py` - Compares both methods and generates reports
- `metrics.py` - Evaluation metrics (precision, recall, F1, etc.)

## Usage

### 1. Create Shared Dataset

First, create the shared evaluation dataset from your gdn.ipynb outputs:

```bash
python llm/evaluation/create_shared_dataset.py \
    --normalized_path path/to/normalized_windows.pt \
    --raw_data_path data/carOBD/obdiidata/ \
    --output_dir llm/evaluation/shared_dataset/ \
    --split test
```

**Note**: This script currently uses placeholder data for unnormalized windows. For full implementation, integrate with `gdn.ipynb` to track window boundaries and load actual unnormalized data.

### 2. Evaluate LLM Baseline

Evaluate the LLM-only method:

```bash
python llm/evaluation/evaluate_llm_baseline.py \
    --dataset llm/evaluation/shared_dataset/test.npz \
    --output results/llm_baseline.json \
    --simulate  # Use --simulate for testing without actual LLM
```

### 3. Evaluate GDN->KG Method

Evaluate the GDN->KG method:

```bash
python llm/evaluation/evaluate_gdn_kg.py \
    --dataset llm/evaluation/shared_dataset/test.npz \
    --model-path anomaly-detection/best_focal_multilabel_gdn.pt \
    --output results/gdn_kg.json \
    --device cpu  # or 'cuda' if GPU available
```

### 4. Compare Methods

Compare both methods and generate reports:

```bash
python llm/evaluation/compare_methods.py \
    --llm-results results/llm_baseline.json \
    --gdn-kg-results results/gdn_kg.json \
    --output comparison_report.html \
    --json-output comparison_report.json
```

## Evaluation Metrics

The evaluation computes:

### Window-Level Metrics

- Accuracy: Did the method correctly identify faulty windows?
- Precision: Of predicted faulty windows, how many were actually faulty?
- Recall: Of actual faulty windows, how many were detected?
- F1 Score: Harmonic mean of precision and recall

### Sensor-Level Metrics

- Accuracy: Did the method correctly identify faulty sensors?
- Precision: Of predicted faulty sensors, how many were actually faulty?
- Recall: Of actual faulty sensors, how many were detected?
- F1 Score: Harmonic mean of precision and recall
- Per-sensor metrics: Individual metrics for each sensor

### Per-Fault-Type Metrics

- Metrics broken down by fault type (VSS_DROPOUT, MAF_SCALE_LOW, etc.)
- Helps identify which method performs better on specific fault types

### Efficiency Metrics

- Processing time per window
- Total processing time
- Windows processed per second
- For GDN->KG: Breakdown of GDN vs KG processing time

## Dataset Format

The shared dataset is saved in multiple formats:

### .npz Format (NumPy Archive)

- Fast loading for programmatic access
- Preserves data types
- Contains: `normalized_windows`, `unnormalized_windows`, `sensor_labels`, `window_labels`, `fault_types`, `statistical_features`

### JSON Metadata

- Human-readable metadata
- Contains: dataset info, sensor names, window IDs, etc.

### CSV Files

- One CSV file per window
- Contains unnormalized sensor values
- Useful for LLM text processing

## Integration Notes

### For LLM Evaluation

The LLM evaluator formats windows as text prompts. You can customize the prompt format in `evaluate_llm_baseline.py`:

- `format_window_for_llm()`: Formats a window for LLM prompt
- `simulate_llm_prediction()`: Placeholder for actual LLM integration

To integrate with an actual LLM:

1. Replace `simulate_llm_prediction()` with your LLM API call
2. Parse LLM response to extract sensor predictions
3. Update the prediction format to match expected output

### For GDN->KG Evaluation

The GDN->KG evaluator:

1. Loads trained GDN model
2. Processes normalized windows through GDN
3. Builds Knowledge Graph from GDN outputs
4. Extracts fault predictions from KG relationships

The prediction extraction logic in `extract_predictions_from_kg()` can be customized to use different KG query strategies.

## Example Output

The comparison report includes:

```
Window-Level Metrics:
Metric          LLM          GDN->KG      Difference    Improvement
accuracy        0.8500       0.9200       0.0700        8.24%
precision       0.7800       0.8900       0.1100        14.10%
recall          0.8200       0.9100       0.0900        10.98%
f1              0.7995       0.8995       0.1000        12.51%

Sensor-Level Metrics:
...
```

## Future Improvements

1. **Actual LLM Integration**: Replace simulation with real LLM API calls
2. **Window Boundary Tracking**: Integrate with gdn.ipynb to properly track window boundaries
3. **Statistical Significance Testing**: Add statistical tests to compare methods
4. **Visualization**: Add plots comparing metrics across fault types
5. **Cross-Validation**: Support k-fold cross-validation for more robust evaluation
