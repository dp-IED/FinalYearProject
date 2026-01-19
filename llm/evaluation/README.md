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

Create the shared evaluation dataset from raw OBD data using the same preprocessing pipeline as GDN:

```bash
python llm/evaluation/create_shared_dataset.py \
    --raw-data-path data/carOBD/obdiidata/ \
    --output-dir llm/evaluation/shared_dataset/ \
    --split test \
    --fault-percentage 0.3
```

This script:

- Loads raw OBD CSV files from the data directory
- Applies the exact same preprocessing as GDN training (deduplication, downsampling, cross-channel features)
- Creates both normalized windows (for GDN) and unnormalized windows (for LLM) from the same data
- Optionally injects faults with sensor-level labels
- Saves as `.npz` for fast loading and JSON metadata

**Optional arguments:**

- `--fault-percentage`: Percentage of windows to inject faults into (default: 0.3)
- `--max-windows`: Limit total windows (useful for quick testing)
- `--random-state`: Random seed for reproducibility

### 2. Evaluate LLM Baseline

Evaluate the LLM-only method:

```bash
python llm/evaluation/evaluate_llm_baseline.py \
    --dataset llm/evaluation/shared_dataset/test.npz \
    --output results/llm_baseline.json \
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

## How create_shared_dataset.py Works

The script creates datasets that are compatible with both the LLM baseline and GDN->KG methods:

1. **Data Loading**: Reads raw OBD CSV files from `data/carOBD/obdiidata/`
2. **Preprocessing**: Applies the identical pipeline as `train_gdn_center_loss.py`:
   - Removes zero-variance columns
   - Fills missing timestamps by averaging duplicates
   - Downsamples data
   - Filters out drives shorter than window_size + forecast_horizon
   - Adds cross-channel features for relationship detection
3. **Window Creation**:
   - **Normalized windows**: Created using `build_clean_windows()` (same as GDN training)
   - **Unnormalized windows**: Created directly from preprocessed data without normalization (for LLM)
   - Both use the same window boundaries for fair comparison
4. **Fault Injection** (optional):
   - Uses `inject_faults_with_sensor_labels()` to inject realistic faults
   - Provides sensor-level ground truth labels
5. **Statistical Features**: Computes 9 features per sensor (mean, std, min, max, range, median, mode, skewness, kurtosis)
6. **Output Format**:
   - `.npz` file: Fast binary format for programmatic access
   - JSON metadata: Human-readable dataset info and sensor names

This ensures both methods evaluate on identical windows with the same ground truth.

## Integration Notes

### For LLM Evaluation

The LLM evaluator formats windows as text prompts. You can customize the prompt format in `evaluate_llm_baseline.py`:

- `format_window_for_llm()`: Formats a window for LLM prompt
- `parse_llm_response()`: Parses LLM response to extract fault predictions

The evaluator uses the MLX LM library to run inference with the Granite model. The prediction format matches the expected output structure with sensor labels, fault types, and reasoning.

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

## How create_shared_dataset.py Compares to gdn.ipynb

Both follow the **same data processing pipeline** to ensure fair evaluation:

| Step                         | gdn.ipynb                     | create_shared_dataset.py          |
| ---------------------------- | ----------------------------- | --------------------------------- |
| Data loading                 | Manual CSV loading            | Automatic from directory          |
| Zero-variance removal        | ✓                             | ✓                                 |
| Duplicate timestamp handling | ✓                             | ✓                                 |
| Downsampling                 | ✓                             | ✓                                 |
| Drive filtering              | ✓                             | ✓                                 |
| Cross-channel features       | ✓                             | ✓                                 |
| Window creation              | Using `build_clean_windows()` | Using `build_clean_windows()`     |
| Normalization                | MinMax scaler                 | MinMax scaler (same)              |
| Unnormalized windows         | Not tracked                   | ✓ Created with same boundaries    |
| Fault injection              | Manual                        | Automated with labels             |
| Statistical features         | Manual                        | Automated (9 features per sensor) |

**Key difference**: The script **automatically creates unnormalized windows with matching boundaries** to the normalized windows, ensuring both LLM and GDN->KG methods evaluate on identical data.

## Typical Workflow

```bash
# 1. Create the shared evaluation dataset
python llm/evaluation/create_shared_dataset.py \
    --raw-data-path data/carOBD/obdiidata \
    --output-dir llm/evaluation/shared_dataset \
    --split test \
    --fault-percentage 0.3

# 2. Run the full evaluation pipeline
python llm/evaluation/run_pipeline.py \
    --split test \
    --skip-dataset-creation

# 3. Results are saved in results/ directory
# - llm_baseline_test.json
# - gdn_kg_test.json
# - comparison_test.html (visual report)
# - comparison_test.json (metrics table)
```

## Future Improvements

1. **Actual LLM Integration**: Replace simulation with real LLM API calls
2. **Statistical Significance Testing**: Add statistical tests to compare methods
3. **Visualization**: Add plots comparing metrics across fault types
4. **Cross-Validation**: Support k-fold cross-validation for more robust evaluation
5. **Batch Processing**: Support processing multiple splits in one command
