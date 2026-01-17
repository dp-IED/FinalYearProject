"""
Test LLM prompt with a known anomalous window to debug parsing issues.
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.evaluation.evaluate_llm_baseline import (
    format_window_for_llm,
    call_llm,
    parse_llm_response,
    load_llm_model
)

# Load dataset
dataset_path = Path('llm/evaluation/shared_dataset/test.npz')
data = np.load(dataset_path, allow_pickle=True)

unnormalized_windows = data['unnormalized_windows']
sensor_labels_true = data['sensor_labels']
window_labels_true = data['window_labels']
fault_types_true = data['fault_types']

# Load metadata
metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
sensor_names = metadata['dataset_info']['sensor_names']

# Find a known faulty window
faulty_indices = np.where(window_labels_true > 0)[0]
if len(faulty_indices) == 0:
    print("No faulty windows found!")
    exit(1)

# Use first faulty window
window_idx = faulty_indices[0]
print("="*80)
print(f"Testing LLM Prompt with Known Anomalous Window #{window_idx}")
print("="*80)
print(f"\nGround Truth:")
print(f"  Window label: {window_labels_true[window_idx]} (1=faulty)")
print(f"  Fault type: {fault_types_true[window_idx]}")
print(f"  Faulty sensors:")
for i, sensor_name in enumerate(sensor_names):
    if sensor_labels_true[window_idx, i] > 0:
        print(f"    - {sensor_name}")

# Get statistical features if available
statistical_features = data.get('statistical_features', None)
stats = statistical_features[window_idx] if statistical_features is not None else None

# Format prompt
print("\n" + "="*80)
print("FORMATTED PROMPT:")
print("="*80)
prompt = format_window_for_llm(
    unnormalized_windows[window_idx],
    sensor_names,
    stats,
    use_statistical_features=True
)
print(prompt[:2000])  # Print first 2000 chars
print("\n... (truncated)")

# Load LLM
print("\n" + "="*80)
print("Loading LLM...")
print("="*80)
model, tokenizer = load_llm_model("mlx-community/granite-4.0-h-micro-4bit")

# Call LLM
print("\n" + "="*80)
print("LLM RESPONSE (Raw):")
print("="*80)
response = call_llm(prompt, model, tokenizer, max_tokens=512)
print(response)
print("\n" + "="*80)
print(f"Response length: {len(response)} characters")

# Parse response
print("\n" + "="*80)
print("PARSED PREDICTION:")
print("="*80)
prediction = parse_llm_response(response, sensor_names)
print(f"Window label: {prediction['window_label']}")
print(f"Fault type: {prediction['fault_type']}")
print(f"Faulty sensors:")
for i, sensor_name in enumerate(sensor_names):
    if prediction['sensor_labels'][i] > 0:
        print(f"    - {sensor_name}")
print(f"\nReasoning: {prediction['reasoning'][:200]}...")

# Compare
print("\n" + "="*80)
print("COMPARISON:")
print("="*80)
print(f"Window label match: {prediction['window_label'] == window_labels_true[window_idx]}")
print(f"Fault type match: {prediction['fault_type'] == fault_types_true[window_idx]}")
sensor_match = np.allclose(prediction['sensor_labels'], sensor_labels_true[window_idx])
print(f"Sensor labels match: {sensor_match}")

if not sensor_match:
    print("\nSensor label differences:")
    for i, sensor_name in enumerate(sensor_names):
        pred_val = prediction['sensor_labels'][i]
        true_val = sensor_labels_true[window_idx, i]
        if pred_val != true_val:
            print(f"  {sensor_name}: predicted={pred_val}, true={true_val}")
