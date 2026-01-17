"""
Test LLM with a simpler, more direct prompt.
"""

import numpy as np
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.evaluation.evaluate_llm_baseline import call_llm, load_llm_model

# Load dataset
dataset_path = Path('llm/evaluation/shared_dataset/test.npz')
data = np.load(dataset_path, allow_pickle=True)

unnormalized_windows = data['unnormalized_windows']
sensor_labels_true = data['sensor_labels']
window_labels_true = data['window_labels']
fault_types_true = data['fault_types']

metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
sensor_names = metadata['dataset_info']['sensor_names']

# Find a known faulty window
faulty_indices = np.where(window_labels_true > 0)[0]
window_idx = faulty_indices[0]

print("="*80)
print("Testing Simplified Prompt")
print("="*80)
print(f"\nGround Truth: Window {window_idx} is faulty")
print(f"Fault type: {fault_types_true[window_idx]}")
print(f"Faulty sensor: VEHICLE_SPEED")

# Create simpler prompt
window_data = unnormalized_windows[window_idx]
# Sample just a few key timesteps
sample_indices = [0, 50, 100, 150, 200, 250, 299]

simple_prompt = """You are analyzing automotive sensor data to detect faults.

Sensor readings at key timesteps:
"""
for t in sample_indices:
    simple_prompt += f"\nTime {t}:\n"
    for i, sensor_name in enumerate(sensor_names):
        value = window_data[t, i]
        simple_prompt += f"  {sensor_name}: {value:.2f}\n"

simple_prompt += """
Task: Identify if VEHICLE_SPEED sensor is faulty.

Look for:
- Sudden drops to zero
- Unusual patterns
- Values that don't match expected behavior

Respond in this exact format:
Faulty Sensors: [VEHICLE_SPEED] or None
Fault Type: [VSS_DROPOUT] or None
Reasoning: [your analysis]
"""

print("\n" + "="*80)
print("SIMPLE PROMPT:")
print("="*80)
print(simple_prompt)

# Load LLM
print("\n" + "="*80)
print("LLM RESPONSE:")
print("="*80)
model, tokenizer = load_llm_model("mlx-community/granite-4.0-h-micro-4bit")

response = call_llm(simple_prompt, model, tokenizer, max_tokens=256)
print(response)
print(f"\nResponse length: {len(response)} characters")
print(f"First 200 chars: {response[:200]}")

# Test with even simpler prompt
print("\n" + "="*80)
print("TESTING EVEN SIMPLER PROMPT:")
print("="*80)

very_simple = """Analyze this automotive sensor data:

VEHICLE_SPEED readings: """
# Show VEHICLE_SPEED values
speed_values = window_data[:, 1]  # VEHICLE_SPEED is index 1
simple_values = [f"{v:.1f}" for v in speed_values[::30]]  # Every 30th value
very_simple += ", ".join(simple_values[:10])

very_simple += """

Question: Is the VEHICLE_SPEED sensor showing a dropout fault (sudden drops to zero)?

Answer format:
Faulty Sensors: VEHICLE_SPEED or None
Fault Type: VSS_DROPOUT or None
Reasoning: [brief explanation]
"""

print(very_simple)
response2 = call_llm(very_simple, model, tokenizer, max_tokens=200)
print("\n" + "="*80)
print("RESPONSE:")
print("="*80)
print(response2)

# Test parsing
print("\n" + "="*80)
print("TESTING PARSING:")
print("="*80)
from llm.evaluation.evaluate_llm_baseline import parse_llm_response

parsed = parse_llm_response(response2, sensor_names)
print(f"Parsed window label: {parsed['window_label']}")
print(f"Parsed fault type: {parsed['fault_type']}")
print(f"Parsed faulty sensors:")
for i, sensor_name in enumerate(sensor_names):
    if parsed['sensor_labels'][i] > 0:
        print(f"  - {sensor_name}")
print(f"\nExpected: VEHICLE_SPEED should be detected")
print(f"Actual: VEHICLE_SPEED detected = {parsed['sensor_labels'][1] > 0}")
