"""
Find specific windows where KAG v1 is correct but KAG v2 fails.
"""

import json
import numpy as np
import argparse
from pathlib import Path


def load_results(results_path: Path):
    """Load evaluation results."""
    with open(results_path, "r") as f:
        return json.load(f)


def load_dataset(dataset_path: Path):
    """Load the dataset to get true labels."""
    data = np.load(dataset_path, allow_pickle=True)
    return {
        "window_labels": data["window_labels"],
        "sensor_labels": data["sensor_labels"],
        "fault_types": data.get("fault_types", None),
    }


def find_failure_cases(
    v1_results_path: Path, v2_results_path: Path, dataset_path: Path
):
    """Find windows where v1 is correct but v2 fails."""

    # Load results
    v1_results = load_results(v1_results_path)
    v2_results = load_results(v2_results_path)
    dataset = load_dataset(dataset_path)

    # Get predictions
    v1_window_pred = np.array(v1_results["predictions"]["window_labels"])
    v2_window_pred = np.array(v2_results["predictions"]["window_labels"])
    v1_sensor_pred = np.array(v1_results["predictions"]["sensor_labels"])
    v2_sensor_pred = np.array(v2_results["predictions"]["sensor_labels"])

    # Get true labels
    true_window_labels = dataset["window_labels"]
    true_sensor_labels = dataset["sensor_labels"]

    # Limit to the number of windows we actually evaluated
    num_evaluated = min(
        len(v1_window_pred), len(v2_window_pred), len(true_window_labels)
    )
    true_window_labels = true_window_labels[:num_evaluated]
    true_sensor_labels = true_sensor_labels[:num_evaluated]
    v1_window_pred = v1_window_pred[:num_evaluated]
    v2_window_pred = v2_window_pred[:num_evaluated]
    v1_sensor_pred = v1_sensor_pred[:num_evaluated]
    v2_sensor_pred = v2_sensor_pred[:num_evaluated]

    # Find windows where v1 is correct and v2 fails
    failure_cases = []

    for window_idx in range(num_evaluated):
        v1_correct = v1_window_pred[window_idx] == true_window_labels[window_idx]
        v2_correct = v2_window_pred[window_idx] == true_window_labels[window_idx]

        # Check if v1 is correct and v2 fails
        if v1_correct and not v2_correct:
            true_label = true_window_labels[window_idx]
            v1_pred = v1_window_pred[window_idx]
            v2_pred = v2_window_pred[window_idx]

            # Get sensor-level details
            true_sensors = np.where(true_sensor_labels[window_idx] == 1)[0]
            v1_sensors = np.where(v1_sensor_pred[window_idx] == 1)[0]
            v2_sensors = np.where(v2_sensor_pred[window_idx] == 1)[0]

            # Get fault type if available
            fault_type = None
            if dataset["fault_types"] is not None:
                fault_type = dataset["fault_types"][window_idx]

            failure_cases.append(
                {
                    "window_idx": int(window_idx),
                    "true_window_label": str(true_label),
                    "v1_predicted": str(v1_pred),
                    "v2_predicted": str(v2_pred),
                    "v1_correct": bool(v1_correct),
                    "v2_correct": bool(v2_correct),
                    "true_faulty_sensors": true_sensors.tolist()
                    if len(true_sensors) > 0
                    else [],
                    "v1_predicted_sensors": v1_sensors.tolist()
                    if len(v1_sensors) > 0
                    else [],
                    "v2_predicted_sensors": v2_sensors.tolist()
                    if len(v2_sensors) > 0
                    else [],
                    "fault_type": str(fault_type) if fault_type is not None else None,
                }
            )

    return failure_cases


def format_case(case, sensor_names):
    """Format a failure case for display."""
    true_sensors_str = (
        ", ".join([sensor_names[i] for i in case["true_faulty_sensors"]])
        if len(case["true_faulty_sensors"]) > 0
        else "None"
    )
    v1_sensors_str = (
        ", ".join([sensor_names[i] for i in case["v1_predicted_sensors"]])
        if len(case["v1_predicted_sensors"]) > 0
        else "None"
    )
    v2_sensors_str = (
        ", ".join([sensor_names[i] for i in case["v2_predicted_sensors"]])
        if len(case["v2_predicted_sensors"]) > 0
        else "None"
    )

    return f"""
Window {case["window_idx"]}:
  True Label: {case["true_window_label"]}
  Fault Type: {case["fault_type"] if case["fault_type"] else "N/A"}
  True Faulty Sensors: {true_sensors_str}
  
  KAG v1 (Heuristic):
    Predicted: {case["v1_predicted"]} ✓ CORRECT
    Predicted Sensors: {v1_sensors_str}
  
  KAG v2 (LLM-Planned):
    Predicted: {case["v2_predicted"]} ✗ INCORRECT
    Predicted Sensors: {v2_sensors_str}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Find windows where KAG v1 is correct but v2 fails"
    )
    parser.add_argument(
        "--v1-results", type=str, required=True, help="Path to KAG v1 results JSON"
    )
    parser.add_argument(
        "--v2-results", type=str, required=True, help="Path to KAG v2 results JSON"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset .npz file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional: Save results to JSON file"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Finding Failure Cases: KAG v1 Correct, KAG v2 Failed")
    print("=" * 80)
    print()

    # Load sensor names
    dataset_path = Path(args.dataset)
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]

    # Find failure cases
    failure_cases = find_failure_cases(
        Path(args.v1_results), Path(args.v2_results), Path(args.dataset)
    )

    print(f"Found {len(failure_cases)} windows where v1 is correct but v2 fails:")
    print()

    # Display first 2 cases
    for i, case in enumerate(failure_cases[:2]):
        print(f"Case {i + 1}:")
        print(format_case(case, sensor_names))
        print("-" * 80)

    if len(failure_cases) > 2:
        print(f"\n... and {len(failure_cases) - 2} more cases")

    # Save to JSON if requested
    if args.output:
        output_data = {
            "total_cases": len(failure_cases),
            "cases": failure_cases,
            "sensor_names": sensor_names,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
