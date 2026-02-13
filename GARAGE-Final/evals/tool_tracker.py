import json
from pathlib import Path
from typing import Dict, List, Optional


class ToolTracker:
    def __init__(self):
        self.tool_calls: List[Dict] = []
        self.window_results: List[Dict] = []

    def record_tool_call(
        self,
        tool_name: str,
        step_id: Optional[int] = None,
        params: Optional[Dict] = None,
        result: Optional[str] = None,
        window_idx: int = -1,
    ):
        self.tool_calls.append({
            "tool_name": tool_name,
            "step_id": step_id,
            "params": params or {},
            "result_preview": (result[:200] + "...") if result and len(result) > 200 else result,
            "window_idx": window_idx,
        })

    def mark_window_result(
        self,
        window_idx: int,
        is_correct: bool,
        predicted_sensors: List[str],
        true_sensors: List[str],
        predicted_window_label: int,
        true_window_label: int,
    ):
        self.window_results.append({
            "window_idx": window_idx,
            "is_correct": is_correct,
            "predicted_sensors": predicted_sensors,
            "true_sensors": true_sensors,
            "predicted_window_label": predicted_window_label,
            "true_window_label": true_window_label,
        })

    def print_summary(self):
        correct = sum(1 for r in self.window_results if r["is_correct"])
        total = len(self.window_results)
        print(f"\nTool Usage Summary: {len(self.tool_calls)} tool calls, {correct}/{total} windows correct")

    def save_report(self, path: str):
        data = {
            "tool_calls": self.tool_calls,
            "window_results": self.window_results,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
