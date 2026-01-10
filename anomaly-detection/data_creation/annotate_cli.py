#!/usr/bin/env python3
"""
OBD-II Time Series Anomaly Annotation CLI Tool

Interactive command-line interface for labeling anomaly candidates from GDN model.

Keyboard Controls:
  0 or n  - Label as Normal
  1 or a  - Label as Anomalous
  u       - Mark as Uncertain (skip)
  b       - Go to previous window
  s       - Save progress
  q       - Quit and save
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AnnotationState:
    """Manages annotation state and persistence."""
    
    def __init__(self, cand_df: pd.DataFrame, test_df: pd.DataFrame, scores: np.ndarray, 
                 save_path: str = "carobd_annotation_progress.csv"):
        self.cand_df = cand_df.copy()
        self.test_df = test_df
        self.scores = scores
        self.save_path = save_path
        self.current_idx = 0
        self.labels = {}
        self.history = []
        self.save_interval = 10
        self.annotation_count = 0
        
        # Initialize label column
        if "label" not in self.cand_df.columns:
            self.cand_df["label"] = np.nan
        
        self.load_progress()
    
    def load_progress(self):
        """Load existing annotations from CSV if available."""
        if Path(self.save_path).exists():
            try:
                saved_df = pd.read_csv(self.save_path)
                if "label" in saved_df.columns:
                    for idx, row in saved_df.iterrows():
                        if not pd.isna(row.get("label")):
                            self.labels[int(row.get("window_idx", idx))] = int(row["label"])
                            if idx < len(self.cand_df):
                                self.cand_df.loc[idx, "label"] = int(row["label"])
                    labeled = len(self.labels)
                    total = len(self.cand_df)
                    print(f"✓ Loaded {labeled} previous annotations")
            except Exception as e:
                print(f"⚠ Could not load previous progress: {e}")
    
    def set_label(self, label_value: int):
        """Record a label for current window."""
        self.labels[self.current_idx] = label_value
        self.cand_df.loc[self.current_idx, "label"] = label_value
        self.history.append((self.current_idx, label_value))
        self.annotation_count += 1
        
        if self.annotation_count % self.save_interval == 0:
            self.save_progress()
    
    def next_window(self) -> bool:
        """Advance to next window."""
        if self.current_idx < len(self.cand_df) - 1:
            self.current_idx += 1
            return True
        return False
    
    def prev_window(self) -> bool:
        """Go back to previous window."""
        if self.current_idx > 0:
            self.current_idx -= 1
            return True
        return False
    
    def get_progress(self) -> Tuple[int, int]:
        """Get labeling progress."""
        labeled = len([v for v in self.labels.values() if not pd.isna(v)])
        total = len(self.cand_df)
        return labeled, total
    
    def save_progress(self):
        """Save annotations to CSV."""
        save_cols = ["start", "end", "gdn_score", "rules", "suggested_sensors", "label"]
        save_cols = [c for c in save_cols if c in self.cand_df.columns]
        
        save_df = self.cand_df[save_cols].copy()
        save_df.index.name = "window_idx"
        save_df.to_csv(self.save_path)
        
        labeled, total = self.get_progress()
        print(f"\n✓ Saved progress: {labeled}/{total} labeled ({100*labeled/total:.1f}%)\n")


class AnnotationInterface:
    """CLI interface for annotation."""
    
    KEY_PIDS = [
        "ENGINE_RPM ()",
        "VEHICLE_SPEED ()",
        "THROTTLE ()",
        "ENGINE_LOAD ()",
        "COOLANT_TEMPERATURE ()",
        "LONG_TERM_FUEL_TRIM_BANK_1 ()"
    ]
    
    CRITERIA = {
        "LTFT": "Long-term Fuel Trim should be ±8% normally, >±10% sustained = anomaly",
        "STFT": "Short-term Fuel Trim should be ±10% normally, >±15% sustained = anomaly",
        "Coolant Temp": "Should reach 85-95°C in 5-8 min, <70°C after 5min run = anomaly",
        "RPM-Speed": "Should be proportional; decoupled >10s = anomaly",
        "Throttle-Load": "Should be correlated; high load + low throttle = anomaly",
        "Engine Load": "0-100% varying with speed; stuck at extremes = anomaly"
    }
    
    def __init__(self, state: AnnotationState):
        self.state = state
    
    def print_header(self):
        """Print banner."""
        print("\n" + "="*80)
        print("OBD-II TIME SERIES ANOMALY ANNOTATION CLI")
        print("="*80)
        print("\nControls:")
        print("  [0/n] Normal    [1/a] Anomaly    [u] Uncertain    [b] Back")
        print("  [s] Save        [q] Quit")
        print("="*80 + "\n")
    
    def print_criteria(self):
        """Print diagnostic criteria reference."""
        print("\n📋 OBD DIAGNOSTIC CRITERIA:")
        print("-" * 80)
        for name, desc in self.CRITERIA.items():
            print(f"  • {name:20s} {desc}")
        print("-" * 80 + "\n")
    
    def plot_window(self, idx: int):
        """Plot the current window with anomaly scores."""
        row = self.state.cand_df.iloc[idx]
        
        # Skip startup windows
        if "note" in row and pd.isna(row.get("start")):
            print(f"⊘ Skipping window {idx}: {row['note']}\n")
            return False
        
        start = int(row["start"])
        end = int(row["end"])
        wdf = self.state.test_df.iloc[start:end]
        window_scores = self.state.scores[start:end]
        
        fig, axes = plt.subplots(len(self.KEY_PIDS) + 1, 1, figsize=(14, 10), sharex=True)
        
        # Plot each PID
        for i, pid in enumerate(self.KEY_PIDS):
            if pid in wdf.columns:
                axes[i].plot(wdf[pid].values, linewidth=1.5, color="steelblue")
                axes[i].set_ylabel(pid.replace(" ()", ""), fontsize=9)
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f"Column '{pid}' not found", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_ylabel(pid.replace(" ()", ""), fontsize=9)
        
        # Plot anomaly scores
        axes[-1].plot(window_scores, linewidth=1.5, color="red", alpha=0.7, label="GDN Score")
        p95 = np.percentile(window_scores, 95)
        axes[-1].axhline(p95, color="orange", linestyle="--", label=f"95th percentile ({p95:.4f})")
        axes[-1].set_ylabel("Anomaly Score", fontsize=9)
        axes[-1].set_xlabel("Timestep (1 Hz)", fontsize=9)
        axes[-1].legend(loc="upper right", fontsize=8)
        axes[-1].grid(True, alpha=0.3)
        
        title = f"Window {idx + 1} / {len(self.state.cand_df)} | "
        title += f"start={start}, end={end} | GDN Score={row['gdn_score']:.4f}"
        fig.suptitle(title, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return True
    
    def print_window_info(self, idx: int):
        """Print info about current window."""
        row = self.state.cand_df.iloc[idx]
        labeled, total = self.state.get_progress()
        
        print(f"\n📊 Window {idx + 1} / {total} | Progress: {labeled}/{total} ({100*labeled/total:.1f}%)")
        print("-" * 80)
        
        if "note" in row and pd.isna(row.get("start")):
            print(f"Note: {row['note']}")
        else:
            start = int(row["start"]) if not pd.isna(row.get("start")) else "N/A"
            end = int(row["end"]) if not pd.isna(row.get("end")) else "N/A"
            print(f"  Indices:        start={start}, end={end}")
            print(f"  GDN Score:      {row['gdn_score']:.4f}")
            print(f"  Rules Fired:    {row.get('rules', 'None') or 'None'}")
            print(f"  Suggested Sensors: {row.get('suggested_sensors', 'None') or 'None'}")
            print(f"  Label Status:   {'✓ Already labeled' if not pd.isna(row.get('label')) else '⚪ Pending'}")
        
        print("-" * 80 + "\n")
    
    def get_user_choice(self) -> Optional[str]:
        """Get user input with validation."""
        valid_inputs = ['0', 'n', '1', 'a', 'u', 'b', 's', 'q']
        
        while True:
            try:
                choice = input("Your choice: ").lower().strip()
                if choice in valid_inputs:
                    return choice
                else:
                    print(f"Invalid input. Valid options: {', '.join(valid_inputs)}")
            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving...")
                self.state.save_progress()
                sys.exit(0)
            except EOFError:
                print("\n\nEOF reached. Saving...")
                self.state.save_progress()
                sys.exit(0)
    
    def run(self):
        """Main annotation loop."""
        self.print_header()
        self.print_criteria()
        
        while True:
            idx = self.state.current_idx
            
            # Print window info
            self.print_window_info(idx)
            
            # Plot window
            if not self.plot_window(idx):
                # Skip startup windows
                if self.state.next_window():
                    continue
                else:
                    print("✓ All windows processed!")
                    self.state.save_progress()
                    return
            
            # Get user input
            choice = self.get_user_choice()
            
            if choice in ['0', 'n']:
                self.state.set_label(0)
                print("✓ Labeled as NORMAL")
                if not self.state.next_window():
                    print("\n✓ All windows annotated!")
                    self.state.save_progress()
                    return
            
            elif choice in ['1', 'a']:
                self.state.set_label(1)
                print("✓ Labeled as ANOMALOUS")
                if not self.state.next_window():
                    print("\n✓ All windows annotated!")
                    self.state.save_progress()
                    return
            
            elif choice == 'u':
                print("⊗ Skipped (marked as uncertain)")
                if not self.state.next_window():
                    print("\n✓ All windows processed!")
                    self.state.save_progress()
                    return
            
            elif choice == 'b':
                if self.state.prev_window():
                    print("← Returned to previous window")
                else:
                    print("Already at first window")
            
            elif choice == 's':
                self.state.save_progress()
            
            elif choice == 'q':
                self.state.save_progress()
                print("✓ Saved and quit")
                return


def main():
    parser = argparse.ArgumentParser(
        description="Annotate OBD-II time series anomaly candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python annotate_cli.py /path/to/candidates.csv /path/to/test.csv /path/to/scores.npy
  python annotate_cli.py candidates.csv test.csv scores.npy --output my_annotations.csv
        """
    )
    
    parser.add_argument("candidates_csv", help="Path to candidates CSV file")
    parser.add_argument("test_csv", help="Path to test data CSV file")
    parser.add_argument("scores_npy", help="Path to anomaly scores NPY file")
    parser.add_argument("--output", default="carobd_annotation_progress.csv",
                       help="Output CSV file for annotations (default: carobd_annotation_progress.csv)")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    try:
        cand_df = pd.read_csv(args.candidates_csv)
        test_df = pd.read_csv(args.test_csv)
        test_df.columns = [c.strip() for c in test_df.columns]
        scores = np.load(args.scores_npy)
        print(f"✓ Loaded {len(cand_df)} candidates")
        print(f"✓ Loaded test data with shape {test_df.shape}")
        print(f"✓ Loaded scores with shape {scores.shape}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Initialize and run
    state = AnnotationState(cand_df, test_df, scores, save_path=args.output)
    interface = AnnotationInterface(state)
    interface.run()


if __name__ == "__main__":
    main()



