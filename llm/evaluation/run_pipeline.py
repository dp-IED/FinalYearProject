"""
Run the complete evaluation pipeline: dataset creation -> LLM eval -> GDN->KG eval -> GDN->KG->LLM eval -> KAG v1 -> KAG v2 -> comparison.

This script orchestrates the entire evaluation workflow with progress visualization.
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import time
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ProgressTracker:
    """Track progress across multiple evaluation steps."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []

    def start_step(self, step_name: str):
        """Mark the start of a new step."""
        self.current_step += 1
        step_start = time.time()
        print("\n" + "=" * 80)
        print(f"STEP {self.current_step}/{self.total_steps}: {step_name}")
        print("=" * 80)
        print(f"[{self._get_progress_bar()}] {self.current_step}/{self.total_steps}")
        print()
        return step_start

    def end_step(self, step_start: float, success: bool = True):
        """Mark the end of a step."""
        step_time = time.time() - step_start
        self.step_times.append(step_time)
        elapsed = time.time() - self.start_time

        status = "✓" if success else "✗"
        print(f"\n{status} Step completed in {step_time:.1f}s (Total: {elapsed:.1f}s)")
        return success

    def _get_progress_bar(self, width: int = 40) -> str:
        """Generate a progress bar."""
        filled = int(width * self.current_step / self.total_steps)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def print_summary(self):
        """Print final summary."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Total steps: {self.current_step}/{self.total_steps}")
        print(f"Total time: {total_time:.1f}s")
        if self.step_times:
            print(
                f"Average step time: {sum(self.step_times) / len(self.step_times):.1f}s"
            )
            print(f"Longest step: {max(self.step_times):.1f}s")
            print(f"Shortest step: {min(self.step_times):.1f}s")
        print("=" * 80)


def run_command(
    cmd: List[str], description: str, tracker: ProgressTracker = None
) -> Tuple[bool, float]:
    """Run a command and handle errors. Streams output in real-time for progress bars."""
    step_start = tracker.start_step(description) if tracker else time.time()

    if tracker:
        print(f"Command: {' '.join(cmd)}")
        print()
        sys.stdout.flush()  # Ensure header is printed before command output

    # Use Popen to stream output in real-time so progress bars work
    # Combine stderr with stdout so all output appears together
    # Set PYTHONUNBUFFERED to ensure child Python processes output immediately
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        env=env,  # Pass environment with PYTHONUNBUFFERED
    )

    # Stream output line by line to preserve progress bars and show all output
    output_lines = []
    try:
        # Read line by line and print immediately
        # Use iter() with readline() for better real-time behavior
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            # Write and flush immediately
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
    except BrokenPipeError:
        # Process may have closed stdout, continue
        pass
    except Exception as e:
        print(f"\n⚠️  Error reading output: {e}", file=sys.stderr)
        sys.stderr.flush()

    # Wait for process to complete
    returncode = process.wait()

    # Ensure all output is flushed
    sys.stdout.flush()

    success = returncode == 0
    if tracker:
        tracker.end_step(step_start, success)
    else:
        if not success:
            print(f"\n❌ Error in {description} (exit code: {returncode})")
            # Show last few lines of output for debugging
            if output_lines:
                print("\nLast output lines:")
                output_text = ''.join(output_lines)
                lines = output_text.split('\n')
                for line in lines[-10:]:
                    print(line)

    return success, time.time() - step_start


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline: dataset -> LLM -> GDN->KG -> GDN->KG->LLM -> comparison"
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="data/carOBD/obdiidata",
        help="Path to directory containing raw OBD CSV files",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="anomaly-detection/best_focal_multilabel_gdn.pt",
        help="Path to GDN model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llm/evaluation/shared_dataset",
        help="Output directory for shared dataset",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--skip-dataset-creation",
        action="store_true",
        help="Skip dataset creation (use existing dataset)",
    )
    parser.add_argument(
        "--skip-llm-baseline",
        action="store_true",
        help="Skip LLM baseline evaluation (use existing results)",
    )
    parser.add_argument(
        "--skip-gdn-kg",
        action="store_true",
        help="Skip GDN->KG evaluation (use existing results)",
    )
    parser.add_argument(
        "--skip-kg-llm",
        action="store_true",
        help="Skip GDN->KG->LLM evaluation (faster, but no KG-enhanced LLM comparison)",
    )
    parser.add_argument(
        "--skip-kag-v1", action="store_true", help="Skip KAG v1 (heuristic) evaluation"
    )
    parser.add_argument(
        "--skip-kag-v2",
        action="store_true",
        help="Skip KAG v2 (LLM-planned) evaluation",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://127.0.0.1:7687",
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j-user", type=str, default="neo4j", help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password", type=str, default="password", help="Neo4j password"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (useful for testing). Applied to dataset creation and all evaluations.",
    )

    args = parser.parse_args()

    # Count total steps for progress tracking
    total_steps = 1  # Dataset creation (if not skipped)
    if not args.skip_dataset_creation:
        total_steps += 0  # Already counted
    if not args.skip_llm_baseline:
        total_steps += 1  # LLM baseline
    if not args.skip_gdn_kg:
        total_steps += 1  # GDN->KG
    if not args.skip_kg_llm:
        total_steps += 1  # GDN->KG->LLM
    if not args.skip_kag_v1:
        total_steps += 1  # KAG v1
    if not args.skip_kag_v2:
        total_steps += 1  # KAG v2
    total_steps += 1  # Comparison

    tracker = ProgressTracker(total_steps)

    print("=" * 80)
    print("EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Neo4j URI: {args.neo4j_uri}")
    if args.limit is not None:
        print(f"⚠️  LIMIT MODE: Processing only {args.limit} windows")
    print(f"\nTotal steps: {total_steps}")
    print()

    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.output_dir) / f"{args.split}.npz"

    # Step 1: Create shared dataset
    if not args.skip_dataset_creation:
        cmd = [
            "python",
            "llm/evaluation/create_shared_dataset.py",
            "--raw-data-path",
            args.raw_data_path,
            "--output-dir",
            args.output_dir,
            "--split",
            args.split,
            "--gdn-model-path",
            args.model_path,
            "--neo4j-uri",
            args.neo4j_uri,
            "--neo4j-user",
            args.neo4j_user,
            "--neo4j-password",
            args.neo4j_password,
        ]
        if args.limit is not None:
            cmd.extend(["--max-windows", str(args.limit)])
        success, _ = run_command(cmd, "Creating Shared Dataset", tracker)
        if not success:
            print("\n⚠️  Dataset creation failed.")
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                print("Cannot proceed without dataset.")
                return 1
            print("Using existing dataset...")
    else:
        tracker.current_step += 1  # Skip counting this step

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please create the dataset first or provide --normalized-path")
        return 1

    print(f"\n✓ Using dataset: {dataset_path}")

    # Step 2: Evaluate LLM baseline
    llm_results_path = Path(args.results_dir) / f"llm_baseline_{args.split}.json"

    if not args.skip_llm_baseline:
        llm_cmd = [
            "python",
            "llm/evaluation/evaluate_llm_baseline.py",
            "--dataset",
            str(dataset_path),
            "--output",
            str(llm_results_path),
        ]
        if args.limit is not None:
            llm_cmd.extend(["--limit", str(args.limit)])

        success, _ = run_command(llm_cmd, "Evaluating LLM Baseline", tracker)
        if not success:
            print("\n⚠️  LLM evaluation failed. Continuing with GDN->KG evaluation...")
    else:
        print("\n⏭️  Skipping LLM baseline evaluation (--skip-llm-baseline flag set)")
        tracker.current_step += 1  # Skip counting this step

    # Resolve model path (needed for GDN->KG, GDN->KG->LLM, and KAG v2)
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try alternative locations
        alt_paths = [Path("anomaly-detection") / args.model_path, Path(args.model_path)]
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break

        if not model_path.exists():
            print(f"\n⚠️  Model not found: {args.model_path}")
            print("Available models:")
            for pt_file in Path("anomaly-detection").glob("*.pt"):
                print(f"  - {pt_file}")
            if not args.skip_gdn_kg:
                print("\nSkipping GDN->KG evaluation...")
                return 0

    # Step 3: Evaluate GDN->KG
    gdn_kg_results_path = Path(args.results_dir) / f"gdn_kg_{args.split}.json"

    if not args.skip_gdn_kg:

        gdn_kg_cmd = [
            "python",
            "llm/evaluation/evaluate_gdn_kg.py",
            "--dataset",
            str(dataset_path),
            "--model-path",
            str(model_path),
            "--output",
            str(gdn_kg_results_path),
            "--device",
            "cpu",  # Use CPU by default
        ]
        if args.limit is not None:
            gdn_kg_cmd.extend(["--limit", str(args.limit)])

        success, _ = run_command(gdn_kg_cmd, "Evaluating GDN->KG Method", tracker)
        if not success:
            print("\n⚠️  GDN->KG evaluation failed.")
            return 1
    else:
        print("\n⏭️  Skipping GDN->KG evaluation (--skip-gdn-kg flag set)")
        tracker.current_step += 1  # Skip counting this step

    # Step 4: Evaluate GDN->KG->LLM
    gdn_kg_llm_results_path = Path(args.results_dir) / f"gdn_kg_llm_{args.split}.json"

    if not args.skip_kg_llm:
        gdn_kg_llm_cmd = [
            "python",
            "llm/evaluation/evaluate_gdn_kg_llm.py",
            "--dataset",
            str(dataset_path),
            "--model-path",
            str(model_path),
            "--output",
            str(gdn_kg_llm_results_path),
            "--device",
            "cpu",  # Use CPU by default
        ]
        if args.limit is not None:
            gdn_kg_llm_cmd.extend(["--limit", str(args.limit)])

        success, _ = run_command(
            gdn_kg_llm_cmd, "Evaluating GDN->KG->LLM Method", tracker
        )
        if not success:
            print("\n⚠️  GDN->KG->LLM evaluation failed. Continuing with comparison...")
    else:
        print("\n⏭️  Skipping GDN->KG->LLM evaluation (--skip-kg-llm flag set)")
        tracker.current_step += 1  # Skip counting this step

    # Step 5: Evaluate KAG v1 (Heuristic)
    kag_v1_results_path = Path(args.results_dir) / f"kag_v1_{args.split}.json"

    if not args.skip_kag_v1:
        kag_v1_cmd = [
            "python",
            "llm/evaluation/evaluate_kag_v1.py",
            "--dataset",
            str(dataset_path),
            "--output",
            str(kag_v1_results_path),
            "--neo4j-uri",
            args.neo4j_uri,
            "--neo4j-user",
            args.neo4j_user,
            "--neo4j-password",
            args.neo4j_password,
        ]
        if args.limit is not None:
            kag_v1_cmd.extend(["--limit", str(args.limit)])

        success, _ = run_command(kag_v1_cmd, "Evaluating KAG v1 (Heuristic)", tracker)
        if not success:
            print("\n⚠️  KAG v1 evaluation failed. Continuing...")
    else:
        print("\n⏭️  Skipping KAG v1 evaluation (--skip-kag-v1 flag set)")
        tracker.current_step += 1  # Skip counting this step

    # Step 6: Evaluate KAG v2 (LLM-planned)
    kag_v2_results_path = Path(args.results_dir) / f"kag_v2_{args.split}.json"

    if not args.skip_kag_v2:
        kag_v2_cmd = [
            "python",
            "llm/evaluation/evaluate_kag_v2.py",
            "--dataset",
            str(dataset_path),
            "--gdn-model",
            str(model_path),
            "--output",
            str(kag_v2_results_path),
            "--neo4j-uri",
            args.neo4j_uri,
            "--neo4j-user",
            args.neo4j_user,
            "--neo4j-password",
            args.neo4j_password,
            "--device",
            "cpu",  # Use CPU by default
        ]
        if args.limit is not None:
            kag_v2_cmd.extend(["--limit", str(args.limit)])

        success, _ = run_command(kag_v2_cmd, "Evaluating KAG v2 (LLM-planned)", tracker)
        if not success:
            print("\n⚠️  KAG v2 evaluation failed. Continuing...")
    else:
        print("\n⏭️  Skipping KAG v2 evaluation (--skip-kag-v2 flag set)")
        tracker.current_step += 1  # Skip counting this step

    # Step 7: Compare methods
    comparison_path = Path(args.results_dir) / f"comparison_{args.split}.html"

    # Determine which comparison to run based on available results
    has_llm = llm_results_path.exists()
    has_gdn_kg = gdn_kg_results_path.exists() if not args.skip_gdn_kg else False
    has_gdn_kg_llm = gdn_kg_llm_results_path.exists() if not args.skip_kg_llm else False
    has_kag_v1 = kag_v1_results_path.exists() if not args.skip_kag_v1 else False
    has_kag_v2 = kag_v2_results_path.exists() if not args.skip_kag_v2 else False

    # Build comparison command - include all available methods
    if not (has_llm and has_gdn_kg and has_gdn_kg_llm):
        print("\n⚠️  Cannot compare methods - missing required result files")
        if not has_llm:
            print(f"  Missing: {llm_results_path}")
        if not has_gdn_kg:
            print(f"  Missing: {gdn_kg_results_path}")
        if not has_gdn_kg_llm and not args.skip_kg_llm:
            print(f"  Missing: {gdn_kg_llm_results_path}")
        tracker.print_summary()
        return 1

    # Build comparison command with all available methods
    compare_cmd = [
        "python",
        "llm/evaluation/compare_methods.py",
        "--llm-results",
        str(llm_results_path),
        "--gdn-kg-results",
        str(gdn_kg_results_path),
        "--gdn-kg-llm-results",
        str(gdn_kg_llm_results_path),
        "--output",
        str(comparison_path),
        "--json-output",
        str(Path(args.results_dir) / f"comparison_{args.split}.json"),
    ]

    # Add optional KAG results if available
    if has_kag_v1:
        compare_cmd.extend(["--kag-v1-results", str(kag_v1_results_path)])
    if has_kag_v2:
        compare_cmd.extend(["--kag-v2-results", str(kag_v2_results_path)])

    success, _ = run_command(compare_cmd, "Comparing Methods", tracker)
    if not success:
        print("\n⚠️  Comparison failed.")
        tracker.print_summary()
        return 1

    tracker.print_summary()

    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - LLM Baseline: {llm_results_path}")
    print(f"  - GDN->KG: {gdn_kg_results_path}")
    if has_gdn_kg_llm:
        print(f"  - GDN->KG->LLM: {gdn_kg_llm_results_path}")
    if has_kag_v1:
        print(f"  - KAG v1 (Heuristic): {kag_v1_results_path}")
    if has_kag_v2:
        print(f"  - KAG v2 (LLM-planned): {kag_v2_results_path}")
    print(f"  - Comparison Report: {comparison_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
