#!/usr/bin/env python3
"""
Global Staged Training Pipeline
Orchestrates all three stages of GDN training sequentially.

Stages:
1. Stage 1: Graph Structure Learning (Self-Supervised Forecasting)
2. Stage 2: Embedding Space Refinement (Supervised Contrastive)
3. Stage 3: Fine-tuning (Task-Specific)
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# Pipeline Configuration
# ============================================================================

DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_DATA_PATH = "/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata"

# Stage configurations
STAGE_CONFIGS = {
    1: {
        "script": "train_gdn_stage1_forecast.py",
        "default_epochs": 75,
        "checkpoint_name": "stage1_best_forecast.pt",
        "description": "Graph Structure Learning (Self-Supervised)",
    },
    2: {
        "script": "train_gdn_stage2_embedding.py",
        "default_epochs": 40,
        "checkpoint_name": "stage2_best_embedding.pt",
        "description": "Embedding Space Refinement (Supervised Contrastive)",
        "requires_previous": 1,
    },
    3: {
        "script": "train_gdn_stage3_finetune.py",
        "default_epochs": 25,
        "checkpoint_name": "stage3_best_finetune.pt",
        "description": "Fine-tuning (Task-Specific)",
        "requires_previous": 2,
    },
}


# ============================================================================
# Pipeline Functions
# ============================================================================


def get_script_path(script_name):
    """Get absolute path to training script."""
    script_dir = Path(__file__).parent
    script_path = script_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return str(script_path)


def run_stage(stage_num, args, checkpoint_dir):
    """
    Run a single training stage.

    Args:
        stage_num: Stage number (1, 2, or 3)
        args: Parsed command-line arguments
        checkpoint_dir: Directory for checkpoints

    Returns:
        Path to checkpoint created by this stage, or None if failed
    """
    config = STAGE_CONFIGS[stage_num]
    script_path = get_script_path(config["script"])

    print(f"\n{'=' * 80}")
    print(f"STAGE {stage_num}: {config['description']}")
    print(f"{'=' * 80}\n")

    # Build command
    cmd = [sys.executable, script_path]

    # Common arguments
    cmd.extend(["--data_path", args.data_path])
    cmd.extend(["--checkpoint_dir", checkpoint_dir])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--lr", str(args.learning_rate)])

    if args.device:
        cmd.extend(["--device", args.device])

    # Stage-specific arguments
    if stage_num == 1:
        cmd.extend(["--epochs", str(args.stage1_epochs or config["default_epochs"])])

    elif stage_num == 2:
        # Require Stage 1 checkpoint
        stage1_checkpoint = os.path.join(
            checkpoint_dir, STAGE_CONFIGS[1]["checkpoint_name"]
        )
        if not os.path.exists(stage1_checkpoint):
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {stage1_checkpoint}\n"
                "Please run Stage 1 first or provide --stage1_checkpoint"
            )
        cmd.extend(["--stage1_checkpoint", stage1_checkpoint])
        cmd.extend(["--epochs", str(args.stage2_epochs or config["default_epochs"])])
        if args.lambda_center is not None:
            cmd.extend(["--lambda_center", str(args.lambda_center)])
        if args.lambda_scl is not None:
            cmd.extend(["--lambda_scl", str(args.lambda_scl)])

    elif stage_num == 3:
        # Require Stage 2 checkpoint
        stage2_checkpoint = os.path.join(
            checkpoint_dir, STAGE_CONFIGS[2]["checkpoint_name"]
        )
        if not os.path.exists(stage2_checkpoint):
            raise FileNotFoundError(
                f"Stage 2 checkpoint not found: {stage2_checkpoint}\n"
                "Please run Stage 2 first or provide --stage2_checkpoint"
            )
        cmd.extend(["--stage2_checkpoint", stage2_checkpoint])
        cmd.extend(["--epochs", str(args.stage3_epochs or config["default_epochs"])])
        if args.lambda_global is not None:
            cmd.extend(["--lambda_global", str(args.lambda_global)])
        if args.lambda_forecast is not None:
            cmd.extend(["--lambda_forecast", str(args.lambda_forecast)])
        if args.use_forecast:
            cmd.append("--use_forecast")

    # Run stage
    print(f"Running: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode != 0:
            print(f"\n✗ Stage {stage_num} failed with return code {result.returncode}")
            return None

        # Check if checkpoint was created
        checkpoint_path = os.path.join(checkpoint_dir, config["checkpoint_name"])
        if os.path.exists(checkpoint_path):
            print(f"\n✓ Stage {stage_num} completed successfully")
            print(f"  Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
        else:
            print(f"\n⚠ Stage {stage_num} completed but checkpoint not found: {checkpoint_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Stage {stage_num} failed: {e}")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠ Stage {stage_num} interrupted by user")
        return None


def run_pipeline(args):
    """
    Run the complete staged training pipeline.

    Args:
        args: Parsed command-line arguments
    """
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create run log
    run_log = {
        "start_time": datetime.now().isoformat(),
        "stages": [],
        "checkpoints": {},
    }

    print(f"\n{'=' * 80}")
    print("STAGED GDN TRAINING PIPELINE")
    print(f"{'=' * 80}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device or 'auto'}")
    print(f"{'=' * 80}\n")

    # Determine which stages to run
    if args.start_stage:
        start_stage = args.start_stage
        end_stage = args.end_stage or 3
    else:
        start_stage = 1
        end_stage = 3

    stages_to_run = list(range(start_stage, end_stage + 1))

    print(f"Running stages: {stages_to_run}\n")

    # Run each stage
    for stage_num in stages_to_run:
        stage_start_time = datetime.now()

        try:
            checkpoint_path = run_stage(stage_num, args, checkpoint_dir)

            stage_end_time = datetime.now()
            duration = (stage_end_time - stage_start_time).total_seconds()

            stage_log = {
                "stage": stage_num,
                "description": STAGE_CONFIGS[stage_num]["description"],
                "start_time": stage_start_time.isoformat(),
                "end_time": stage_end_time.isoformat(),
                "duration_seconds": duration,
                "success": checkpoint_path is not None,
                "checkpoint": checkpoint_path,
            }

            run_log["stages"].append(stage_log)
            if checkpoint_path:
                run_log["checkpoints"][f"stage{stage_num}"] = checkpoint_path

            if checkpoint_path is None:
                print(f"\n⚠ Stage {stage_num} did not produce a checkpoint")
                if args.stop_on_error:
                    print("Stopping pipeline due to error (--stop_on_error)")
                    break

        except Exception as e:
            print(f"\n✗ Stage {stage_num} failed with exception: {e}")
            stage_log = {
                "stage": stage_num,
                "description": STAGE_CONFIGS[stage_num]["description"],
                "start_time": stage_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
            }
            run_log["stages"].append(stage_log)

            if args.stop_on_error:
                print("Stopping pipeline due to error (--stop_on_error)")
                break

    # Save run log
    run_log["end_time"] = datetime.now().isoformat()
    total_duration = (
        datetime.fromisoformat(run_log["end_time"])
        - datetime.fromisoformat(run_log["start_time"])
    ).total_seconds()
    run_log["total_duration_seconds"] = total_duration

    log_path = os.path.join(checkpoint_dir, "pipeline_run_log.json")
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"\n{'=' * 80}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total duration: {total_duration / 60:.2f} minutes")
    print(f"Stages completed: {sum(1 for s in run_log['stages'] if s.get('success'))}/{len(run_log['stages'])}")
    print(f"Run log saved: {log_path}")
    print(f"{'=' * 80}\n")

    # Print checkpoint summary
    if run_log["checkpoints"]:
        print("Checkpoints created:")
        for stage_key, checkpoint_path in run_log["checkpoints"].items():
            print(f"  {stage_key}: {checkpoint_path}")
        print()


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Staged GDN Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages sequentially
  python train_gdn_staged_pipeline.py

  # Run only Stage 1
  python train_gdn_staged_pipeline.py --start_stage 1 --end_stage 1

  # Resume from Stage 2
  python train_gdn_staged_pipeline.py --start_stage 2

  # Custom epochs per stage
  python train_gdn_staged_pipeline.py --stage1_epochs 100 --stage2_epochs 50 --stage3_epochs 30
        """,
    )

    # Data and paths
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints",
    )

    # Stage control
    parser.add_argument(
        "--start_stage",
        type=int,
        choices=[1, 2, 3],
        help="Start from this stage (default: 1)",
    )
    parser.add_argument(
        "--end_stage",
        type=int,
        choices=[1, 2, 3],
        help="End at this stage (default: 3)",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop pipeline if any stage fails",
    )

    # Common hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for all stages"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3, Stage 3 uses 1e-4)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cpu/cuda, default: auto)"
    )

    # Stage-specific epochs
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=None,
        help=f"Epochs for Stage 1 (default: {STAGE_CONFIGS[1]['default_epochs']})",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=None,
        help=f"Epochs for Stage 2 (default: {STAGE_CONFIGS[2]['default_epochs']})",
    )
    parser.add_argument(
        "--stage3_epochs",
        type=int,
        default=None,
        help=f"Epochs for Stage 3 (default: {STAGE_CONFIGS[3]['default_epochs']})",
    )

    # Stage 2 hyperparameters
    parser.add_argument(
        "--lambda_center",
        type=float,
        default=None,
        help="Center loss weight for Stage 2 (default: 0.8)",
    )
    parser.add_argument(
        "--lambda_scl",
        type=float,
        default=None,
        help="Supervised contrastive loss weight for Stage 2 (default: 0.5)",
    )

    # Stage 3 hyperparameters
    parser.add_argument(
        "--lambda_global",
        type=float,
        default=None,
        help="Global loss weight for Stage 3 (default: 0.3)",
    )
    parser.add_argument(
        "--lambda_forecast",
        type=float,
        default=None,
        help="Forecast loss weight for Stage 3 (default: 0.1)",
    )
    parser.add_argument(
        "--use_forecast",
        action="store_true",
        help="Use forecast regularization in Stage 3",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.start_stage and args.end_stage and args.start_stage > args.end_stage:
        parser.error("--start_stage must be <= --end_stage")

    # Run pipeline
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
