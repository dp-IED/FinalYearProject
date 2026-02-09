#!/usr/bin/env python3
"""
Complete Training Pipeline
Runs Stage 1 and Stage 2 training sequentially.

Usage:
    python train_pipeline.py --epochs1 5 --epochs2 2 --cpu_only
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Default paths
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_DATA_PATH = str(Path(__file__).parent / "data" / "carOBD" / "obdiidata")

# Stage 1 checkpoint name
STAGE1_CHECKPOINT_NAME = "stage1_best_forecast.pt"


def run_stage1(args):
    """Run Stage 1 training."""
    print("=" * 80)
    print("STAGE 1: Graph Structure Learning (Self-Supervised)")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "training" / "train_stage1.py"),
        "--data_path",
        args.data_path,
        "--epochs",
        str(args.epochs1),
        "--batch_size",
        str(args.batch_size),
        "--checkpoint_dir",
        args.checkpoint_dir,
    ]

    if args.cpu_only:
        cmd.append("--cpu_only")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.max_batches_per_epoch1:
        cmd.extend(["--max_batches_per_epoch", str(args.max_batches_per_epoch1)])
    if args.use_compile:
        cmd.append("--use_compile")
    if args.use_amp:
        cmd.append("--use_amp")
    if args.gradient_accumulation_steps:
        cmd.extend(
            ["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)]
        )

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Stage 1 training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Stage 1 training failed with exit code {e.returncode}")
        return False


def run_stage2(args, stage1_checkpoint_path):
    """Run Stage 2 training."""
    print("\n" + "=" * 80)
    print("STAGE 2: Supervised Center Loss Training")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "training" / "train_stage2.py"),
        "--data_path",
        args.data_path,
        "--stage1_checkpoint",
        stage1_checkpoint_path,
        "--epochs",
        str(args.epochs2),
        "--batch_size",
        str(args.batch_size),
        "--checkpoint_dir",
        args.checkpoint_dir,
    ]

    if args.cpu_only:
        cmd.append("--cpu_only")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.max_batches_per_epoch2:
        cmd.extend(["--max_batches_per_epoch", str(args.max_batches_per_epoch2)])
    if args.use_compile:
        cmd.append("--use_compile")
    if args.use_amp:
        cmd.append("--use_amp")
    if args.gradient_accumulation_steps:
        cmd.extend(
            ["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)]
        )
    if args.lambda_center:
        cmd.extend(["--lambda_center", str(args.lambda_center)])
    if args.lambda_global:
        cmd.extend(["--lambda_global", str(args.lambda_global)])
    if args.lr2:
        cmd.extend(["--lr", str(args.lr2)])

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Stage 2 training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Stage 2 training failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete Training Pipeline: Stage 1 + Stage 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (2 epochs each, CPU only)
  python train_pipeline.py --epochs1 2 --epochs2 2 --cpu_only --max_batches_per_epoch1 10 --max_batches_per_epoch2 10

  # Full training run
  python train_pipeline.py --epochs1 75 --epochs2 40

  # Custom checkpoint directory
  python train_pipeline.py --checkpoint_dir my_checkpoints --epochs1 5 --epochs2 2
        """,
    )

    # Data and checkpoint paths
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to data directory (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory to save checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )

    # Stage 1 arguments
    parser.add_argument(
        "--epochs1",
        type=int,
        default=75,
        help="Number of epochs for Stage 1 (default: 75)",
    )
    parser.add_argument(
        "--max_batches_per_epoch1",
        type=int,
        default=None,
        help="Limit number of batches per epoch for Stage 1 (for testing)",
    )

    # Stage 2 arguments
    parser.add_argument(
        "--epochs2",
        type=int,
        default=40,
        help="Number of epochs for Stage 2 (default: 40)",
    )
    parser.add_argument(
        "--max_batches_per_epoch2",
        type=int,
        default=None,
        help="Limit number of batches per epoch for Stage 2 (for testing)",
    )
    parser.add_argument(
        "--lambda_center",
        type=float,
        default=None,
        help="Center loss weight for Stage 2 (default: 0.5)",
    )
    parser.add_argument(
        "--lambda_global",
        type=float,
        default=None,
        help="Global loss weight for Stage 2 (default: 0.3)",
    )
    parser.add_argument(
        "--lr2",
        type=float,
        default=None,
        help="Learning rate for Stage 2 (default: 5e-4)",
    )

    # Common arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for both stages (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU usage (disable CUDA auto-detection)",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile() to optimize model (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for CUDA devices",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of batches to accumulate gradients before updating",
    )

    # Pipeline control
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip Stage 1 and use existing checkpoint",
    )
    parser.add_argument(
        "--skip_stage2",
        action="store_true",
        help="Skip Stage 2 (only run Stage 1)",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint (if skipping Stage 1 or using custom path)",
    )

    args = parser.parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Determine Stage 1 checkpoint path
    if args.stage1_checkpoint:
        stage1_checkpoint_path = args.stage1_checkpoint
    else:
        stage1_checkpoint_path = os.path.join(
            args.checkpoint_dir, STAGE1_CHECKPOINT_NAME
        )

    # Run Stage 1
    stage1_success = True
    if not args.skip_stage1:
        stage1_success = run_stage1(args)
        if not stage1_success:
            print("\n✗ Pipeline failed at Stage 1. Exiting.")
            sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("SKIPPING STAGE 1 (using existing checkpoint)")
        print("=" * 80)
        if not os.path.exists(stage1_checkpoint_path):
            print(
                f"\n✗ Stage 1 checkpoint not found: {stage1_checkpoint_path}\n"
                "Please provide --stage1_checkpoint or run Stage 1 first."
            )
            sys.exit(1)
        print(f"✓ Using Stage 1 checkpoint: {stage1_checkpoint_path}")

    # Run Stage 2
    if not args.skip_stage2:
        stage2_success = run_stage2(args, stage1_checkpoint_path)
        if not stage2_success:
            print("\n✗ Pipeline failed at Stage 2. Exiting.")
            sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("SKIPPING STAGE 2")
        print("=" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}/")
    print(f"  - Stage 1: {STAGE1_CHECKPOINT_NAME}")
    if not args.skip_stage2:
        print(f"  - Stage 2: stage2_best.pt")
    print()


if __name__ == "__main__":
    main()
