"""
Run the complete evaluation pipeline: dataset creation -> LLM eval -> GDN->KG eval -> GDN->KG->LLM eval -> comparison.

This script orchestrates the entire evaluation workflow.
"""

import argparse
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_command(cmd, description):
    """Run a command and handle errors. Streams output in real-time for progress bars."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Use Popen to stream output in real-time so progress bars work
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Stream output line by line to preserve progress bars
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print immediately (line already has newline)
        output_lines.append(line)
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ Error in {description}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete evaluation pipeline: dataset -> LLM -> GDN->KG -> GDN->KG->LLM -> comparison'
    )
    parser.add_argument(
        '--raw-data-path',
        type=str,
        default='data/carOBD/obdiidata',
        help='Path to directory containing raw OBD CSV files'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='anomaly-detection/best_focal_multilabel_gdn.pt',
        help='Path to GDN model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='llm/evaluation/shared_dataset',
        help='Output directory for shared dataset'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for evaluation results'
    )
    parser.add_argument(
        '--skip-dataset-creation',
        action='store_true',
        help='Skip dataset creation (use existing dataset)'
    )
    parser.add_argument(
        '--skip-kg-llm',
        action='store_true',
        help='Skip GDN->KG->LLM evaluation (faster, but no KG-enhanced LLM comparison)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of windows to process (useful for testing). Applied to dataset creation and all evaluations.'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION PIPELINE")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results directory: {args.results_dir}")
    if args.limit is not None:
        print(f"⚠️  LIMIT MODE: Processing only {args.limit} windows")
    print()
    
    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    dataset_path = Path(args.output_dir) / f"{args.split}.npz"
    
    # Step 1: Create shared dataset
    if not args.skip_dataset_creation:
        cmd = [
            'python', 'llm/evaluation/create_shared_dataset.py',
            '--raw-data-path', args.raw_data_path,
            '--output-dir', args.output_dir,
            '--split', args.split
        ]
        if args.limit is not None:
            cmd.extend(['--max-windows', str(args.limit)])
        if not run_command(cmd, "Creating Shared Dataset"):
            print("\n⚠️  Dataset creation failed.")
            if not dataset_path.exists():
                print(f"❌ Dataset not found: {dataset_path}")
                print("Cannot proceed without dataset.")
                return 1
            print("Using existing dataset...")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("Please create the dataset first or provide --normalized-path")
        return 1
    
    print(f"\n✓ Using dataset: {dataset_path}")
    
    # Step 2: Evaluate LLM baseline
    llm_results_path = Path(args.results_dir) / f"llm_baseline_{args.split}.json"
    
    llm_cmd = [
        'python', 'llm/evaluation/evaluate_llm_baseline.py',
        '--dataset', str(dataset_path),
        '--output', str(llm_results_path)
    ]
    if args.limit is not None:
        llm_cmd.extend(['--limit', str(args.limit)])
    
    if not run_command(llm_cmd, "Evaluating LLM Baseline"):
        print("\n⚠️  LLM evaluation failed. Continuing with GDN->KG evaluation...")
    
    # Step 3: Evaluate GDN->KG
    gdn_kg_results_path = Path(args.results_dir) / f"gdn_kg_{args.split}.json"
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try alternative locations
        alt_paths = [
            Path('anomaly-detection') / args.model_path,
            Path(args.model_path)
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {args.model_path}")
            print("Available models:")
            for pt_file in Path('anomaly-detection').glob('*.pt'):
                print(f"  - {pt_file}")
            print("\nSkipping GDN->KG evaluation...")
            return 0
    
    gdn_kg_cmd = [
        'python', 'llm/evaluation/evaluate_gdn_kg.py',
        '--dataset', str(dataset_path),
        '--model-path', str(model_path),
        '--output', str(gdn_kg_results_path),
        '--device', 'cpu'  # Use CPU by default
    ]
    if args.limit is not None:
        gdn_kg_cmd.extend(['--limit', str(args.limit)])
    
    if not run_command(gdn_kg_cmd, "Evaluating GDN->KG Method"):
        print("\n⚠️  GDN->KG evaluation failed.")
        return 1
    
    # Step 4: Evaluate GDN->KG->LLM
    gdn_kg_llm_results_path = Path(args.results_dir) / f"gdn_kg_llm_{args.split}.json"
    
    if not args.skip_kg_llm:
        gdn_kg_llm_cmd = [
            'python', 'llm/evaluation/evaluate_gdn_kg_llm.py',
            '--dataset', str(dataset_path),
            '--model-path', str(model_path),
            '--output', str(gdn_kg_llm_results_path),
            '--device', 'cpu'  # Use CPU by default
        ]
        if args.limit is not None:
            gdn_kg_llm_cmd.extend(['--limit', str(args.limit)])
        
        if not run_command(gdn_kg_llm_cmd, "Evaluating GDN->KG->LLM Method"):
            print("\n⚠️  GDN->KG->LLM evaluation failed. Continuing with comparison...")
    else:
        print("\n⏭️  Skipping GDN->KG->LLM evaluation (--skip-kg-llm flag set)")
    
    # Step 5: Compare methods
    comparison_path = Path(args.results_dir) / f"comparison_{args.split}.html"
    
    # Determine which comparison to run based on available results
    has_llm = llm_results_path.exists()
    has_gdn_kg = gdn_kg_results_path.exists()
    has_gdn_kg_llm = gdn_kg_llm_results_path.exists()
    
    if has_llm and has_gdn_kg and has_gdn_kg_llm:
        # Three-way comparison
        compare_cmd = [
            'python', 'llm/evaluation/compare_methods.py',
            '--llm-results', str(llm_results_path),
            '--gdn-kg-results', str(gdn_kg_results_path),
            '--gdn-kg-llm-results', str(gdn_kg_llm_results_path),
            '--output', str(comparison_path),
            '--json-output', str(Path(args.results_dir) / f"comparison_{args.split}.json")
        ]
    elif has_llm and has_gdn_kg:
        # Two-way comparison (fallback)
        compare_cmd = [
            'python', 'llm/evaluation/compare_methods.py',
            '--llm-results', str(llm_results_path),
            '--gdn-kg-results', str(gdn_kg_results_path),
            '--output', str(comparison_path),
            '--json-output', str(Path(args.results_dir) / f"comparison_{args.split}.json")
        ]
    else:
        print("\n⚠️  Cannot compare methods - missing result files")
        if not has_llm:
            print(f"  Missing: {llm_results_path}")
        if not has_gdn_kg:
            print(f"  Missing: {gdn_kg_results_path}")
        if not has_gdn_kg_llm and not args.skip_kg_llm:
            print(f"  Missing: {gdn_kg_llm_results_path}")
        return 1
    
    if not run_command(compare_cmd, "Comparing Methods"):
        print("\n⚠️  Comparison failed.")
        return 1
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults:")
    print(f"  - LLM Baseline: {llm_results_path}")
    print(f"  - GDN->KG: {gdn_kg_results_path}")
    if has_gdn_kg_llm:
        print(f"  - GDN->KG->LLM: {gdn_kg_llm_results_path}")
    print(f"  - Comparison: {comparison_path}")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
