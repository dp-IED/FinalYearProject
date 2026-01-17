"""
Run the complete evaluation pipeline: dataset creation -> LLM eval -> GDN->KG eval -> comparison.

This script orchestrates the entire evaluation workflow.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete evaluation pipeline: dataset -> LLM -> GDN->KG -> comparison'
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION PIPELINE")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results directory: {args.results_dir}")
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
    
    if not run_command(gdn_kg_cmd, "Evaluating GDN->KG Method"):
        print("\n⚠️  GDN->KG evaluation failed.")
        return 1
    
    # Step 4: Compare methods
    if llm_results_path.exists() and gdn_kg_results_path.exists():
        comparison_path = Path(args.results_dir) / f"comparison_{args.split}.html"
        
        compare_cmd = [
            'python', 'llm/evaluation/compare_methods.py',
            '--llm-results', str(llm_results_path),
            '--gdn-kg-results', str(gdn_kg_results_path),
            '--output', str(comparison_path),
            '--json-output', str(Path(args.results_dir) / f"comparison_{args.split}.json")
        ]
        
        if not run_command(compare_cmd, "Comparing Methods"):
            print("\n⚠️  Comparison failed.")
            return 1
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nResults:")
        print(f"  - LLM Baseline: {llm_results_path}")
        print(f"  - GDN->KG: {gdn_kg_results_path}")
        print(f"  - Comparison: {comparison_path}")
        print()
        
        return 0
    else:
        print("\n⚠️  Cannot compare methods - missing result files")
        if not llm_results_path.exists():
            print(f"  Missing: {llm_results_path}")
        if not gdn_kg_results_path.exists():
            print(f"  Missing: {gdn_kg_results_path}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
