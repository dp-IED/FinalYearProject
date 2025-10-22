"""
Enhanced Main Script for Fault Injection Dataset Generation
Creates validated datasets with configurable anomaly percentages and precise tracking.
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any, Optional
import argparse

def run_enhanced_dataset_generation(
    idle_anomaly_percentage: float = 0.05,
    motion_anomaly_percentage: float = 0.05,
    injection_strategy: str = 'random',
    data_path: str = "../data/carOBD/obdiidata",
    output_path: str = "fault_injected_datasets"
) -> bool:
    """Run the enhanced fault injection dataset generation."""
    print("Step 1: Generating enhanced fault-injected datasets...")
    print("-" * 60)
    print(f"Configuration:")
    print(f"  Idle anomaly percentage: {idle_anomaly_percentage:.1%}")
    print(f"  Motion anomaly percentage: {motion_anomaly_percentage:.1%}")
    print(f"  Injection strategy: {injection_strategy}")
    print(f"  Data path: {data_path}")
    print(f"  Output path: {output_path}")
    print("-" * 60)
    
    try:
        # Import and run the enhanced fault injection
        from fault_injection import create_enhanced_fault_injection_datasets
        
        results = create_enhanced_fault_injection_datasets(
            data_path=data_path,
            output_path=output_path,
            idle_anomaly_percentage=idle_anomaly_percentage,
            motion_anomaly_percentage=motion_anomaly_percentage,
            injection_strategy=injection_strategy
        )
        
        print("✓ Enhanced dataset generation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced dataset generation failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_validation_test() -> bool:
    """Run the validation test on generated datasets."""
    print("\nStep 2: Validating generated datasets...")
    print("-" * 60)
    
    # Check if validation script exists
    if not os.path.exists('validation_test.py'):
        print("⚠️  Validation script not found. Cannot validate against paper methodology.")
        print("   Enhanced datasets have been generated successfully!")
        return False
    
    try:
        result = subprocess.run([sys.executable, 'validation_test.py'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✓ Validation test completed!")
            print(result.stdout)
            return True
        else:
            print("✗ Validation test failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running validation test: {e}")
        return False

def check_enhanced_dataset_files(output_path: str = "fault_injected_datasets") -> bool:
    """Check if required enhanced dataset files exist."""
    required_files = [
        f'{output_path}/idle/idle_normal_train.csv',
        f'{output_path}/idle/idle_normal_test.csv',
        f'{output_path}/idle/idle_anomaly_injected.csv',
        f'{output_path}/idle/idle_combined.csv',
        f'{output_path}/idle/idle_injection_metadata.json',
        f'{output_path}/idle/idle_injection_records.csv',
        f'{output_path}/idle/idle_summary.json',
        f'{output_path}/motion/motion_normal_train.csv',
        f'{output_path}/motion/motion_normal_test.csv',
        f'{output_path}/motion/motion_anomaly_injected.csv',
        f'{output_path}/motion/motion_combined.csv',
        f'{output_path}/motion/motion_injection_metadata.json',
        f'{output_path}/motion/motion_injection_records.csv',
        f'{output_path}/motion/motion_summary.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Missing required enhanced dataset files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✓ All required enhanced dataset files found!")
        return True

def analyze_injection_results(output_path: str = "fault_injected_datasets") -> Dict[str, Any]:
    """Analyze the injection results and provide detailed statistics."""
    print("\nStep 3: Analyzing injection results...")
    print("-" * 60)
    
    results = {}
    
    for mode in ['idle', 'motion']:
        mode_path = f"{output_path}/{mode}"
        
        # Load summary
        summary_file = f"{mode_path}/{mode}_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                results[mode] = summary
                
                print(f"\n{mode.upper()} Mode Results:")
                print(f"  Total samples: {summary['total_samples']:,}")
                print(f"  Normal train: {summary['normal_train_samples']:,}")
                print(f"  Normal test: {summary['normal_test_samples']:,}")
                print(f"  Anomaly samples: {summary['anomaly_samples']:,}")
                print(f"  Injection percentage: {summary['injection_percentage']:.2%}")
                print(f"  Level distribution: {summary['level_distribution']}")
        
        # Load injection records
        records_file = f"{mode_path}/{mode}_injection_records.csv"
        if os.path.exists(records_file):
            import pandas as pd
            records_df = pd.read_csv(records_file)
            
            print(f"  Injection Records:")
            print(f"    Total injections: {len(records_df)}")
            print(f"    SNR levels used: {records_df['snr_db'].unique()}")
            print(f"    Injection types: {records_df['injection_type'].value_counts().to_dict()}")
            
            # Calculate noise statistics
            noise_stats = records_df['noise_added'].describe()
            print(f"    Noise statistics:")
            print(f"      Mean: {noise_stats['mean']:.4f}")
            print(f"      Std: {noise_stats['std']:.4f}")
            print(f"      Min: {noise_stats['min']:.4f}")
            print(f"      Max: {noise_stats['max']:.4f}")
    
    return results

def cleanup_old_results(output_path: str = "fault_injected_datasets"):
    """Clean up any old results."""
    import shutil
    
    if os.path.exists(output_path):
        print("Cleaning up old results...")
        shutil.rmtree(output_path)
    
    if os.path.exists('results'):
        shutil.rmtree('results')

def main():
    """Main function with configurable parameters."""
    parser = argparse.ArgumentParser(description='Enhanced Fault Injection Dataset Generator')
    parser.add_argument('--idle-percentage', type=float, default=0.05, 
                       help='Percentage of anomalies for idle mode (0.0 to 1.0)')
    parser.add_argument('--motion-percentage', type=float, default=0.05,
                       help='Percentage of anomalies for motion mode (0.0 to 1.0)')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'sequential', 'clustered'],
                       help='Injection strategy')
    parser.add_argument('--data-path', type=str, default='../data/carOBD/obdiidata',
                       help='Path to carOBD data')
    parser.add_argument('--output-path', type=str, default='fault_injected_datasets',
                       help='Output path for datasets')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation test')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old results before starting')
    
    args = parser.parse_args()
    
    print("Enhanced Fault Injection Dataset Generator & Validator")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Idle anomaly percentage: {args.idle_percentage:.1%}")
    print(f"  Motion anomaly percentage: {args.motion_percentage:.1%}")
    print(f"  Injection strategy: {args.strategy}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output path: {args.output_path}")
    print("=" * 70)
    
    # Clean up old results if requested
    if args.cleanup:
        cleanup_old_results(args.output_path)
    
    # Step 1: Generate enhanced datasets
    if not run_enhanced_dataset_generation(
        idle_anomaly_percentage=args.idle_percentage,
        motion_anomaly_percentage=args.motion_percentage,
        injection_strategy=args.strategy,
        data_path=args.data_path,
        output_path=args.output_path
    ):
        print("\n✗ Enhanced dataset generation failed. Exiting.")
        return False
    
    # Check if files were created
    if not check_enhanced_dataset_files(args.output_path):
        print("\n✗ Required enhanced dataset files not found. Exiting.")
        return False
    
    # Step 2: Analyze injection results
    analysis_results = analyze_injection_results(args.output_path)
    
    # Step 3: Validate datasets against paper methodology
    if not run_validation_test():
        print("\n✗ Validation test failed. Enhanced system may need adjustment.")
        print("   However, the enhanced tracking and configurable percentages are working correctly.")
        return False
    
    print(f"\n{'='*70}")
    print("SUCCESS! Enhanced fault-injected datasets are ready!")
    print("=" * 70)
    print("Generated datasets:")
    print(f"  - {args.output_path}/idle/ (idle mode data)")
    print(f"  - {args.output_path}/motion/ (motion mode data)")
    print("\nEach dataset contains:")
    print("  - normal_train.csv: Normal training data")
    print("  - normal_test.csv: Normal test data")
    print("  - anomaly_injected.csv: Data with injected anomalies")
    print("  - combined.csv: All data combined")
    print("  - injection_metadata.json: Detailed injection metadata")
    print("  - injection_records.csv: Precise injection records")
    print("  - summary.json: Dataset summary statistics")
    print(f"\nInjection Configuration:")
    print(f"  Idle anomalies: {args.idle_percentage:.1%}")
    print(f"  Motion anomalies: {args.motion_percentage:.1%}")
    print(f"  Strategy: {args.strategy}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
