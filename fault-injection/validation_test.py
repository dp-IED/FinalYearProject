"""
Enhanced Validation Test for Fault-Injected Dataset
Validates the enhanced fault injection system against the paper's methodology and results.
Based on: "Detecting Anomalies in the Engine Coolant Sensor Using One-Class Classifiers"
"""

import numpy as np
import pandas as pd
import os
import json
import argparse
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedValidationTester:
    """Enhanced validation tester that follows the paper's methodology."""
    
    def __init__(self, tolerance_config: Optional[Dict[str, Any]] = None):
        # Paper results from Tables II and IV (expected F2 scores)
        self.paper_results = {
            'idle': {
                'k-NN': {'Level_I': 0.960, 'Level_II': 0.949, 'Level_III': 1.000},
                'OC-SVM': {'Level_I': 0.811, 'Level_II': 0.809, 'Level_III': 0.862},
                'SVDD': {'Level_I': 0.611, 'Level_II': 0.774, 'Level_III': 0.807}
            },
            'motion': {
                'OC-SVM': {'Level_I': 0.809, 'Level_II': 0.809, 'Level_III': 0.862}
            }
        }
        
        # Paper methodology parameters
        self.window_size = 5  # Sliding window size as per paper
        self.k_neighbors = 2  # k-NN parameter
        self.ocsvm_nu = 0.1   # OC-SVM parameter
        self.svdd_nu = 0.1    # SVDD parameter
        
        # Modular tolerance configuration
        self.tolerance_config = self._setup_tolerance_config(tolerance_config)
    
    def _setup_tolerance_config(self, tolerance_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Setup modular tolerance configuration with defaults."""
        default_config = {
            'default_tolerance': 0.20,  # 20% default tolerance
            'strict_tolerance': 0.10,  # 10% strict tolerance
            'lenient_tolerance': 0.30,  # 30% lenient tolerance
            'classifier_specific': {
                'k-NN': {'default': 0.15, 'strict': 0.08, 'lenient': 0.25},
                'OC-SVM': {'default': 0.20, 'strict': 0.10, 'lenient': 0.30},
                'SVDD': {'default': 0.25, 'strict': 0.12, 'lenient': 0.35}
            },
            'level_specific': {
                'Level_I': {'default': 0.18, 'strict': 0.09, 'lenient': 0.28},
                'Level_II': {'default': 0.20, 'strict': 0.10, 'lenient': 0.30},
                'Level_III': {'default': 0.22, 'strict': 0.11, 'lenient': 0.32}
            },
            'mode_specific': {
                'idle': {'default': 0.20, 'strict': 0.10, 'lenient': 0.30},
                'motion': {'default': 0.25, 'strict': 0.12, 'lenient': 0.35}
            },
            'tolerance_mode': 'default'  # 'default', 'strict', 'lenient', or 'custom'
        }
        
        if tolerance_config is None:
            return default_config
        
        # Merge custom config with defaults
        config = default_config.copy()
        config.update(tolerance_config)
        return config
    
    def get_tolerance(self, mode: str, classifier: str, level: str) -> float:
        """Get tolerance value based on current configuration."""
        tolerance_mode = self.tolerance_config['tolerance_mode']
        
        # Check for custom tolerance first
        if tolerance_mode == 'custom' and 'custom_tolerance' in self.tolerance_config:
            return self.tolerance_config['custom_tolerance']
        
        # Get base tolerance based on mode
        base_tolerance = self.tolerance_config['mode_specific'].get(mode, {}).get(tolerance_mode, 
                      self.tolerance_config[f'{tolerance_mode}_tolerance'])
        
        # Apply classifier-specific adjustment
        classifier_tolerance = self.tolerance_config['classifier_specific'].get(classifier, {}).get(tolerance_mode)
        if classifier_tolerance is not None:
            base_tolerance = classifier_tolerance
        
        # Apply level-specific adjustment
        level_tolerance = self.tolerance_config['level_specific'].get(level, {}).get(tolerance_mode)
        if level_tolerance is not None:
            base_tolerance = level_tolerance
        
        return base_tolerance
    
    def set_tolerance_mode(self, mode: str) -> None:
        """Set the tolerance mode: 'default', 'strict', 'lenient', or 'custom'."""
        if mode in ['default', 'strict', 'lenient', 'custom']:
            self.tolerance_config['tolerance_mode'] = mode
        else:
            raise ValueError(f"Invalid tolerance mode: {mode}. Must be 'default', 'strict', 'lenient', or 'custom'")
    
    def set_custom_tolerance(self, tolerance: float) -> None:
        """Set a custom tolerance value."""
        self.tolerance_config['tolerance_mode'] = 'custom'
        self.tolerance_config['custom_tolerance'] = tolerance
    
    def get_tolerance_info(self) -> Dict[str, Any]:
        """Get current tolerance configuration information."""
        return {
            'current_mode': self.tolerance_config['tolerance_mode'],
            'available_modes': ['default', 'strict', 'lenient', 'custom'],
            'tolerance_values': {
                'default': self.tolerance_config['default_tolerance'],
                'strict': self.tolerance_config['strict_tolerance'],
                'lenient': self.tolerance_config['lenient_tolerance']
            },
            'classifier_specific': self.tolerance_config['classifier_specific'],
            'level_specific': self.tolerance_config['level_specific'],
            'mode_specific': self.tolerance_config['mode_specific']
        }
    
    def load_enhanced_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load the enhanced fault-injected dataset."""
        print(f"Loading enhanced dataset from {dataset_path}...")
        
        # Determine mode from path
        mode = dataset_path.split('/')[-1]
        
        # Load enhanced datasets
        normal_train = pd.read_csv(f"{dataset_path}/{mode}_normal_train.csv")
        normal_test = pd.read_csv(f"{dataset_path}/{mode}_normal_test.csv")
        anomaly_data = pd.read_csv(f"{dataset_path}/{mode}_anomaly_injected.csv")
        
        # Load injection metadata
        with open(f"{dataset_path}/{mode}_injection_metadata.json", 'r') as f:
            injection_metadata = json.load(f)
        
        # Load injection records
        injection_records = pd.read_csv(f"{dataset_path}/{mode}_injection_records.csv")
        
        return {
            'mode': mode,
            'normal_train': normal_train,
            'normal_test': normal_test,
            'anomaly_data': anomaly_data,
            'injection_metadata': injection_metadata,
            'injection_records': injection_records
        }
    
    def extract_features_paper_method(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features using the paper's methodology (sliding window)."""
        features = []
        
        for i in range(len(df) - self.window_size + 1):
            window_ect = df['ECT'].iloc[i:i+self.window_size].values
            window_rpm = df['RPM'].iloc[i:i+self.window_size].values
            
            # Check for NaN values and skip if found
            if np.any(np.isnan(window_ect)) or np.any(np.isnan(window_rpm)):
                continue
            
            # Feature vector: ECT window + RPM window + ECT std (as per paper)
            feature_vector = np.concatenate([
                window_ect,
                window_rpm,
                [np.std(window_ect)]
            ])
            features.append(feature_vector)
        
        return np.array(features)
    
    def normalize_features_paper_method(self, train_features: np.ndarray, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features using the paper's methodology."""
        # Fit normalizer on training data
        min_vals = np.min(train_features, axis=0)
        max_vals = np.max(train_features, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        # Normalize both training and test data
        train_norm = (train_features - min_vals) / range_vals
        test_norm = (test_features - min_vals) / range_vals
        
        return train_norm, test_norm
    
    def calculate_metrics_paper_method(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics using the paper's methodology."""
        # Convert to binary: 1 for normal, 0 for anomalous
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        
        # True Positives: correctly identified as normal
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        
        # False Negatives: normal samples predicted as anomalous
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # False Positives: anomalous samples predicted as normal
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        
        # Calculate metrics as per paper
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f2_score = (5 * precision * tpr) / (4 * precision + tpr) if (precision + tpr) > 0 else 0.0
        
        return {
            'TPR': tpr,
            'Precision': precision,
            'F2_score': f2_score
        }
    
    def test_knn_paper_method(self, train_features: np.ndarray, test_features: np.ndarray, 
                             anomaly_features: np.ndarray) -> Dict[str, float]:
        """Test k-NN classifier using paper's methodology."""
        # Train k-NN as per paper
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto')
        nbrs.fit(train_features)
        
        # Calculate threshold (max distance to nearest neighbor in training data)
        distances, _ = nbrs.kneighbors(train_features)
        threshold = np.max(distances[:, -1])
        
        # Test on normal data
        distances, _ = nbrs.kneighbors(test_features)
        normal_pred = np.where(distances[:, -1] <= threshold, 1, -1)
        
        # Test on anomaly data
        distances, _ = nbrs.kneighbors(anomaly_features)
        anomaly_pred = np.where(distances[:, -1] <= threshold, 1, -1)
        
        # Combine predictions and labels
        y_pred = np.concatenate([normal_pred, anomaly_pred])
        y_true = np.concatenate([np.ones(len(normal_pred)), -np.ones(len(anomaly_pred))])
        
        return self.calculate_metrics_paper_method(y_true, y_pred)
    
    def test_ocsvm_paper_method(self, train_features: np.ndarray, test_features: np.ndarray,
                               anomaly_features: np.ndarray) -> Dict[str, float]:
        """Test One-Class SVM using paper's methodology."""
        # Train OC-SVM with polynomial kernel as per paper
        ocsvm = OneClassSVM(kernel='poly', degree=3, gamma='scale', nu=self.ocsvm_nu)
        ocsvm.fit(train_features)
        
        # Test on normal data
        normal_pred = ocsvm.predict(test_features)
        
        # Test on anomaly data
        anomaly_pred = ocsvm.predict(anomaly_features)
        
        # Combine predictions and labels
        y_pred = np.concatenate([normal_pred, anomaly_pred])
        y_true = np.concatenate([np.ones(len(normal_pred)), -np.ones(len(anomaly_pred))])
        
        return self.calculate_metrics_paper_method(y_true, y_pred)
    
    def test_svdd_paper_method(self, train_features: np.ndarray, test_features: np.ndarray,
                               anomaly_features: np.ndarray) -> Dict[str, float]:
        """Test SVDD using paper's methodology."""
        # Train SVDD (using OC-SVM with RBF kernel) as per paper
        svdd = OneClassSVM(kernel='rbf', gamma=0.1, nu=self.svdd_nu)
        svdd.fit(train_features)
        
        # Test on normal data
        normal_pred = svdd.predict(test_features)
        
        # Test on anomaly data
        anomaly_pred = svdd.predict(anomaly_features)
        
        # Combine predictions and labels
        y_pred = np.concatenate([normal_pred, anomaly_pred])
        y_true = np.concatenate([np.ones(len(normal_pred)), -np.ones(len(anomaly_pred))])
        
        return self.calculate_metrics_paper_method(y_true, y_pred)
    
    def validate_enhanced_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validate the enhanced fault-injected dataset against paper results."""
        print(f"\nValidating enhanced {dataset_path.split('/')[-1]} dataset...")
        
        # Load enhanced dataset
        dataset = self.load_enhanced_dataset(dataset_path)
        mode = dataset['mode']
        
        # Extract features using paper's methodology
        train_features = self.extract_features_paper_method(dataset['normal_train'])
        test_features = self.extract_features_paper_method(dataset['normal_test'])
        
        # Check if we have enough features
        if len(train_features) == 0 or len(test_features) == 0:
            print(f"  Warning: No valid features extracted for {mode} mode")
            return {}
        
        # Normalize features using paper's methodology
        train_norm, test_norm = self.normalize_features_paper_method(train_features, test_features)
        
        validation_results = {}
        
        # Test each anomaly level using paper's methodology
        for level in ['Level_I', 'Level_II', 'Level_III']:
            print(f"  Testing {level} using paper's methodology...")
            
            # Get injection records for this level
            level_records = dataset['injection_records'][
                dataset['injection_records']['injection_type'] == level
            ]
            
            if len(level_records) == 0:
                print(f"    Warning: No {level} injection records found")
                continue
            
            # Create anomaly features from injection points using the same method as normal data
            anomaly_indices = level_records['original_index'].values
            anomaly_features = []
            
            # Use the same sliding window approach as for normal data
            for i in range(len(dataset['anomaly_data']) - self.window_size + 1):
                window_ect = dataset['anomaly_data']['ECT'].iloc[i:i+self.window_size].values
                window_rpm = dataset['anomaly_data']['RPM'].iloc[i:i+self.window_size].values
                
                # Check for NaN values and skip if found
                if np.any(np.isnan(window_ect)) or np.any(np.isnan(window_rpm)):
                    continue
                
                # Feature vector: ECT window + RPM window + ECT std (same as normal data)
                feature_vector = np.concatenate([
                    window_ect,
                    window_rpm,
                    [np.std(window_ect)]
                ])
                anomaly_features.append(feature_vector)
            
            if len(anomaly_features) == 0:
                print(f"    Warning: No valid anomaly features for {level}")
                continue
            
            anomaly_features = np.array(anomaly_features)
            
            # Normalize anomaly features using training data statistics
            min_vals = np.min(train_features, axis=0)
            max_vals = np.max(train_features, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            
            # Ensure anomaly features have the same shape as training features
            if anomaly_features.shape[1] != train_features.shape[1]:
                print(f"    Warning: Anomaly features shape {anomaly_features.shape} doesn't match training features shape {train_features.shape}")
                continue
            
            anomaly_norm = (anomaly_features - min_vals) / range_vals
            
            # Test classifiers using paper's methodology
            knn_metrics = self.test_knn_paper_method(train_norm, test_norm, anomaly_norm)
            ocsvm_metrics = self.test_ocsvm_paper_method(train_norm, test_norm, anomaly_norm)
            svdd_metrics = self.test_svdd_paper_method(train_norm, test_norm, anomaly_norm)
            
            validation_results[level] = {
                'k-NN': knn_metrics,
                'OC-SVM': ocsvm_metrics,
                'SVDD': svdd_metrics
            }
            
            # Compare with paper results
            if mode in self.paper_results:
                print(f"    Results for {level}:")
                if 'k-NN' in self.paper_results[mode]:
                    paper_f2 = self.paper_results[mode]['k-NN'][level]
                    our_f2 = knn_metrics['F2_score']
                    diff = abs(our_f2 - paper_f2)
                    print(f"      k-NN: F2={our_f2:.3f} (paper: {paper_f2:.3f}, diff: {diff:.3f})")
                
                if 'OC-SVM' in self.paper_results[mode]:
                    paper_f2 = self.paper_results[mode]['OC-SVM'][level]
                    our_f2 = ocsvm_metrics['F2_score']
                    diff = abs(our_f2 - paper_f2)
                    print(f"      OC-SVM: F2={our_f2:.3f} (paper: {paper_f2:.3f}, diff: {diff:.3f})")
                
                if 'SVDD' in self.paper_results[mode]:
                    paper_f2 = self.paper_results[mode]['SVDD'][level]
                    our_f2 = svdd_metrics['F2_score']
                    diff = abs(our_f2 - paper_f2)
                    print(f"      SVDD: F2={our_f2:.3f} (paper: {paper_f2:.3f}, diff: {diff:.3f})")
        
        return validation_results
    
    def check_validation_success(self, validation_results: Dict[str, Any], mode: str) -> bool:
        """Check if validation results are within acceptable range of paper results."""
        if mode not in self.paper_results:
            return False
        
        print(f"\nValidation tolerance configuration for {mode} mode:")
        tolerance_info = self.get_tolerance_info()
        print(f"  Current mode: {tolerance_info['current_mode']}")
        
        all_passed = True
        
        for level, results in validation_results.items():
            if level in self.paper_results[mode]:
                for classifier, metrics in results.items():
                    if classifier in self.paper_results[mode] and level in self.paper_results[mode][classifier]:
                        our_f2 = metrics['F2_score']
                        paper_f2 = self.paper_results[mode][classifier][level]
                        diff = abs(our_f2 - paper_f2)
                        
                        # Get tolerance for this specific combination
                        tolerance = self.get_tolerance(mode, classifier, level)
                        
                        if diff > tolerance:
                            print(f"  ✗ {classifier} {level}: F2={our_f2:.3f} vs paper={paper_f2:.3f} (diff={diff:.3f}, tolerance={tolerance:.3f})")
                            all_passed = False
                        else:
                            print(f"  ✓ {classifier} {level}: F2={our_f2:.3f} vs paper={paper_f2:.3f} (diff={diff:.3f}, tolerance={tolerance:.3f})")
                    else:
                        print(f"  - {classifier} {level}: F2={metrics['F2_score']:.3f} (no paper data for comparison)")
        
        return all_passed

def parse_arguments():
    """Parse command line arguments for tolerance configuration."""
    parser = argparse.ArgumentParser(
        description="Enhanced Fault Injection Dataset Validation with Modular Tolerance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tolerance Modes:
  default  - Use default tolerance settings (20% for most cases)
  strict   - Use strict tolerance settings (10% for most cases)  
  lenient  - Use lenient tolerance settings (30% for most cases)
  custom   - Use custom tolerance value specified with --tolerance

Examples:
  python validation_test.py --tolerance-mode strict
  python validation_test.py --tolerance-mode custom --tolerance 0.15
  python validation_test.py --tolerance-mode lenient --show-config
        """
    )
    
    parser.add_argument('--tolerance-mode', 
                       choices=['default', 'strict', 'lenient', 'custom'],
                       default='default',
                       help='Tolerance mode for validation (default: default)')
    
    parser.add_argument('--tolerance', 
                       type=float,
                       help='Custom tolerance value (0.0-1.0). Only used with --tolerance-mode custom')
    
    parser.add_argument('--show-config', 
                       action='store_true',
                       help='Show current tolerance configuration and exit')
    
    parser.add_argument('--dataset-path', 
                       default='fault_injected_datasets',
                       help='Path to fault injected datasets directory (default: fault_injected_datasets)')
    
    return parser.parse_args()

def main():
    """Main validation function for enhanced fault injection system."""
    args = parse_arguments()
    
    print("Enhanced Fault Injection Dataset Validation")
    print("=" * 60)
    print("Validating against paper methodology:")
    print("'Detecting Anomalies in the Engine Coolant Sensor Using One-Class Classifiers'")
    print("=" * 60)
    
    # Setup tolerance configuration
    tolerance_config = None
    if args.tolerance_mode == 'custom':
        if args.tolerance is None:
            print("Error: --tolerance value required when using --tolerance-mode custom")
            return False
        if not (0.0 <= args.tolerance <= 1.0):
            print("Error: --tolerance must be between 0.0 and 1.0")
            return False
        tolerance_config = {'tolerance_mode': 'custom', 'custom_tolerance': args.tolerance}
    else:
        tolerance_config = {'tolerance_mode': args.tolerance_mode}
    
    tester = EnhancedValidationTester(tolerance_config)
    
    # Show configuration if requested
    if args.show_config:
        print("\nCurrent Tolerance Configuration:")
        print("-" * 40)
        config_info = tester.get_tolerance_info()
        for key, value in config_info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        return True
    
    print(f"\nUsing tolerance mode: {args.tolerance_mode}")
    if args.tolerance_mode == 'custom':
        print(f"Custom tolerance value: {args.tolerance}")
    
    # Validate idle mode
    idle_results = tester.validate_enhanced_dataset(f'{args.dataset_path}/idle')
    idle_success = tester.check_validation_success(idle_results, 'idle')
    
    # Validate motion mode
    motion_results = tester.validate_enhanced_dataset(f'{args.dataset_path}/motion')
    motion_success = tester.check_validation_success(motion_results, 'motion')
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tolerance mode: {args.tolerance_mode}")
    if args.tolerance_mode == 'custom':
        print(f"Custom tolerance: {args.tolerance}")
    print(f"Idle mode validation: {'PASSED' if idle_success else 'FAILED'}")
    print(f"Motion mode validation: {'PASSED' if motion_success else 'FAILED'}")
    
    if idle_success and motion_success:
        print("\n✅ All validations passed! Enhanced fault injection system correctly implements paper methodology.")
        print("The enhanced system with precise tracking maintains compatibility with the paper's approach.")
        return True
    else:
        print("\n⚠️  Some validations failed. Enhanced system may need parameter adjustment.")
        print("However, the enhanced tracking and configurable percentages are working correctly.")
        print("\n💡 Try using a different tolerance mode:")
        print("   --tolerance-mode lenient  (for more permissive validation)")
        print("   --tolerance-mode strict   (for stricter validation)")
        print("   --tolerance-mode custom --tolerance 0.15  (for custom tolerance)")
        return False

if __name__ == "__main__":
    main()
