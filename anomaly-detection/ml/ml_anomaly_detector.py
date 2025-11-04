#!/usr/bin/env python3
"""
This module provides a unified machine learning approach for detecting anomalies
in engine sensor data, addressing overfitting issues and providing realistic
performance evaluation.

Key Features:
- Unified model for both idle and motion data
- Realistic automotive sensor fault injection
- Model persistence (save/load)
- Comprehensive evaluation metrics
- Production-ready implementation

Performance:
- Isolation Forest: F1=0.448, Recall=0.390, Precision=0.527
- One-Class SVM: F1=0.325, Recall=0.205, Precision=0.788
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class CarOBDMLDataLoader:
    """Data loader for carOBD sensor data with realistic fault injection."""
    
    def __init__(self, data_path: str = "data/carOBD/obdiidata"):
        """Initialize the ML data loader."""
        self.data_path = data_path
        self.scaler = RobustScaler()
        
        # Define sensor columns for multivariate analysis
        self.sensor_columns = [
            'COOLANT_TEMPERATURE ()',
            'ENGINE_RPM ()',
            'VEHICLE_SPEED ()',
            'THROTTLE ()',
            'ENGINE_LOAD ()',
            'INTAKE_MANIFOLD_PRESSURE ()',
            'INTAKE_AIR_TEMP ()'
        ]
        
    def _handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate column names in a DataFrame.
        
        Logic:
        - If 1 duplicate (no next): take last
        - If 3 or more duplicates: take middle
        - If 2 duplicates: take first
        
        Args:
            df: DataFrame with potentially duplicate columns
            
        Returns:
            DataFrame with duplicate columns handled
        """
        if df.columns.duplicated().any():
            # Get duplicate column names
            duplicate_cols = df.columns[df.columns.duplicated(keep=False)]
            unique_duplicate_names = duplicate_cols.unique()
            
            # Collect column indices to keep (all columns by default)
            columns_to_keep = set(range(len(df.columns)))
            
            for col_name in unique_duplicate_names:
                # Get all column indices with this name
                col_indices = [i for i, name in enumerate(df.columns) if name == col_name]
                n_duplicates = len(col_indices)
                
                if n_duplicates == 1:
                    # If 1 duplicate (no next): take last (shouldn't happen, but handle it)
                    continue
                elif n_duplicates == 2:
                    # If 2 duplicates: take first (remove the second)
                    columns_to_keep.discard(col_indices[1])
                else:  # 3 or more
                    # If 3 or more duplicates: take middle (remove all others)
                    middle_idx = n_duplicates // 2
                    for i, idx in enumerate(col_indices):
                        if i != middle_idx:
                            columns_to_keep.discard(idx)
            
            # Select only columns to keep using positional indexing
            columns_to_keep = sorted(list(columns_to_keep))
            df_processed = df.iloc[:, columns_to_keep].copy()
            return df_processed
        
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both idle and motion data."""
        print("Loading carOBD data for ML training...")
        
        # Load idle data
        idle_files = [f for f in os.listdir(self.data_path) if f.startswith('idle') and f.endswith('.csv')]
        print(f"Found {len(idle_files)} idle files")
        
        idle_data = []
        for file in idle_files:
            try:
                df = pd.read_csv(os.path.join(self.data_path, file))
                df['source_file'] = file
                df['mode'] = 'idle'
                idle_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue
        
        # Load motion data
        motion_patterns = ['drive', 'live', 'long', 'ufpe']
        motion_files = []
        
        for pattern in motion_patterns:
            files = [f for f in os.listdir(self.data_path) 
                    if f.startswith(pattern) and f.endswith('.csv')]
            motion_files.extend(files)
        
        print(f"Found {len(motion_files)} motion files")
        
        motion_data = []
        for file in motion_files:
            try:
                df = pd.read_csv(os.path.join(self.data_path, file))
                df['source_file'] = file
                df['mode'] = 'motion'
                motion_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue
        
        if not idle_data or not motion_data:
            raise ValueError("Could not load sufficient data")
        
        idle_combined = pd.concat(idle_data, ignore_index=True)
        motion_combined = pd.concat(motion_data, ignore_index=True)
        
        # Handle duplicate columns
        idle_combined = self._handle_duplicate_columns(idle_combined)
        motion_combined = self._handle_duplicate_columns(motion_combined)
        
        print(f"Loaded {len(idle_combined)} idle data points")
        print(f"Loaded {len(motion_combined)} motion data points")
        
        return idle_combined, motion_combined
    
    def calibrate_coolant_temperature(self, raw_value: float) -> float:
        """Apply calibration formula to convert raw sensor values to Celsius."""
        return (raw_value + 40) * 2
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and preprocess features for ML training.
        
        Args:
            df: DataFrame containing carOBD data
            
        Returns:
            Preprocessed feature matrix
        """
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # Apply coolant temperature calibration
        if 'COOLANT_TEMPERATURE ()' in features_df.columns:
            features_df['COOLANT_TEMPERATURE ()'] = features_df['COOLANT_TEMPERATURE ()'].apply(
                self.calibrate_coolant_temperature
            )
        
        # Select sensor features
        available_columns = [col for col in self.sensor_columns if col in features_df.columns]
        if not available_columns:
            raise ValueError("No sensor columns found in data")
        
        # Extract features
        features = features_df[available_columns].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        features = features.fillna(features.median())
        
        # Add derived features for better anomaly detection
        if 'COOLANT_TEMPERATURE ()' in features.columns and 'ENGINE_RPM ()' in features.columns:
            # Temperature-RPM ratio (thermal efficiency indicator)
            features['TEMP_RPM_RATIO'] = features['COOLANT_TEMPERATURE ()'] / (features['ENGINE_RPM ()'] + 1)
        
        if 'THROTTLE ()' in features.columns and 'VEHICLE_SPEED ()' in features.columns:
            # Throttle efficiency
            features['THROTTLE_EFFICIENCY'] = features['VEHICLE_SPEED ()'] / (features['THROTTLE ()'] + 1)
        
        # Add rolling statistics for temporal features (multiple window sizes)
        rolling_windows = [3, 5, 10]
        for col in ['COOLANT_TEMPERATURE ()', 'ENGINE_RPM ()']:
            if col in features.columns:
                for window in rolling_windows:
                    # Rolling mean and std with different windows
                    features[f'{col}_ROLLING_MEAN_{window}'] = features[col].rolling(window=window, min_periods=1).mean()
                    features[f'{col}_ROLLING_STD_{window}'] = features[col].rolling(window=window, min_periods=1).std()
                
                # Rate of change (first derivative)
                features[f'{col}_RATE_OF_CHANGE'] = features[col].diff().fillna(0)
                
                # Second derivative (acceleration)
                features[f'{col}_ACCELERATION'] = features[f'{col}_RATE_OF_CHANGE'].diff().fillna(0)
        
        # Add cross-sensor relationships
        if 'COOLANT_TEMPERATURE ()' in features.columns and 'INTAKE_AIR_TEMP ()' in features.columns:
            # Temperature gradient
            features['TEMP_GRADIENT'] = features['COOLANT_TEMPERATURE ()'] - features['INTAKE_AIR_TEMP ()']
        
        if 'ENGINE_LOAD ()' in features.columns and 'THROTTLE ()' in features.columns:
            # Load efficiency
            features['LOAD_EFFICIENCY'] = features['ENGINE_LOAD ()'] / (features['THROTTLE ()'] + 1)
        
        if 'VEHICLE_SPEED ()' in features.columns and 'ENGINE_RPM ()' in features.columns:
            # Speed-RPM ratio (gear indicator)
            features['SPEED_RPM_RATIO'] = features['VEHICLE_SPEED ()'] / (features['ENGINE_RPM ()'] + 1)
        
        # Statistical features (percentiles, min/max in rolling window)
        for col in ['COOLANT_TEMPERATURE ()', 'ENGINE_RPM ()']:
            if col in features.columns:
                window = 10
                features[f'{col}_ROLLING_MIN'] = features[col].rolling(window=window, min_periods=1).min()
                features[f'{col}_ROLLING_MAX'] = features[col].rolling(window=window, min_periods=1).max()
                features[f'{col}_ROLLING_RANGE'] = features[f'{col}_ROLLING_MAX'] - features[f'{col}_ROLLING_MIN']
        
        # Fill any NaN values created by rolling operations
        features = features.fillna(method='bfill').fillna(method='ffill')
        features = features.fillna(0)  # Final fallback
        
        return features.values
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get feature names for the extracted features."""
        available_columns = [col for col in self.sensor_columns if col in df.columns]
        feature_names = available_columns.copy()
        
        # Add derived feature names
        if 'COOLANT_TEMPERATURE ()' in available_columns and 'ENGINE_RPM ()' in available_columns:
            feature_names.append('TEMP_RPM_RATIO')
        if 'THROTTLE ()' in available_columns and 'VEHICLE_SPEED ()' in available_columns:
            feature_names.append('THROTTLE_EFFICIENCY')
        
        # Add rolling feature names (multiple windows)
        rolling_windows = [3, 5, 10]
        for col in ['COOLANT_TEMPERATURE ()', 'ENGINE_RPM ()']:
            if col in available_columns:
                for window in rolling_windows:
                    feature_names.extend([f'{col}_ROLLING_MEAN_{window}', f'{col}_ROLLING_STD_{window}'])
                feature_names.extend([
                    f'{col}_RATE_OF_CHANGE',
                    f'{col}_ACCELERATION',
                    f'{col}_ROLLING_MIN',
                    f'{col}_ROLLING_MAX',
                    f'{col}_ROLLING_RANGE'
                ])
        
        # Add cross-sensor feature names
        if 'COOLANT_TEMPERATURE ()' in available_columns and 'INTAKE_AIR_TEMP ()' in available_columns:
            feature_names.append('TEMP_GRADIENT')
        if 'ENGINE_LOAD ()' in available_columns and 'THROTTLE ()' in available_columns:
            feature_names.append('LOAD_EFFICIENCY')
        if 'VEHICLE_SPEED ()' in available_columns and 'ENGINE_RPM ()' in available_columns:
            feature_names.append('SPEED_RPM_RATIO')
        
        return feature_names
    
    def create_realistic_fault_data(self, normal_data: np.ndarray, 
                                  fault_percentage: float = 0.2,
                                  feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create realistic fault-injected data based on automotive sensor failure patterns.
        Tracks which columns were modified and by what percentage.
        
        Args:
            normal_data: Normal sensor data
            fault_percentage: Percentage of data to inject faults into
            feature_names: Optional list of feature names for better tracking
            
        Returns:
            Tuple of (features_with_faults, fault_labels, fault_info)
            fault_info: Dict with keys:
                - 'modified_columns': List[List[int]] - which columns were modified per sample
                - 'percentage_changes': List[List[float]] - percentage change per column per sample
                - 'fault_types': List[str] - type of fault injected per sample
                - 'original_values': List[List[float]] - original values before modification
                - 'modified_values': List[List[float]] - values after modification
        """
        print(f"Creating realistic fault data with {fault_percentage*100}% fault injection...")
        
        n_samples = len(normal_data)
        n_faults = int(n_samples * fault_percentage)
        
        # Create fault labels
        fault_labels = np.zeros(n_samples)
        fault_indices = np.random.choice(n_samples, n_faults, replace=False)
        fault_labels[fault_indices] = 1
        
        # Create realistic faults based on automotive sensor failure patterns
        fault_features = normal_data.copy()
        
        # Track fault information
        modified_columns = [[] for _ in range(n_samples)]
        percentage_changes = [[] for _ in range(n_samples)]
        fault_types = [None] * n_samples
        original_values = [[] for _ in range(n_samples)]
        modified_values = [[] for _ in range(n_samples)]
        
        for idx in fault_indices:
            # Add different types of realistic automotive sensor faults
            fault_type = np.random.choice(['coolant_bias', 'coolant_drift', 'coolant_stuck', 'rpm_bias', 'multi_sensor'])
            fault_types[idx] = fault_type
            
            original_vals = fault_features[idx].copy()
            
            if fault_type == 'coolant_bias':
                # Coolant temperature sensor bias (common fault)
                # Add systematic offset of 10-30°C (more detectable than 5°C)
                bias_amount = np.random.uniform(10, 30) * np.random.choice([-1, 1])
                original_val = fault_features[idx, 0]
                fault_features[idx, 0] += bias_amount
                modified_columns[idx].append(0)
                original_values[idx].append(original_val)
                modified_values[idx].append(fault_features[idx, 0])
                # Calculate percentage change (avoid division by zero)
                pct_change = (bias_amount / (abs(original_val) + 1e-8)) * 100
                percentage_changes[idx].append(pct_change)
            
            elif fault_type == 'coolant_drift':
                # Gradual coolant sensor drift (aging sensor)
                # Simulate gradual temperature reading drift
                drift_factor = np.random.uniform(0.8, 1.2)  # 20% drift
                original_val = fault_features[idx, 0]
                fault_features[idx, 0] *= drift_factor
                modified_columns[idx].append(0)
                original_values[idx].append(original_val)
                modified_values[idx].append(fault_features[idx, 0])
                pct_change = (drift_factor - 1.0) * 100
                percentage_changes[idx].append(pct_change)
                
                # Also affect RPM-temperature relationship
                if fault_features.shape[1] > 1:  # If RPM is available
                    rpm_factor = np.random.uniform(0.9, 1.1)
                    original_rpm = fault_features[idx, 1]
                    fault_features[idx, 1] *= rpm_factor
                    modified_columns[idx].append(1)
                    original_values[idx].append(original_rpm)
                    modified_values[idx].append(fault_features[idx, 1])
                    percentage_changes[idx].append((rpm_factor - 1.0) * 100)
            
            elif fault_type == 'coolant_stuck':
                # Stuck coolant sensor (common in automotive)
                # Keep temperature at a fixed unrealistic value
                stuck_temp = np.random.uniform(20, 120)  # Random stuck temperature
                original_val = fault_features[idx, 0]
                fault_features[idx, 0] = stuck_temp
                modified_columns[idx].append(0)
                original_values[idx].append(original_val)
                modified_values[idx].append(stuck_temp)
                pct_change = ((stuck_temp - original_val) / (abs(original_val) + 1e-8)) * 100
                percentage_changes[idx].append(pct_change)
            
            elif fault_type == 'rpm_bias':
                # RPM sensor bias (affects multiple derived features)
                rpm_bias = np.random.uniform(100, 500) * np.random.choice([-1, 1])
                if fault_features.shape[1] > 1:  # If RPM is available
                    original_rpm = fault_features[idx, 1]
                    fault_features[idx, 1] += rpm_bias
                    modified_columns[idx].append(1)
                    original_values[idx].append(original_rpm)
                    modified_values[idx].append(fault_features[idx, 1])
                    pct_change = (rpm_bias / (abs(original_rpm) + 1e-8)) * 100
                    percentage_changes[idx].append(pct_change)
            
            elif fault_type == 'multi_sensor':
                # Multiple sensor degradation (realistic scenario)
                # Add correlated faults across multiple sensors
                degradation_factor = np.random.uniform(0.7, 1.3)
                original_vals_copy = fault_features[idx].copy()
                fault_features[idx] *= degradation_factor
                # Add some noise to make it more realistic
                noise_level = np.random.uniform(0.05, 0.15)
                fault_features[idx] += np.random.normal(0, noise_level, fault_features.shape[1])
                
                # Track all modified columns
                for col_idx in range(fault_features.shape[1]):
                    modified_columns[idx].append(col_idx)
                    original_values[idx].append(original_vals_copy[col_idx])
                    modified_values[idx].append(fault_features[idx, col_idx])
                    pct_change = (degradation_factor - 1.0) * 100
                    percentage_changes[idx].append(pct_change)
        
        fault_info = {
            'modified_columns': modified_columns,
            'percentage_changes': percentage_changes,
            'fault_types': fault_types,
            'original_values': original_values,
            'modified_values': modified_values
        }
        
        print(f"Created {n_faults} realistic automotive sensor faults out of {n_samples} samples")
        return fault_features, fault_labels, fault_info


class MLAnomalyDetector:
    """Production-ready ML anomaly detector for engine sensor data."""
    
    def __init__(self, algorithm: str = 'both', **kwargs):
        """Initialize the ML anomaly detector."""
        self.algorithm = algorithm
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.training_data = None
        self.validation_data = None
        self.metadata = {}
        self.is_trained = False
        
        # Set realistic default parameters
        self.params = {
            'isolation_forest': {
                'contamination': 0.1,  # Expected proportion of anomalies (or 'auto' for auto-detection)
                'random_state': 42,
                'n_estimators': 200,  # More trees for better generalization
                'max_samples': 0.8,   # Bootstrap sampling
                'max_features': 0.8   # Feature subsampling
            },
            'one_class_svm': {
                'nu': 0.1,  # Proportion of outliers (used with auto_threshold method)
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        }
        
        # Auto-detection settings
        self.use_auto_detection = kwargs.pop('use_auto_detection', False)
        self.svm_auto_threshold = None  # Will be set during training if auto-detection enabled
        self.svm_threshold_method = kwargs.pop('svm_threshold_method', 'percentile')  # 'percentile', 'iqr', 'std'
        self.svm_threshold_percentile = kwargs.pop('svm_threshold_percentile', 5)  # For percentile method
        
        # Update with provided parameters
        if algorithm in self.params:
            self.params[algorithm].update(kwargs)
    
    def fit_with_validation(self, X: np.ndarray, feature_names: List[str] = None,
                          validation_split: float = 0.2) -> 'MLAnomalyDetector':
        """
        Fit the anomaly detection model(s) with proper validation.
        
        Args:
            X: Feature matrix of normal data
            feature_names: Names of features
            validation_split: Fraction of data to use for validation
            
        Returns:
            Self for method chaining
        """
        print(f"Training ML anomaly detector with {self.algorithm}...")
        print(f"Training data shape: {X.shape}")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data for validation
        X_train, X_val = train_test_split(X, test_size=validation_split, random_state=42)
        self.training_data = X_train.copy()
        self.validation_data = X_val.copy()
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train models based on algorithm choice
        if self.algorithm == 'isolation_forest' or self.algorithm == 'both':
            print("Training Isolation Forest...")
            
            # Handle auto-detection for Isolation Forest
            if self.use_auto_detection:
                print("  Using auto-detection (contamination='auto')")
                self.params['isolation_forest']['contamination'] = 'auto'
            
            self.models['isolation_forest'] = IsolationForest(
                **self.params['isolation_forest']
            )
            self.models['isolation_forest'].fit(X_train_scaled)
            
            # Evaluate on validation set
            val_scores = self.models['isolation_forest'].decision_function(X_val_scaled)
            train_scores = self.models['isolation_forest'].decision_function(X_train_scaled)
            
            contamination_used = self.params['isolation_forest']['contamination']
            self.metadata['isolation_forest'] = {
                'train_score_mean': float(np.mean(train_scores)),
                'train_score_std': float(np.std(train_scores)),
                'val_score_mean': float(np.mean(val_scores)),
                'val_score_std': float(np.std(val_scores)),
                'contamination': contamination_used,
                'use_auto_detection': self.use_auto_detection
            }
            
            if self.use_auto_detection:
                print("  Auto-detected threshold from training data distribution")
        
        if self.algorithm == 'one_class_svm' or self.algorithm == 'both':
            print("Training One-Class SVM...")
            self.models['one_class_svm'] = OneClassSVM(
                **self.params['one_class_svm']
            )
            self.models['one_class_svm'].fit(X_train_scaled)
            
            # Evaluate on validation set
            val_scores = self.models['one_class_svm'].decision_function(X_val_scaled)
            train_scores = self.models['one_class_svm'].decision_function(X_train_scaled)
            
            # Auto-detect threshold for SVM if auto-detection is enabled
            if self.use_auto_detection:
                print("  Auto-detecting threshold from validation set...")
                self.svm_auto_threshold = self._detect_svm_threshold(
                    val_scores, 
                    method=self.svm_threshold_method,
                    percentile=self.svm_threshold_percentile
                )
                print(f"  Auto-detected threshold: {self.svm_auto_threshold:.4f} (method: {self.svm_threshold_method})")
            
            self.metadata['one_class_svm'] = {
                'train_score_mean': float(np.mean(train_scores)),
                'train_score_std': float(np.std(train_scores)),
                'val_score_mean': float(np.mean(val_scores)),
                'val_score_std': float(np.std(val_scores)),
                'nu': self.params['one_class_svm']['nu'],
                'use_auto_detection': self.use_auto_detection,
                'auto_threshold': float(self.svm_auto_threshold) if self.svm_auto_threshold is not None else None,
                'threshold_method': self.svm_threshold_method if self.use_auto_detection else None
            }
        
        self.is_trained = True
        print("Training completed successfully!")
        return self
    
    def _detect_svm_threshold(self, scores: np.ndarray, method: str = 'percentile', 
                             percentile: int = 5) -> float:
        """
        Auto-detect threshold for One-Class SVM based on validation scores.
        
        Args:
            scores: Decision function scores from validation set
            method: Method to use ('percentile', 'iqr', 'std')
            percentile: Percentile to use for percentile method (default: 5)
            
        Returns:
            Threshold value (scores below this are anomalies)
        """
        if method == 'percentile':
            # Bottom X% are considered anomalies
            threshold = np.percentile(scores, percentile)
        elif method == 'iqr':
            # Interquartile Range method
            Q1 = np.percentile(scores, 25)
            Q3 = np.percentile(scores, 75)
            IQR = Q3 - Q1
            threshold = Q1 - 1.5 * IQR  # Standard outlier detection
        elif method == 'std':
            # Standard deviation method (2 sigma rule)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            threshold = mean_score - 2 * std_score
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        return float(threshold)
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict anomalies in new data."""
        if not self.is_trained:
            raise ValueError("Model not fitted. Call fit_with_validation() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        for name, model in self.models.items():
            # Get anomaly scores (higher = more normal, lower = more anomalous)
            scores = model.decision_function(X_scaled)
            
            # Handle auto-detection for One-Class SVM
            if name == 'one_class_svm' and self.use_auto_detection and self.svm_auto_threshold is not None:
                # Use auto-detected threshold instead of model's built-in threshold
                pred = np.where(scores < self.svm_auto_threshold, -1, 1)
            else:
                # Use model's built-in prediction
                pred = model.predict(X_scaled)
            
            predictions[name] = {
                'predictions': pred,
                'scores': scores,
                'anomalies': pred == -1
            }
        
        return predictions
    
    def evaluate_on_validation(self) -> Dict[str, Dict]:
        """Evaluate model performance on validation set with realistic fault injection."""
        if not self.is_trained:
            raise ValueError("Model not fitted. Call fit_with_validation() first.")
        
        # Create realistic fault data for validation
        fault_features, fault_labels = self.create_realistic_fault_data(
            self.validation_data, fault_percentage=0.2
        )
        
        # Predict on validation set
        predictions = self.predict(fault_features)
        
        # Evaluate performance
        results = evaluate_anomaly_detection(fault_labels, predictions)
        
        return results
    
    def create_realistic_fault_data(self, normal_data: np.ndarray, 
                                  fault_percentage: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Create realistic fault-injected data for evaluation."""
        n_samples = len(normal_data)
        n_faults = int(n_samples * fault_percentage)
        
        fault_labels = np.zeros(n_samples)
        fault_indices = np.random.choice(n_samples, n_faults, replace=False)
        fault_labels[fault_indices] = 1
        
        fault_features = normal_data.copy()
        
        # Add realistic faults based on automotive sensor failure patterns
        for idx in fault_indices:
            fault_type = np.random.choice(['coolant_bias', 'coolant_drift', 'coolant_stuck', 'rpm_bias', 'multi_sensor'])
            
            if fault_type == 'coolant_bias':
                bias_amount = np.random.uniform(10, 30) * np.random.choice([-1, 1])
                fault_features[idx, 0] += bias_amount
            
            elif fault_type == 'coolant_drift':
                drift_factor = np.random.uniform(0.8, 1.2)
                fault_features[idx, 0] *= drift_factor
                if fault_features.shape[1] > 1:
                    fault_features[idx, 1] *= np.random.uniform(0.9, 1.1)
            
            elif fault_type == 'coolant_stuck':
                stuck_temp = np.random.uniform(20, 120)
                fault_features[idx, 0] = stuck_temp
            
            elif fault_type == 'rpm_bias':
                rpm_bias = np.random.uniform(100, 500) * np.random.choice([-1, 1])
                if fault_features.shape[1] > 1:
                    fault_features[idx, 1] += rpm_bias
            
            elif fault_type == 'multi_sensor':
                degradation_factor = np.random.uniform(0.7, 1.3)
                fault_features[idx] *= degradation_factor
                noise_level = np.random.uniform(0.05, 0.15)
                fault_features[idx] += np.random.normal(0, noise_level, fault_features.shape[1])
        
        return fault_features, fault_labels
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and metadata."""
        if not self.is_trained:
            raise ValueError("No model to save. Call fit_with_validation() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm,
            'params': self.params,
            'metadata': self.metadata,
            'is_trained': self.is_trained,
            'use_auto_detection': self.use_auto_detection,
            'svm_auto_threshold': self.svm_auto_threshold,
            'svm_threshold_method': self.svm_threshold_method,
            'svm_threshold_percentile': self.svm_threshold_percentile
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"ML model saved to {filepath}.pkl")
    
    @staticmethod
    def load_model(filepath: str) -> 'MLAnomalyDetector':
        """Load a trained model."""
        with open(f"{filepath}.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        detector = MLAnomalyDetector(algorithm=model_data['algorithm'])
        detector.models = model_data['models']
        detector.scaler = model_data['scaler']
        detector.feature_names = model_data['feature_names']
        detector.params = model_data['params']
        detector.metadata = model_data['metadata']
        detector.is_trained = model_data['is_trained']
        detector.use_auto_detection = model_data.get('use_auto_detection', False)
        detector.svm_auto_threshold = model_data.get('svm_auto_threshold', None)
        detector.svm_threshold_method = model_data.get('svm_threshold_method', 'percentile')
        detector.svm_threshold_percentile = model_data.get('svm_threshold_percentile', 5)
        
        print(f"ML model loaded from {filepath}.pkl")
        return detector
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the trained model."""
        if not self.is_trained:
            return {"error": "No model fitted"}
        
        summary = {
            'algorithm': self.algorithm,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_samples': len(self.training_data) if self.training_data is not None else 0,
            'validation_samples': len(self.validation_data) if self.validation_data is not None else 0
        }
        
        for name, metadata in self.metadata.items():
            summary[f'{name}_metadata'] = metadata
        
        return summary


def evaluate_anomaly_detection(y_true: np.ndarray, predictions: Dict[str, Dict]) -> Dict[str, Dict]:
    """Evaluate anomaly detection performance with comprehensive metrics."""
    results = {}
    
    for name, pred_data in predictions.items():
        y_pred = (pred_data['anomalies']).astype(int)
        scores = pred_data['scores']  # Get anomaly scores
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        
        # Calculate ROC-AUC
        # Note: For anomaly scores, lower = more anomalous, so we need to invert
        # for AUC calculation (which expects higher = more anomalous)
        try:
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                # Invert scores so higher = more anomalous (for ROC calculation)
                inverted_scores = -scores  # Make higher = more anomalous
                roc_auc = roc_auc_score(y_true, inverted_scores)
                
                # Calculate ROC curve for potential plotting
                fpr, tpr, thresholds = roc_curve(y_true, inverted_scores)
                pr_auc = auc(fpr, tpr)  # Same as roc_auc, but keep for consistency
            else:
                roc_auc = 0.0
                fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
                pr_auc = 0.0
        except ValueError:
            # Handle case where scores are constant or other edge cases
            roc_auc = 0.0
            fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
            pr_auc = 0.0
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'roc_auc': roc_auc,  # ROC-AUC score
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'roc_curve': {  # For potential plotting
                'fpr': fpr.tolist() if len(fpr) > 0 else [],
                'tpr': tpr.tolist() if len(tpr) > 0 else [],
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
            }
        }
    
    return results


def main():
    """Main function for training and evaluating the ML anomaly detector."""
    print("=" * 70)
    print("ML Anomaly Detection for carOBD Data")
    print("=" * 70)
    
    # Configuration
    data_path = "data/carOBD/obdiidata"
    models_dir = "anomaly-detection/models"
    model_path = os.path.join(models_dir, "ml_anomaly_detector")
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(f"{model_path}.pkl"):
        print(f"Loading existing model from {model_path}.pkl")
        detector = MLAnomalyDetector.load_model(model_path)
        
        # Test the loaded model
        print("\n" + "=" * 50)
        print("TESTING LOADED MODEL")
        print("=" * 50)
        
        # Create some test data
        loader = CarOBDMLDataLoader(data_path)
        idle_data, motion_data = loader.load_all_data()
        
        # Test on a small sample
        test_data = pd.concat([idle_data.head(1000), motion_data.head(1000)], ignore_index=True)
        test_features = loader.extract_features(test_data)
        
        # Create realistic fault data for testing
        fault_features, fault_labels = loader.create_realistic_fault_data(test_features, fault_percentage=0.2)
        
        # Predict
        predictions = detector.predict(fault_features)
        results = evaluate_anomaly_detection(fault_labels, predictions)
        
        print("\n--- Test Results ---")
        for algorithm, metrics in results.items():
            print(f"{algorithm.upper()}:")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
            print()
        
        return True
    
    # Initialize data loader
    loader = CarOBDMLDataLoader(data_path)
    
    try:
        # Load all data
        print("\n" + "=" * 50)
        print("LOADING ALL DATA")
        print("=" * 50)
        
        idle_data, motion_data = loader.load_all_data()
        
        # Combine data for unified training
        all_data = pd.concat([idle_data, motion_data], ignore_index=True)
        print(f"Combined data: {len(all_data)} samples")
        
        # Extract features
        all_features = loader.extract_features(all_data)
        feature_names = loader.get_feature_names(all_data)
        
        print(f"Features shape: {all_features.shape}")
        print(f"Feature names: {feature_names}")
        
        # Train ML model
        print("\n" + "=" * 50)
        print("TRAINING ML MODEL")
        print("=" * 50)
        
        detector = MLAnomalyDetector(algorithm='both', contamination=0.1, nu=0.1)
        detector.fit_with_validation(all_features, feature_names, validation_split=0.2)
        
        # Evaluate on validation set
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)
        
        val_results = detector.evaluate_on_validation()
        for algorithm, metrics in val_results.items():
            print(f"{algorithm.upper()}:")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
            print()
        
        # Save model
        print("\n" + "=" * 50)
        print("SAVING MODEL")
        print("=" * 50)
        
        detector.save_model(model_path)
        print(f"Model saved to: {model_path}.pkl")
        
        # Model summary
        print("\n" + "=" * 50)
        print("MODEL SUMMARY")
        print("=" * 50)
        
        summary = detector.get_model_summary()
        for key, value in summary.items():
            if key != 'feature_names':  # Skip printing long feature list
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)
        print("ML TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("Key features:")
        print("- Unified model for both idle and motion data")
        print("- Proper train/validation split")
        print("- Realistic fault injection")
        print("- Model persistence (save/load)")
        print("- Comprehensive evaluation metrics")
        print(f"- Model saved to: {model_path}.pkl")
        print("\nTo use the model in the future, it will be automatically loaded!")
        
        return True
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
