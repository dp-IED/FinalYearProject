"""
Enhanced Fault Injection System with Precise Anomaly Tracking
Creates fault-injected datasets with configurable percentages and precise datapoint tracking.
"""

import numpy as np
import pandas as pd
import os
import glob
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
from dataclasses import dataclass
from datetime import datetime
warnings.filterwarnings('ignore')

@dataclass
class AnomalyInjectionConfig:
    """Configuration for anomaly injection."""
    mode: str  # 'idle' or 'motion'
    anomaly_percentage: float  # Percentage of data to inject anomalies (0.0 to 1.0)
    snr_levels: Dict[str, int]  # SNR levels for different anomaly types
    injection_strategy: str = 'random'  # 'random', 'sequential', 'clustered'
    seed: int = 42
    min_anomaly_samples: int = 10  # Minimum number of anomaly samples
    max_anomaly_samples: int = 1000  # Maximum number of anomaly samples

@dataclass
class AnomalyInjectionRecord:
    """Record of a single anomaly injection."""
    original_index: int  # Original index in the dataset
    original_value: float  # Original ECT value
    injected_value: float  # ECT value after injection
    snr_db: float  # SNR level used
    injection_type: str  # Type of injection (Level_I, Level_II, Level_III)
    timestamp: str  # When the injection was made
    noise_added: float  # Amount of noise added

class EnhancedFlickerNoiseGenerator:
    """Enhanced flicker noise generator with better control."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_flicker_noise(self, length: int, sample_rate: float = 1.0) -> np.ndarray:
        """Generate flicker noise with 1/f power spectral density."""
        # Generate white noise
        white_noise = np.random.normal(0, 1, length)
        
        # Apply 1/f filter in frequency domain
        freqs = np.fft.fftfreq(length, 1/sample_rate)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # 1/f power spectral density
        f_filter = 1 / np.sqrt(np.abs(freqs))
        f_filter[0] = 0  # DC component
        
        # Apply filter
        noise_fft = np.fft.fft(white_noise)
        filtered_fft = noise_fft * f_filter
        
        # Convert back to time domain
        flicker_noise = np.real(np.fft.ifft(filtered_fft))
        
        return flicker_noise

class EnhancedFaultInjector:
    """Enhanced fault injector with precise tracking and configurable percentages."""
    
    def __init__(self, seed: int = 42):
        self.noise_generator = EnhancedFlickerNoiseGenerator(seed)
        self.seed = seed
        self.injection_records: List[AnomalyInjectionRecord] = []
    
    def calculate_signal_power(self, signal: np.ndarray) -> float:
        """Calculate the power of a signal."""
        return np.mean(signal ** 2)
    
    def inject_flicker_noise_with_tracking(self, signal: np.ndarray, snr_db: float, 
                                         injection_type: str) -> Tuple[np.ndarray, AnomalyInjectionRecord]:
        """
        Inject flicker noise with precise tracking.
        
        Args:
            signal: Original signal
            snr_db: Target SNR in dB
            injection_type: Type of injection (Level_I, Level_II, Level_III)
            
        Returns:
            Tuple of (noisy_signal, injection_record)
        """
        # Calculate signal power
        signal_power = self.calculate_signal_power(signal)
        
        # Calculate required noise power for target SNR
        required_noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate flicker noise
        flicker_noise = self.noise_generator.generate_flicker_noise(len(signal))
        
        # Scale noise to achieve target SNR
        current_noise_power = self.calculate_signal_power(flicker_noise)
        if current_noise_power > 0:
            scale_factor = np.sqrt(required_noise_power / current_noise_power)
            scaled_noise = flicker_noise * scale_factor
        else:
            scaled_noise = flicker_noise
        
        # Add noise to signal
        noisy_signal = signal + scaled_noise
        
        # Create injection record
        injection_record = AnomalyInjectionRecord(
            original_index=0,  # Will be set by caller
            original_value=float(signal[0]),
            injected_value=float(noisy_signal[0]),
            snr_db=snr_db,
            injection_type=injection_type,
            timestamp=datetime.now().isoformat(),
            noise_added=float(scaled_noise[0])
        )
        
        return noisy_signal, injection_record
    
    def select_injection_points(self, data_length: int, config: AnomalyInjectionConfig) -> List[int]:
        """Select points for anomaly injection based on strategy."""
        # Calculate number of anomalies to inject
        num_anomalies = int(data_length * config.anomaly_percentage)
        num_anomalies = max(config.min_anomaly_samples, 
                           min(num_anomalies, config.max_anomaly_samples))
        
        np.random.seed(config.seed)
        
        if config.injection_strategy == 'random':
            # Random selection
            indices = np.random.choice(data_length, num_anomalies, replace=False)
        elif config.injection_strategy == 'sequential':
            # Sequential selection (evenly spaced)
            step = data_length // num_anomalies
            indices = np.arange(0, data_length, step)[:num_anomalies]
        elif config.injection_strategy == 'clustered':
            # Clustered selection (groups of anomalies)
            cluster_size = max(1, num_anomalies // 5)  # 5 clusters
            cluster_starts = np.random.choice(data_length - cluster_size, 
                                            num_anomalies // cluster_size, replace=False)
            indices = []
            for start in cluster_starts:
                cluster_indices = np.arange(start, start + cluster_size)
                indices.extend(cluster_indices)
            indices = np.array(indices)[:num_anomalies]
        else:
            raise ValueError(f"Unknown injection strategy: {config.injection_strategy}")
        
        return sorted(indices.tolist())
    
    def inject_anomalies_with_tracking(self, ect_data: np.ndarray, rpm_data: np.ndarray,
                                     config: AnomalyInjectionConfig) -> Tuple[np.ndarray, np.ndarray, List[AnomalyInjectionRecord]]:
        """
        Inject anomalies with precise tracking.
        
        Args:
            ect_data: ECT sensor data
            rpm_data: RPM data
            config: Injection configuration
            
        Returns:
            Tuple of (modified_ect, modified_rpm, injection_records)
        """
        print(f"Injecting anomalies for {config.mode} mode...")
        print(f"  Strategy: {config.injection_strategy}")
        print(f"  Percentage: {config.anomaly_percentage:.2%}")
        
        # Create copies to avoid modifying original data
        modified_ect = ect_data.copy()
        modified_rpm = rpm_data.copy()
        
        # Select injection points
        injection_indices = self.select_injection_points(len(ect_data), config)
        print(f"  Selected {len(injection_indices)} injection points")
        
        # Distribute anomalies across SNR levels
        num_levels = len(config.snr_levels)
        level_names = list(config.snr_levels.keys())
        level_snrs = list(config.snr_levels.values())
        
        injection_records = []
        
        for i, idx in enumerate(injection_indices):
            # Determine which level to use
            level_idx = i % num_levels
            level_name = level_names[level_idx]
            snr_db = level_snrs[level_idx]
            
            # Inject anomaly at this point
            original_value = ect_data[idx]
            sample_array = np.array([original_value])
            
            noisy_sample, injection_record = self.inject_flicker_noise_with_tracking(
                sample_array, snr_db, level_name
            )
            
            # Update the data
            modified_ect[idx] = noisy_sample[0]
            
            # Update injection record with actual index
            injection_record.original_index = idx
            injection_record.original_value = original_value
            injection_record.injected_value = noisy_sample[0]
            injection_record.noise_added = noisy_sample[0] - original_value
            
            injection_records.append(injection_record)
        
        print(f"  Injected {len(injection_records)} anomalies")
        print(f"  Level distribution: {self._get_level_distribution(injection_records)}")
        
        return modified_ect, modified_rpm, injection_records
    
    def _get_level_distribution(self, records: List[AnomalyInjectionRecord]) -> Dict[str, int]:
        """Get distribution of injection levels."""
        distribution = {}
        for record in records:
            level = record.injection_type
            distribution[level] = distribution.get(level, 0) + 1
        return distribution

class EnhancedCarOBDDataProcessor:
    """Enhanced processor with precise anomaly tracking and configurable percentages."""
    
    def __init__(self, data_path: str = "../data/carOBD/obdiidata"):
        self.data_path = data_path
        self.fault_injector = EnhancedFaultInjector()
        
        # Default SNR levels
        self.default_snr_levels = {
            'Level_I': 18,   # dB - mild anomaly
            'Level_II': 8,   # dB - moderate anomaly  
            'Level_III': 0   # dB - severe anomaly
        }
    
    def load_dataset(self) -> Dict[str, Any]:
        """Load and separate carOBD dataset into idle and motion modes."""
        print("Loading carOBD dataset...")
        
        # Get all CSV files
        all_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        print(f"Found {len(all_files)} CSV files")
        
        # Separate into idle and motion modes
        idle_files = [f for f in all_files if "idle" in os.path.basename(f)]
        motion_files = [f for f in all_files if any(mode in os.path.basename(f) 
                                                   for mode in ["drive", "live", "long", "ufpe"])]
        
        print(f"Idle mode files: {len(idle_files)}")
        print(f"Motion mode files: {len(motion_files)}")
        
        # Load idle mode data
        idle_data = []
        for file_path in idle_files:
            try:
                df = pd.read_csv(file_path, header=0)
                if len(df) > 0:
                    idle_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        # Load motion mode data
        motion_data = []
        for file_path in motion_files:
            try:
                df = pd.read_csv(file_path, header=0)
                if len(df) > 0:
                    motion_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        print(f"Successfully loaded {len(idle_data)} idle mode files")
        print(f"Successfully loaded {len(motion_data)} motion mode files")
        
        return {
            'idle_data': idle_data,
            'motion_data': motion_data,
            'idle_files': idle_files,
            'motion_files': motion_files
        }
    
    def extract_ect_rpm_features(self, dataframes: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ECT sensor and RPM features from dataframes."""
        ect_data = []
        rpm_data = []
        
        # Column indices (0-indexed)
        ect_column = 5  # COOLANT_TEMPERATURE
        rpm_column = 1  # ENGINE_RPM
        
        for df in dataframes:
            if len(df) > 0 and ect_column < len(df.columns) and rpm_column < len(df.columns):
                # Extract ECT and RPM columns
                ect_values = df.iloc[:, ect_column].values
                rpm_values = df.iloc[:, rpm_column].values
                
                # Remove any NaN values
                valid_mask = ~(np.isnan(ect_values) | np.isnan(rpm_values))
                ect_data.extend(ect_values[valid_mask])
                rpm_data.extend(rpm_values[valid_mask])
        
        return np.array(ect_data), np.array(rpm_data)
    
    def create_enhanced_fault_injected_dataset(self, normal_ect: np.ndarray, normal_rpm: np.ndarray,
                                             config: AnomalyInjectionConfig) -> Dict[str, Any]:
        """
        Create enhanced fault-injected dataset with precise tracking.
        
        Args:
            normal_ect: Normal ECT sensor data
            normal_rpm: Normal RPM data
            config: Injection configuration
            
        Returns:
            Dictionary containing enhanced fault-injected dataset
        """
        print(f"Creating enhanced fault-injected dataset for {config.mode} mode...")
        print(f"Configuration: {config}")
        
        # Create train/test split (70/30 as per paper)
        split_idx = int(len(normal_ect) * 0.7)
        
        train_ect = normal_ect[:split_idx]
        train_rpm = normal_rpm[:split_idx]
        test_ect = normal_ect[split_idx:]
        test_rpm = normal_rpm[split_idx:]
        
        # Create normal datasets
        normal_train_data = pd.DataFrame({
            'ECT': train_ect,
            'RPM': train_rpm,
            'label': 'normal',
            'split': 'train',
            'original_index': np.arange(len(train_ect))
        })
        
        normal_test_data = pd.DataFrame({
            'ECT': test_ect,
            'RPM': test_rpm,
            'label': 'normal',
            'split': 'test',
            'original_index': np.arange(len(test_ect)) + len(train_ect)
        })
        
        # Inject anomalies into test data
        modified_test_ect, modified_test_rpm, injection_records = self.fault_injector.inject_anomalies_with_tracking(
            test_ect, test_rpm, config
        )
        
        # Create anomaly dataset
        anomaly_data = pd.DataFrame({
            'ECT': modified_test_ect,
            'RPM': modified_test_rpm,
            'label': 'anomaly',
            'split': 'test',
            'original_index': np.arange(len(test_ect)) + len(train_ect)
        })
        
        # Create comprehensive dataset with all data
        combined_data = pd.concat([normal_train_data, normal_test_data, anomaly_data], ignore_index=True)
        
        # Add injection metadata
        injection_metadata = {
            'total_injections': len(injection_records),
            'injection_percentage': config.anomaly_percentage,
            'injection_strategy': config.injection_strategy,
            'snr_levels': config.snr_levels,
            'level_distribution': self.fault_injector._get_level_distribution(injection_records),
            'injection_records': [
                {
                    'original_index': record.original_index,
                    'original_value': record.original_value,
                    'injected_value': record.injected_value,
                    'snr_db': record.snr_db,
                    'injection_type': record.injection_type,
                    'timestamp': record.timestamp,
                    'noise_added': record.noise_added
                }
                for record in injection_records
            ]
        }
        
        dataset = {
            'mode': config.mode,
            'normal_train': normal_train_data,
            'normal_test': normal_test_data,
            'anomaly_data': anomaly_data,
            'combined_data': combined_data,
            'injection_metadata': injection_metadata,
            'injection_records': injection_records
        }
        
        return dataset
    
    def save_enhanced_dataset(self, dataset: Dict[str, Any], output_path: str) -> None:
        """Save the enhanced fault-injected dataset with precise tracking."""
        print(f"Saving enhanced fault-injected dataset to {output_path}...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save individual datasets
        dataset['normal_train'].to_csv(f"{output_path}/{dataset['mode']}_normal_train.csv", index=False)
        dataset['normal_test'].to_csv(f"{output_path}/{dataset['mode']}_normal_test.csv", index=False)
        dataset['anomaly_data'].to_csv(f"{output_path}/{dataset['mode']}_anomaly_injected.csv", index=False)
        dataset['combined_data'].to_csv(f"{output_path}/{dataset['mode']}_combined.csv", index=False)
        
        # Save injection metadata
        with open(f"{output_path}/{dataset['mode']}_injection_metadata.json", 'w') as f:
            json.dump(dataset['injection_metadata'], f, indent=2)
        
        # Save detailed injection records
        injection_records_df = pd.DataFrame([
            {
                'original_index': record.original_index,
                'original_value': record.original_value,
                'injected_value': record.injected_value,
                'snr_db': record.snr_db,
                'injection_type': record.injection_type,
                'timestamp': record.timestamp,
                'noise_added': record.noise_added
            }
            for record in dataset['injection_records']
        ])
        injection_records_df.to_csv(f"{output_path}/{dataset['mode']}_injection_records.csv", index=False)
        
        # Save summary statistics
        summary_stats = {
            'mode': dataset['mode'],
            'total_samples': len(dataset['combined_data']),
            'normal_train_samples': len(dataset['normal_train']),
            'normal_test_samples': len(dataset['normal_test']),
            'anomaly_samples': len(dataset['anomaly_data']),
            'injection_percentage': dataset['injection_metadata']['injection_percentage'],
            'injection_strategy': dataset['injection_metadata']['injection_strategy'],
            'level_distribution': dataset['injection_metadata']['level_distribution'],
            'snr_levels': dataset['injection_metadata']['snr_levels']
        }
        
        with open(f"{output_path}/{dataset['mode']}_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Enhanced dataset saved successfully!")
        print(f"  Normal train samples: {summary_stats['normal_train_samples']}")
        print(f"  Normal test samples: {summary_stats['normal_test_samples']}")
        print(f"  Anomaly samples: {summary_stats['anomaly_samples']}")
        print(f"  Injection percentage: {summary_stats['injection_percentage']:.2%}")
        print(f"  Level distribution: {summary_stats['level_distribution']}")

def create_enhanced_fault_injection_datasets(
    data_path: str = "../data/carOBD/obdiidata",
    output_path: str = "fault_injected_datasets",
    idle_anomaly_percentage: float = 0.05,  # 5% anomalies
    motion_anomaly_percentage: float = 0.05,  # 5% anomalies
    injection_strategy: str = 'random',
    custom_snr_levels: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Create enhanced fault injection datasets with configurable percentages.
    
    Args:
        data_path: Path to carOBD data
        output_path: Output path for datasets
        idle_anomaly_percentage: Percentage of anomalies for idle mode (0.0 to 1.0)
        motion_anomaly_percentage: Percentage of anomalies for motion mode (0.0 to 1.0)
        injection_strategy: Strategy for injection ('random', 'sequential', 'clustered')
        custom_snr_levels: Custom SNR levels (optional)
        
    Returns:
        Dictionary with dataset information
    """
    print("Enhanced Fault Injection Dataset Generator")
    print("=" * 60)
    
    # Use custom SNR levels if provided, otherwise use defaults
    snr_levels = custom_snr_levels if custom_snr_levels else {
        'Level_I': 18,   # dB - mild anomaly
        'Level_II': 8,   # dB - moderate anomaly  
        'Level_III': 0   # dB - severe anomaly
    }
    
    # Initialize processor
    processor = EnhancedCarOBDDataProcessor(data_path)
    
    # Load dataset
    data = processor.load_dataset()
    
    results = {}
    
    # Process idle mode
    print(f"\nProcessing idle mode with {idle_anomaly_percentage:.1%} anomalies...")
    idle_ect, idle_rpm = processor.extract_ect_rpm_features(data['idle_data'])
    print(f"Idle mode: {len(idle_ect)} ECT samples, {len(idle_rpm)} RPM samples")
    
    idle_config = AnomalyInjectionConfig(
        mode='idle',
        anomaly_percentage=idle_anomaly_percentage,
        snr_levels=snr_levels,
        injection_strategy=injection_strategy,
        seed=42
    )
    
    idle_dataset = processor.create_enhanced_fault_injected_dataset(idle_ect, idle_rpm, idle_config)
    processor.save_enhanced_dataset(idle_dataset, f"{output_path}/idle")
    results['idle'] = idle_dataset
    
    # Process motion mode
    print(f"\nProcessing motion mode with {motion_anomaly_percentage:.1%} anomalies...")
    motion_ect, motion_rpm = processor.extract_ect_rpm_features(data['motion_data'])
    print(f"Motion mode: {len(motion_ect)} ECT samples, {len(motion_rpm)} RPM samples")
    
    motion_config = AnomalyInjectionConfig(
        mode='motion',
        anomaly_percentage=motion_anomaly_percentage,
        snr_levels=snr_levels,
        injection_strategy=injection_strategy,
        seed=42
    )
    
    motion_dataset = processor.create_enhanced_fault_injected_dataset(motion_ect, motion_rpm, motion_config)
    processor.save_enhanced_dataset(motion_dataset, f"{output_path}/motion")
    results['motion'] = motion_dataset
    
    print(f"\n{'='*60}")
    print("Enhanced fault injection dataset generation completed!")
    print(f"Datasets saved to: {output_path}/")
    print(f"Idle anomalies: {idle_anomaly_percentage:.1%}")
    print(f"Motion anomalies: {motion_anomaly_percentage:.1%}")
    print(f"Strategy: {injection_strategy}")
    print(f"{'='*60}")
    
    return results

def main():
    """Main function with configurable parameters."""
    # Configuration parameters
    IDLE_ANOMALY_PERCENTAGE = 0.05  # 5% anomalies for idle mode
    MOTION_ANOMALY_PERCENTAGE = 0.05  # 5% anomalies for motion mode
    INJECTION_STRATEGY = 'random'  # 'random', 'sequential', 'clustered'
    
    # Custom SNR levels (optional)
    custom_snr_levels = {
        'Level_I': 18,   # dB - mild anomaly
        'Level_II': 8,   # dB - moderate anomaly  
        'Level_III': 0   # dB - severe anomaly
    }
    
    # Create enhanced datasets
    results = create_enhanced_fault_injection_datasets(
        data_path="../data/carOBD/obdiidata",
        output_path="fault_injected_datasets",
        idle_anomaly_percentage=IDLE_ANOMALY_PERCENTAGE,
        motion_anomaly_percentage=MOTION_ANOMALY_PERCENTAGE,
        injection_strategy=INJECTION_STRATEGY,
        custom_snr_levels=custom_snr_levels
    )
    
    return results

if __name__ == "__main__":
    results = main()
