from typing import Tuple, Dict, Optional, List
import numpy as np

def create_realistic_fault_data(
    normal_data, 
    fault_percentage: float = 0.2,
    random_state: Optional[int] = None,
    fault_types: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if fault_types is None:
        fault_types = [
            'gradual_drift',
            'intermittent_spike', 
            'slow_response',
            'bias_offset',
            'electrical_jitter'
        ]
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(normal_data)
    n_sensors = normal_data.shape[1]
    
    # Initialize outputs
    fault_features = normal_data.copy()
    fault_labels = np.zeros(n_samples, dtype=int)
    
    # Metadata tracking
    fault_type_list = [None] * n_samples
    original_values = [[] for _ in range(n_samples)]
    modified_values = [[] for _ in range(n_samples)]
    
    # Calculate how many fault events to inject
    target_fault_timesteps = int(n_samples * fault_percentage)
    current_fault_count = 0
    
    # Inject faults until we reach target percentage
    while current_fault_count < target_fault_timesteps:
        # Choose fault type
        fault_type = np.random.choice(fault_types)
        
        # Choose sensor and start position
        sensor_idx = np.random.randint(0, n_sensors)
        start_idx = np.random.randint(0, n_samples - 50)  # Leave room for fault duration
        
        # Inject based on fault type
        if fault_type == 'gradual_drift':
            duration = min(np.random.randint(50, 200), n_samples - start_idx)
            end_idx = start_idx + duration
            
            # Drift magnitude: 0.5 to 2.0 std (subtle drift)
            drift_magnitude = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
            drift = np.linspace(0, drift_magnitude, duration)
            
            # Store original values
            for t in range(start_idx, end_idx):
                original_values[t].append(fault_features[t, sensor_idx])
                fault_type_list[t] = 'gradual_drift'
            
            # Apply drift
            fault_features[start_idx:end_idx, sensor_idx] += drift
            
            # Store modified values
            for t in range(start_idx, end_idx):
                modified_values[t].append(fault_features[t, sensor_idx])
                fault_labels[t] = 1
            
            current_fault_count += duration
        
        elif fault_type == 'intermittent_spike':
            duration = min(np.random.randint(1, 4), n_samples - start_idx)
            end_idx = start_idx + duration
            
            # Large but brief spike
            spike_value = np.random.uniform(3.0, 5.0) * np.random.choice([-1, 1])
            
            for t in range(start_idx, end_idx):
                original_values[t].append(fault_features[t, sensor_idx])
                fault_type_list[t] = 'intermittent_spike'
            
            fault_features[start_idx:end_idx, sensor_idx] += spike_value
            
            for t in range(start_idx, end_idx):
                modified_values[t].append(fault_features[t, sensor_idx])
                fault_labels[t] = 1
            
            current_fault_count += duration
        
        elif fault_type == 'slow_response':
            duration = min(np.random.randint(5, 15), n_samples - start_idx)
            end_idx = start_idx + duration
            
            # Freeze at starting value
            stuck_value = fault_features[start_idx, sensor_idx]
            
            for t in range(start_idx, end_idx):
                original_values[t].append(fault_features[t, sensor_idx])
                fault_type_list[t] = 'slow_response'
            
            fault_features[start_idx:end_idx, sensor_idx] = stuck_value
            
            for t in range(start_idx, end_idx):
                modified_values[t].append(fault_features[t, sensor_idx])
                fault_labels[t] = 1
            
            current_fault_count += duration
        
        elif fault_type == 'bias_offset':
            duration = min(np.random.randint(100, 300), n_samples - start_idx)
            end_idx = start_idx + duration
            
            # Small consistent bias
            bias = np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1])
            
            for t in range(start_idx, end_idx):
                original_values[t].append(fault_features[t, sensor_idx])
                fault_type_list[t] = 'bias_offset'
            
            fault_features[start_idx:end_idx, sensor_idx] += bias
            
            for t in range(start_idx, end_idx):
                modified_values[t].append(fault_features[t, sensor_idx])
                fault_labels[t] = 1
            
            current_fault_count += duration
        
        elif fault_type == 'electrical_jitter':
            duration = min(np.random.randint(20, 60), n_samples - start_idx)
            end_idx = start_idx + duration
            
            # High-frequency noise
            noise = np.random.normal(0, 0.8, duration)
            
            for t in range(start_idx, end_idx):
                original_values[t].append(fault_features[t, sensor_idx])
                fault_type_list[t] = 'electrical_jitter'
            
            fault_features[start_idx:end_idx, sensor_idx] += noise
            
            for t in range(start_idx, end_idx):
                modified_values[t].append(fault_features[t, sensor_idx])
                fault_labels[t] = 1
            
            current_fault_count += duration
        
        # Safety check to avoid infinite loop
        if current_fault_count > target_fault_timesteps * 1.5:
            break
    
    # Create fault info dictionary
    fault_info = {
        'fault_types': fault_type_list,
        'original_values': original_values,
        'modified_values': modified_values,
        'n_faults': fault_labels.sum(),
        'fault_percentage': fault_labels.mean()
    }
    
    return fault_features, fault_labels, fault_info