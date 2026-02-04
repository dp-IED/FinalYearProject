"""
Sensor-specific configuration for rule-based summarization.

Defines thresholds for each sensor type:
- Normal ranges (for level classification: low/normal/high)
- Spike detection thresholds (absolute change per timestep)
- Dropout detection thresholds (near-zero value threshold, min duration)
- Plateau detection thresholds (variance threshold, min duration)
- Trend epsilon (for stable vs increasing/decreasing)
"""

from typing import Dict, Tuple

# Sensor-specific configuration
SENSOR_CONFIG: Dict[str, Dict] = {
    "ENGINE_RPM ()": {
        "normal_range": (600, 6000),  # rpm
        "spike_threshold": 500,  # rpm change per timestep
        "dropout_threshold": 50,  # rpm (near-zero threshold)
        "dropout_min_duration": 5,  # timesteps
        "plateau_variance_threshold": 100,  # variance threshold
        "plateau_min_duration": 20,  # timesteps
        "trend_epsilon": 10,  # rpm per timestep
    },
    "VEHICLE_SPEED ()": {
        "normal_range": (0, 120),  # mi/h
        "spike_threshold": 10,  # mi/h change per timestep
        "dropout_threshold": 1,  # mi/h (near-zero threshold)
        "dropout_min_duration": 3,  # timesteps
        "plateau_variance_threshold": 2,  # variance threshold
        "plateau_min_duration": 15,  # timesteps
        "trend_epsilon": 0.5,  # mi/h per timestep
    },
    "THROTTLE ()": {
        "normal_range": (0, 100),  # percentage
        "spike_threshold": 15,  # percentage change per timestep
        "dropout_threshold": 2,  # percentage (near-zero threshold)
        "dropout_min_duration": 5,  # timesteps
        "plateau_variance_threshold": 5,  # variance threshold
        "plateau_min_duration": 20,  # timesteps
        "trend_epsilon": 0.5,  # percentage per timestep
    },
    "ENGINE_LOAD ()": {
        "normal_range": (0, 100),  # percentage
        "spike_threshold": 20,  # percentage change per timestep
        "dropout_threshold": 5,  # percentage (near-zero threshold)
        "dropout_min_duration": 5,  # timesteps
        "plateau_variance_threshold": 10,  # variance threshold
        "plateau_min_duration": 20,  # timesteps
        "trend_epsilon": 1.0,  # percentage per timestep
    },
    "COOLANT_TEMPERATURE ()": {
        "normal_range": (70, 110),  # Celsius
        "spike_threshold": 5,  # Celsius change per timestep
        "dropout_threshold": 20,  # Celsius (near-zero threshold - engine off)
        "dropout_min_duration": 10,  # timesteps
        "plateau_variance_threshold": 2,  # variance threshold
        "plateau_min_duration": 30,  # timesteps (coolant temp is slow-changing)
        "trend_epsilon": 0.1,  # Celsius per timestep
    },
    "INTAKE_MANIFOLD_PRESSURE ()": {
        "normal_range": (0, 20),  # psig
        "spike_threshold": 3,  # psig change per timestep
        "dropout_threshold": 0.5,  # psig (near-zero threshold)
        "dropout_min_duration": 5,  # timesteps
        "plateau_variance_threshold": 0.5,  # variance threshold
        "plateau_min_duration": 20,  # timesteps
        "trend_epsilon": 0.2,  # psig per timestep
    },
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()": {
        "normal_range": (-25, 25),  # percentage
        "spike_threshold": 10,  # percentage change per timestep
        "dropout_threshold": 0.5,  # percentage (near-zero threshold)
        "dropout_min_duration": 5,  # timesteps
        "plateau_variance_threshold": 2,  # variance threshold
        "plateau_min_duration": 15,  # timesteps
        "trend_epsilon": 0.5,  # percentage per timestep
    },
    "LONG_TERM_FUEL_TRIM_BANK_1 ()": {
        "normal_range": (-25, 25),  # percentage
        "spike_threshold": 5,  # percentage change per timestep (long-term is slower)
        "dropout_threshold": 0.5,  # percentage (near-zero threshold)
        "dropout_min_duration": 5,  # timesteps
        "plateau_variance_threshold": 2,  # variance threshold
        "plateau_min_duration": 20,  # timesteps (long-term is more stable)
        "trend_epsilon": 0.2,  # percentage per timestep
    },
}


def get_sensor_config(sensor_name: str) -> Dict:
    """
    Get configuration for a specific sensor.
    
    Args:
        sensor_name: Name of the sensor
        
    Returns:
        Dictionary with sensor configuration, or default config if sensor not found
    """
    return SENSOR_CONFIG.get(sensor_name, {
        "normal_range": (0, 100),
        "spike_threshold": 10,
        "dropout_threshold": 1,
        "dropout_min_duration": 5,
        "plateau_variance_threshold": 5,
        "plateau_min_duration": 20,
        "trend_epsilon": 0.5,
    })


def get_normal_range(sensor_name: str) -> Tuple[float, float]:
    """
    Get normal range for a sensor.
    
    Args:
        sensor_name: Name of the sensor
        
    Returns:
        Tuple of (min, max) normal values
    """
    config = get_sensor_config(sensor_name)
    return config["normal_range"]
