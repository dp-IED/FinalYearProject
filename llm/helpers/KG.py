"""
Knowledge Graph Helper Class for constructing semantic knowledge graphs from GDN outputs.

This module provides temporal traversal of windows from gdn.ipynb, building relationships
between sensors across time and tracking anomaly propagation for automotive diagnostics.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# Sensor Metadata and Subsystem Mapping
# ============================================================================

SENSOR_SUBSYSTEMS = {
    'ENGINE_RPM ()': 'Engine System',
    'ENGINE_LOAD ()': 'Engine System',
    'COOLANT_TEMPERATURE ()': 'Engine System',
    'SHORT_TERM_FUEL_TRIM_BANK_1 ()': 'Fuel System',
    'LONG_TERM_FUEL_TRIM_BANK_1 ()': 'Fuel System',
    'INTAKE_MANIFOLD_PRESSURE ()': 'Intake System',
    'THROTTLE ()': 'Intake System',
    'VEHICLE_SPEED ()': 'Drivetrain',
}

SENSOR_DESCRIPTIONS = {
    'ENGINE_RPM ()': {
        'description': 'Engine speed in revolutions per minute',
        'unit': 'rpm',
        'normal_range': (600, 6000),
        'fault_injection_eligible': True,
        'subsystem': 'Engine System'
    },
    'VEHICLE_SPEED ()': {
        'description': 'Vehicle speed sensor reading',
        'unit': 'mi/h',
        'normal_range': (0, 120),
        'fault_injection_eligible': True,
        'subsystem': 'Drivetrain'
    },
    'THROTTLE ()': {
        'description': 'Throttle position percentage',
        'unit': '%',
        'normal_range': (0, 100),
        'fault_injection_eligible': True,
        'subsystem': 'Intake System'
    },
    'ENGINE_LOAD ()': {
        'description': 'Calculated engine load value',
        'unit': '%',
        'normal_range': (0, 100),
        'fault_injection_eligible': True,
        'subsystem': 'Engine System'
    },
    'COOLANT_TEMPERATURE ()': {
        'description': 'Engine coolant temperature',
        'unit': 'C',
        'normal_range': (70, 110),
        'fault_injection_eligible': True,
        'subsystem': 'Engine System'
    },
    'INTAKE_MANIFOLD_PRESSURE ()': {
        'description': 'Intake manifold absolute pressure',
        'unit': 'psig',
        'normal_range': (0, 20),
        'fault_injection_eligible': True,
        'subsystem': 'Intake System'
    },
    'SHORT_TERM_FUEL_TRIM_BANK_1 ()': {
        'description': 'Short-term fuel trim adjustment',
        'unit': '%',
        'normal_range': (-25, 25),
        'fault_injection_eligible': True,
        'subsystem': 'Fuel System'
    },
    'LONG_TERM_FUEL_TRIM_BANK_1 ()': {
        'description': 'Long-term fuel trim adjustment',
        'unit': '%',
        'normal_range': (-25, 25),
        'fault_injection_eligible': True,
        'subsystem': 'Fuel System'
    },
}

# Expected correlations between sensors (normal operation)
EXPECTED_CORRELATIONS = {
    ('ENGINE_RPM ()', 'VEHICLE_SPEED ()'): {
        'type': 'expected_to_increase_with',
        'strength': 'strong',
        'description': 'RPM and vehicle speed should correlate positively under normal conditions'
    },
    ('THROTTLE ()', 'ENGINE_LOAD ()'): {
        'type': 'expected_to_increase_with',
        'strength': 'strong',
        'description': 'Throttle position and engine load should increase together'
    },
    ('THROTTLE ()', 'INTAKE_MANIFOLD_PRESSURE ()'): {
        'type': 'expected_to_increase_with',
        'strength': 'moderate',
        'description': 'Throttle opening increases intake manifold pressure'
    },
    ('ENGINE_RPM ()', 'COOLANT_TEMPERATURE ()'): {
        'type': 'correlates_with',
        'strength': 'weak',
        'description': 'Higher RPM may increase coolant temperature over time'
    },
    ('SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()'): {
        'type': 'correlates_with',
        'strength': 'moderate',
        'description': 'Short-term and long-term fuel trim adjustments are related'
    },
    ('INTAKE_MANIFOLD_PRESSURE ()', 'SHORT_TERM_FUEL_TRIM_BANK_1 ()'): {
        'type': 'correlates_with',
        'strength': 'moderate',
        'description': 'Intake pressure affects fuel trim calculations'
    },
}


# ============================================================================
# Fault Knowledge Base
# ============================================================================

FAULT_DOCUMENTS = {
    'VSS_DROPOUT': {
        'name': 'Vehicle Speed Sensor Dropout',
        'description': 'Sudden drop to zero followed by noise in vehicle speed sensor',
        'affected_sensors': ['VEHICLE_SPEED ()'],
        'expected_correlations': {
            'violates': [('ENGINE_RPM ()', 'VEHICLE_SPEED ()')],
            'description': 'RPM-Speed correlation breaks down when VSS drops out'
        },
        'diagnostic_patterns': [
            'Sudden drop to zero in VEHICLE_SPEED',
            'RPM continues normally while speed reads zero',
            'Intermittent noise after dropout'
        ],
        'obd_codes': ['P0500', 'P0501', 'P0502'],
        'propagation_pattern': 'May affect transmission-related sensors'
    },
    'MAF_SCALE_LOW': {
        'name': 'Mass Air Flow Scale Low',
        'description': 'Entire signal scaled down 20-25%',
        'affected_sensors': ['INTAKE_MANIFOLD_PRESSURE ()'],
        'expected_correlations': {
            'violates': [
                ('INTAKE_MANIFOLD_PRESSURE ()', 'SHORT_TERM_FUEL_TRIM_BANK_1 ()'),
                ('THROTTLE ()', 'INTAKE_MANIFOLD_PRESSURE ()')
            ],
            'description': 'Fuel trim and throttle correlations affected by scaled pressure readings'
        },
        'diagnostic_patterns': [
            'Consistent 20-25% reduction in intake pressure',
            'Fuel trim adjustments increase',
            'Throttle-pressure relationship maintained but shifted'
        ],
        'obd_codes': ['P0101', 'P0102', 'P0103'],
        'propagation_pattern': 'Affects fuel system sensors'
    },
    'COOLANT_DROPOUT': {
        'name': 'Coolant Temperature Dropout',
        'description': 'Multiple sharp dips in coolant temperature',
        'affected_sensors': ['COOLANT_TEMPERATURE ()'],
        'expected_correlations': {
            'violates': [('ENGINE_RPM ()', 'COOLANT_TEMPERATURE ()')],
            'description': 'RPM-temperature correlation disrupted during dropouts'
        },
        'diagnostic_patterns': [
            'Intermittent sharp drops in coolant temperature',
            'Temperature readings inconsistent with engine load',
            'Multiple rapid fluctuations'
        ],
        'obd_codes': ['P0115', 'P0116', 'P0117', 'P0118'],
        'propagation_pattern': 'May trigger engine protection modes affecting other sensors'
    },
    'TPS_STUCK': {
        'name': 'Throttle Position Sensor Stuck',
        'description': 'Value freezes in second half of window',
        'affected_sensors': ['THROTTLE ()'],
        'expected_correlations': {
            'violates': [
                ('THROTTLE ()', 'ENGINE_LOAD ()'),
                ('THROTTLE ()', 'INTAKE_MANIFOLD_PRESSURE ()')
            ],
            'description': 'Throttle-load and throttle-pressure correlations break when TPS is stuck'
        },
        'diagnostic_patterns': [
            'Throttle value remains constant while other sensors change',
            'Engine load and intake pressure continue to vary',
            'Mismatch between throttle position and actual engine state'
        ],
        'obd_codes': ['P0120', 'P0121', 'P0122', 'P0123'],
        'propagation_pattern': 'Affects intake and fuel system sensors'
    },
    'RPM_SPEED_DECOUPLE': {
        'name': 'RPM-Speed Decoupling',
        'description': 'RPM and speed signals diverge, indicating transmission/clutch issues',
        'affected_sensors': ['ENGINE_RPM ()', 'VEHICLE_SPEED ()'],
        'expected_correlations': {
            'violates': [('ENGINE_RPM ()', 'VEHICLE_SPEED ()')],
            'description': 'Strong correlation between RPM and speed breaks down'
        },
        'diagnostic_patterns': [
            'RPM increases while speed remains low or constant',
            'Speed increases while RPM remains constant',
            'Ratio between RPM and speed becomes inconsistent'
        ],
        'obd_codes': ['P0700', 'P0730', 'P0731', 'P0732'],
        'propagation_pattern': 'May affect drivetrain-related sensors'
    },
    'gradual_drift': {
        'name': 'Gradual Sensor Drift',
        'description': 'Slow sensor drift patterns over time',
        'affected_sensors': ['*'],  # Can affect any sensor
        'expected_correlations': {
            'violates': [],
            'description': 'Gradual drift may slowly degrade correlations'
        },
        'diagnostic_patterns': [
            'Gradual change in sensor values over multiple windows',
            'Correlation strength slowly decreases',
            'Values drift outside normal operating range'
        ],
        'obd_codes': ['P0125', 'P0130', 'P0135'],
        'propagation_pattern': 'May affect sensors in same subsystem'
    },
    'intermittent_spike': {
        'name': 'Intermittent Spike',
        'description': 'Sudden spikes, electrical interference',
        'affected_sensors': ['*'],
        'expected_correlations': {
            'violates': [],
            'description': 'Spikes cause temporary correlation violations'
        },
        'diagnostic_patterns': [
            'Sudden large deviations from normal values',
            'Brief duration (1-3 timesteps)',
            'Returns to normal after spike'
        ],
        'obd_codes': ['P0131', 'P0132', 'P0133'],
        'propagation_pattern': 'Usually isolated, minimal propagation'
    },
    'slow_response': {
        'name': 'Slow Sensor Response',
        'description': 'Sensor response delays',
        'affected_sensors': ['*'],
        'expected_correlations': {
            'violates': [],
            'description': 'Delayed response causes temporal misalignment in correlations'
        },
        'diagnostic_patterns': [
            'Sensor value changes lag behind expected timing',
            'Correlation with related sensors shows temporal offset',
            'Values freeze or change slowly'
        ],
        'obd_codes': ['P0134', 'P0135'],
        'propagation_pattern': 'May affect sensors that depend on this sensor'
    },
    'bias_offset': {
        'name': 'Bias Offset',
        'description': 'Constant offset errors',
        'affected_sensors': ['*'],
        'expected_correlations': {
            'violates': [],
            'description': 'Constant offset may shift correlation baseline'
        },
        'diagnostic_patterns': [
            'Consistent offset from expected values',
            'Correlation patterns maintained but shifted',
            'Values consistently above or below normal range'
        ],
        'obd_codes': ['P0136', 'P0137'],
        'propagation_pattern': 'May affect sensors that use this sensor as input'
    },
    'electrical_jitter': {
        'name': 'Electrical Jitter',
        'description': 'High-frequency noise patterns',
        'affected_sensors': ['*'],
        'expected_correlations': {
            'violates': [],
            'description': 'High-frequency noise degrades correlation quality'
        },
        'diagnostic_patterns': [
            'High-frequency oscillations in sensor values',
            'Noise pattern visible in time series',
            'Correlation strength reduced due to noise'
        ],
        'obd_codes': ['P0138', 'P0139'],
        'propagation_pattern': 'Usually isolated, may affect signal quality'
    },
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WindowStats:
    """Statistical summary for a sensor within a window"""
    mean: float
    std: float
    min: float
    max: float
    variance: float
    num_zeros: int  # Count of zero readings (important for dropout detection)
    trend: float  # Linear regression slope
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    variation_from_normal: float  # Deviation from expected normal range
    anomaly_score: float = 0.0


@dataclass
class TemporalEdge:
    """Temporal edge connecting sensors across consecutive windows"""
    source_window: int
    target_window: int
    source_sensor: str
    target_sensor: str
    edge_type: str  # 'temporal_continuation', 'value_change', 'relationship_evolution', 'anomaly_propagation'
    value_change: float = 0.0
    correlation_change: float = 0.0
    metadata: Dict[str, Any] = None


# ============================================================================
# Knowledge Graph Builder Class
# ============================================================================

class KnowledgeGraphBuilder:
    """
    Builds a semantic knowledge graph from GDN outputs by temporally traversing windows.
    
    Processes windows sequentially, builds sensor relationships within each window,
    tracks relationships across consecutive windows, and monitors anomaly propagation.
    
    Edge Types:
    - 'correlates_with': All correlations between sensors (single edge type with rich attributes)
      Attributes include:
        - correlation_strength: Absolute value of correlation (0-1)
        - correlation_direction: 'positive' or 'negative'
        - domain_expected_type: Domain knowledge expectation type (if exists)
        - domain_expected_strength: 'strong', 'moderate', or 'weak' (if exists)
        - violates_domain_expectation: Boolean flag for domain knowledge violations
        - expected_correlation_gdn: GDN-learned expected correlation
        - deviation_from_gdn: Difference from GDN expectation
        - violates_gdn_expectation: Boolean flag for GDN expectation violations
        - gdn_score_source: GDN anomaly prediction for source sensor
        - gdn_score_target: GDN anomaly prediction for target sensor
        - potential_fault_indicator: Boolean flag when violations + high GDN scores
    
    - 'correlation_evolution': Temporal edges tracking correlation changes across windows
      Attributes include:
        - correlation_change: Change in correlation strength
        - prev_correlation: Previous window correlation strength
        - curr_correlation: Current window correlation strength
        - evolution_type: 'strengthening', 'weakening', or 'stable'
    
    - 'temporal_continuation': Connects same sensor across consecutive windows
    - 'value_change': Tracks significant value changes in sensors
    - 'anomaly_propagation': Tracks fault propagation based on GDN predictions
    """
    
    def __init__(self, sensor_names: List[str], sensor_embeddings: np.ndarray, 
                 adjacency_matrix: np.ndarray, sensor_centers: Optional[np.ndarray] = None):
        """
        Initialize the Knowledge Graph Builder.
        
        Args:
            sensor_names: List of sensor names (must match order in data)
            sensor_embeddings: Learned sensor embeddings from GDN (num_sensors, embed_dim)
            adjacency_matrix: Adjacency matrix from GDN (num_sensors, num_sensors)
            sensor_centers: Optional (num_sensors, 2, hidden_dim) array - sensor-specific centers
                          from multi-level center loss. If None, uses window-level center for all sensors.
        """
        self.sensor_names = sensor_names
        self.sensor_embeddings = sensor_embeddings
        self.adjacency_matrix = adjacency_matrix
        self.sensor_centers = sensor_centers  # (num_sensors, 2, hidden_dim) or None
        self.num_sensors = len(sensor_names)
        
        # Create sensor name to index mapping
        self.sensor_to_idx = {name: idx for idx, name in enumerate(sensor_names)}
        
        # Initialize graph structures
        self.kg = nx.MultiDiGraph()  # Main knowledge graph
        self.window_graphs = {}  # Per-window graphs
        self.window_stats = {}  # Per-window statistics
        self.temporal_edges = []  # Temporal edges between windows
        self.anomaly_propagation_chains = []  # Fault propagation chains
        self.window_embeddings = {}  # Per-window embedding data
        
        # Store raw window data for Layer 3 (time-series access)
        self.X_windows = None  # Normalized windows (N, 300, 8)
        self.X_windows_unnormalized = None  # Unnormalized windows (N, 300, 8)
        
        # Initialize sensor nodes in KG
        self._initialize_sensor_nodes()
        
    def _initialize_sensor_nodes(self):
        """Initialize sensor nodes in the knowledge graph with metadata"""
        for sensor_name in self.sensor_names:
            subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, 'Unknown')
            description = SENSOR_DESCRIPTIONS.get(sensor_name, {})
            
            self.kg.add_node(
                sensor_name,
                type='sensor',
                subsystem=subsystem,
                description=description.get('description', ''),
                unit=description.get('unit', ''),
                normal_range=description.get('normal_range', (0, 100)),
                fault_injection_eligible=description.get('fault_injection_eligible', False)
            )
    
    def build_from_gdn_windows(self, X_windows: np.ndarray, gdn_predictions: np.ndarray,
                                X_windows_unnormalized: Optional[np.ndarray] = None) -> nx.MultiDiGraph:
        """
        Main entry point: Build KG by traversing windows temporally.
        
        Builds knowledge graph from GDN model outputs (predictions), NOT ground truth labels.
        The KG contains evidence (prediction scores, correlations, statistical features) that
        the LLM can reason over, not the conclusion (ground truth labels).
        
        Args:
            X_windows: (num_windows, 300, 8) array - normalized sensor data windows
            gdn_predictions: (num_windows, 8) array - GDN anomaly scores (0.0-1.0) per sensor per window
            X_windows_unnormalized: (num_windows, 300, 8) array - unnormalized sensor data windows (optional)
            
        Returns:
            Knowledge graph with temporal traversal
        """
        num_windows = len(X_windows)
        
        # Store window data for Layer 3 (time-series access)
        self.X_windows = X_windows
        self.X_windows_unnormalized = X_windows_unnormalized
        
        # Traverse windows sequentially
        for window_idx in range(num_windows):
            window_data = X_windows[window_idx]  # (300, 8)
            window_gdn_scores = gdn_predictions[window_idx]  # (8,) - GDN predictions for this window
            
            self._process_window(window_idx, window_data, window_gdn_scores)
            
            if window_idx > 0:
                self._build_temporal_edges(window_idx - 1, window_idx, 
                                         X_windows[window_idx - 1], window_data,
                                         gdn_predictions[window_idx - 1], window_gdn_scores)
        
        self._track_anomaly_propagation(gdn_predictions)
        
        return self.kg
    
    def store_window_embeddings(
        self,
        window_idx: int,
        embedding: np.ndarray,
        dist_normal: float,
        dist_anomalous: float
    ) -> None:
        """
        Store window embedding data for a specific window.
        
        Args:
            window_idx: Index of the window
            embedding: (hidden_dim,) numpy array - window embedding vector
            dist_normal: Euclidean distance to normal center
            dist_anomalous: Euclidean distance to anomalous center
        """
        # Compute confidence: sigmoid of distance difference
        # Higher confidence when dist_normal < dist_anomalous (closer to normal)
        # Lower confidence when dist_normal > dist_anomalous (closer to anomalous)
        confidence = 1.0 / (1.0 + np.exp(dist_normal - dist_anomalous))
        
        self.window_embeddings[window_idx] = {
            'embedding': embedding.copy(),  # Store copy to avoid reference issues
            'dist_normal': float(dist_normal),
            'dist_anomalous': float(dist_anomalous),
            'confidence': float(confidence)
        }
    
    def _process_window(self, window_idx: int, window_data: np.ndarray,
                       gdn_scores: np.ndarray):
        """
        Process a single window using GDN predictions, not ground truth labels.
        
        - Compute within-window sensor correlations
        - Compare to expected correlations (from adjacency_matrix)
        - Detect relationship violations
        - Add nodes and edges for this window
        
        Args:
            window_idx: Index of the window
            window_data: (300, 8) array - normalized sensor data for this window
            gdn_scores: (8,) array - GDN anomaly prediction scores (0.0-1.0) per sensor
        """
        window_stats = {}
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            sensor_values = window_data[:, sensor_idx]
            
            mean_val = float(np.mean(sensor_values))
            std_val = float(np.std(sensor_values))
            variance_val = float(np.var(sensor_values))
            num_zeros_val = int(np.sum(sensor_values == 0))
            
            timesteps = np.arange(len(sensor_values))
            if len(sensor_values) > 1 and np.var(timesteps) > 0:
                trend_val = float(np.polyfit(timesteps, sensor_values, 1)[0])
            else:
                trend_val = 0.0
            
            median_val = float(np.median(sensor_values))
            q25_val = float(np.percentile(sensor_values, 25))
            q75_val = float(np.percentile(sensor_values, 75))
            
            stats = WindowStats(
                mean=mean_val,
                std=std_val,
                min=float(np.min(sensor_values)),
                max=float(np.max(sensor_values)),
                variance=variance_val,
                num_zeros=num_zeros_val,
                trend=trend_val,
                median=median_val,
                q25=q25_val,
                q75=q75_val,
                variation_from_normal=self._compute_variation_from_normal(
                    sensor_name, sensor_values
                ),
                anomaly_score=float(gdn_scores[sensor_idx])  # GDN prediction, not ground truth
            )
            window_stats[sensor_name] = stats
        
        self.window_stats[window_idx] = window_stats
        
        correlation_matrix = np.corrcoef(window_data.T)
        
        window_graph = nx.Graph()
        
        # Use prediction threshold (0.5) to determine if sensor is likely anomalous
        prediction_threshold = 0.5
        for sensor_name, stats in window_stats.items():
            window_graph.add_node(
                sensor_name,
                window_idx=window_idx,
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                variation_from_normal=stats.variation_from_normal,
                is_faulty=bool(stats.anomaly_score > prediction_threshold)  # Based on GDN prediction threshold
            )
        
        for i, sensor_i in enumerate(self.sensor_names):
            for j, sensor_j in enumerate(self.sensor_names):
                if i >= j:
                    continue
                
                window_corr = correlation_matrix[i, j]
                
                expected_corr = self.adjacency_matrix[i, j]
                
                edge_type, edge_attrs = self._infer_semantic_edge(
                    sensor_i, sensor_j, window_corr, expected_corr, 
                    window_stats[sensor_i], window_stats[sensor_j]
                )
                
                if edge_type:
                    # Build edge data with new rich attributes
                    # Preserve 'correlation' for backward compatibility
                    edge_data = {
                        'window_idx': window_idx,
                        'edge_type': edge_type,
                        'correlation': float(window_corr),  # Preserved for backward compatibility
                        **edge_attrs  # Contains all new rich attributes
                    }
                    # Ensure edge_type from method takes precedence
                    if 'edge_type' in edge_attrs:
                        edge_data['edge_type'] = edge_attrs['edge_type']
                    
                    window_graph.add_edge(sensor_i, sensor_j, **edge_data)
                    
                    self.kg.add_edge(sensor_i, sensor_j, **edge_data)
        
        self.window_graphs[window_idx] = window_graph
    
    def _infer_semantic_edge(self, sensor_i: str, sensor_j: str, 
                            window_corr: float, expected_corr_gdn: float,
                            stats_i: WindowStats, stats_j: WindowStats) -> Tuple[Optional[str], Dict]:
        """
        Create edges based on observed correlations with semantic labels.
        
        Strategy:
        1. Create edges for ALL significant correlations (baseline: 'correlates_with')
        2. Add semantic labels from domain knowledge (EXPECTED_CORRELATIONS)
        3. Flag violations when observed differs from expected (both domain and GDN)
        
        This method uses a single edge type ('correlates_with') with rich attributes
        that capture semantic meaning, rather than multiple edge types. This makes
        querying simpler and more flexible for LLM reasoning.
        
        Args:
            sensor_i: Name of source sensor
            sensor_j: Name of target sensor
            window_corr: Observed correlation coefficient (-1 to 1)
            expected_corr_gdn: GDN-learned expected correlation from adjacency matrix
            stats_i: WindowStats for source sensor
            stats_j: WindowStats for target sensor
        
        Returns:
            (edge_type, edge_attributes) or (None, {}) if no edge should be created
        """
        # Filter only invalid/noise correlations
        if np.isnan(window_corr) or np.isinf(window_corr):
            return None, {}
        
        # Lower threshold to capture more relationships (LLM can reason about significance)
        corr_threshold = 0.1  # Reduced from 0.2
        
        if abs(window_corr) < corr_threshold:
            return None, {}
        
        # Lookup domain knowledge (automotive physics expectations)
        expected_rel = EXPECTED_CORRELATIONS.get((sensor_i, sensor_j)) or \
                       EXPECTED_CORRELATIONS.get((sensor_j, sensor_i))
        
        # ===== STEP 1: Determine primary edge type =====
        # Default: all correlations are 'correlates_with'
        edge_type = 'correlates_with'
        
        edge_attrs = {
            'correlation_strength': abs(window_corr),
            'correlation_direction': 'positive' if window_corr > 0 else 'negative'
        }
        
        # ===== STEP 2: Add domain knowledge labels =====
        if expected_rel:
            domain_expected_type = expected_rel['type']
            domain_strength = expected_rel['strength']  # 'strong', 'moderate', 'weak'
            
            edge_attrs['domain_expected_type'] = domain_expected_type
            edge_attrs['domain_expected_strength'] = domain_strength
            
            # Check if domain expectation is violated
            # Map strength to threshold
            strength_thresholds = {
                'strong': 0.6,    # Strong correlations should be > 0.6
                'moderate': 0.4,  # Moderate should be > 0.4
                'weak': 0.2       # Weak should be > 0.2
            }
            expected_threshold = strength_thresholds.get(domain_strength, 0.5)
            
            # Violation conditions:
            # 1. Expected positive but observed negative (or vice versa)
            # 2. Expected strong but observed weak
            is_sign_mismatch = False
            if 'increase_with' in domain_expected_type and window_corr < 0:
                is_sign_mismatch = True
            
            is_magnitude_violation = abs(window_corr) < expected_threshold
            
            if is_sign_mismatch or is_magnitude_violation:
                edge_attrs['violates_domain_expectation'] = True
                edge_attrs['violation_type'] = 'sign_mismatch' if is_sign_mismatch else 'magnitude_too_weak'
            else:
                edge_attrs['violates_domain_expectation'] = False
        else:
            # No domain knowledge for this pair
            edge_attrs['violates_domain_expectation'] = False
        
        # ===== STEP 3: Add GDN model expectation =====
        deviation_from_gdn = abs(window_corr - expected_corr_gdn)
        edge_attrs['expected_correlation_gdn'] = float(expected_corr_gdn)
        edge_attrs['deviation_from_gdn'] = float(deviation_from_gdn)
        
        # Flag if GDN expected different pattern
        gdn_violation_threshold = 0.3
        if deviation_from_gdn > gdn_violation_threshold:
            edge_attrs['violates_gdn_expectation'] = True
        else:
            edge_attrs['violates_gdn_expectation'] = False
        
        # ===== STEP 4: Add GDN anomaly context =====
        edge_attrs['gdn_score_source'] = float(stats_i.anomaly_score)
        edge_attrs['gdn_score_target'] = float(stats_j.anomaly_score)
        
        # High GDN scores + violation = strong fault evidence (but don't label as "supports_fault")
        if (edge_attrs.get('violates_domain_expectation', False) or 
            edge_attrs.get('violates_gdn_expectation', False)):
            if stats_i.anomaly_score > 0.5 or stats_j.anomaly_score > 0.5:
                edge_attrs['potential_fault_indicator'] = True
            else:
                edge_attrs['potential_fault_indicator'] = False
        else:
            edge_attrs['potential_fault_indicator'] = False
        
        edge_attrs['edge_type'] = edge_type
        return edge_type, edge_attrs
    
    def _build_temporal_edges(self, prev_window_idx: int, curr_window_idx: int,
                             prev_window_data: np.ndarray, curr_window_data: np.ndarray,
                             prev_gdn_scores: np.ndarray, curr_gdn_scores: np.ndarray):
        """
        Build temporal edges connecting consecutive windows using GDN predictions.
        
        - Connect same sensor across windows
        - Track value changes
        - Track relationship evolution
        - Track anomaly propagation (based on GDN predictions, not ground truth)
        
        Args:
            prev_window_idx: Index of previous window
            curr_window_idx: Index of current window
            prev_window_data: (300, 8) array - previous window data
            curr_window_data: (300, 8) array - current window data
            prev_gdn_scores: (8,) array - GDN predictions for previous window
            curr_gdn_scores: (8,) array - GDN predictions for current window
        """
        prev_stats = self.window_stats[prev_window_idx]
        curr_stats = self.window_stats[curr_window_idx]
        
        # Use prediction threshold (0.5) to identify likely anomalous sensors
        prediction_threshold = 0.5
        
        for sensor_name in self.sensor_names:
            prev_stat = prev_stats[sensor_name]
            curr_stat = curr_stats[sensor_name]
            
            # Temporal continuation edge
            self.kg.add_edge(
                f"{sensor_name}@window_{prev_window_idx}",
                f"{sensor_name}@window_{curr_window_idx}",
                edge_type='temporal_continuation',
                sensor=sensor_name,
                source_window=prev_window_idx,
                target_window=curr_window_idx
            )
            
            # Value change tracking
            value_change = curr_stat.mean - prev_stat.mean
            
            if abs(value_change) > 0.1:  # Significant change threshold
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type='value_change',
                    sensor=sensor_name,
                    value_change=float(value_change),
                    source_window=prev_window_idx,
                    target_window=curr_window_idx
                )
            
            # Anomaly propagation tracking (based on GDN predictions, not ground truth)
            prev_faulty = prev_stat.anomaly_score > prediction_threshold
            curr_faulty = curr_stat.anomaly_score > prediction_threshold
            
            if prev_faulty and curr_faulty:
                # Fault persists (according to GDN predictions)
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type='anomaly_propagation',
                    sensor=sensor_name,
                    propagation_type='persists',
                    source_window=prev_window_idx,
                    target_window=curr_window_idx
                )
            elif not prev_faulty and curr_faulty:
                # Fault appears (according to GDN predictions)
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type='anomaly_propagation',
                    sensor=sensor_name,
                    propagation_type='appears',
                    source_window=prev_window_idx,
                    target_window=curr_window_idx
                )
        
        # Track relationship evolution between sensors (correlation changes across windows)
        if prev_window_idx in self.window_graphs and curr_window_idx in self.window_graphs:
            prev_graph = self.window_graphs[prev_window_idx]
            curr_graph = self.window_graphs[curr_window_idx]
            
            for sensor_i in self.sensor_names:
                for sensor_j in self.sensor_names:
                    if sensor_i >= sensor_j:
                        continue
                    
                    # Check if edge exists in both windows
                    if prev_graph.has_edge(sensor_i, sensor_j) and \
                       curr_graph.has_edge(sensor_i, sensor_j):
                        
                        # Get correlations from both windows
                        prev_edge_data = prev_graph[sensor_i][sensor_j]
                        curr_edge_data = curr_graph[sensor_i][sensor_j]
                        
                        # Use correlation_strength if available (new format), otherwise fall back to correlation
                        prev_corr = prev_edge_data.get('correlation_strength', 
                                                       prev_edge_data.get('correlation', 0))
                        curr_corr = curr_edge_data.get('correlation_strength',
                                                       curr_edge_data.get('correlation', 0))
                        
                        # Also get raw correlation values for direction tracking
                        prev_corr_raw = prev_edge_data.get('correlation', 0)
                        curr_corr_raw = curr_edge_data.get('correlation', 0)
                        
                        corr_change = curr_corr - prev_corr
                        corr_change_raw = curr_corr_raw - prev_corr_raw
                        
                        # Track significant changes (threshold: 0.1)
                        if abs(corr_change) > 0.1:
                            # Determine evolution type
                            if abs(curr_corr) > abs(prev_corr):
                                evolution_type = 'strengthening'
                            elif abs(curr_corr) < abs(prev_corr):
                                evolution_type = 'weakening'
                            else:
                                evolution_type = 'stable'
                            
                            self.kg.add_edge(
                                f"{sensor_i}@window_{prev_window_idx}",
                                f"{sensor_j}@window_{curr_window_idx}",
                                edge_type='correlation_evolution',
                                source_sensor=sensor_i,
                                target_sensor=sensor_j,
                                correlation_change=float(corr_change),
                                correlation_change_raw=float(corr_change_raw),
                                prev_correlation=float(prev_corr),
                                prev_correlation_raw=float(prev_corr_raw),
                                curr_correlation=float(curr_corr),
                                curr_correlation_raw=float(curr_corr_raw),
                                evolution_type=evolution_type,
                                source_window=prev_window_idx,
                                target_window=curr_window_idx
                            )
    
    def _compute_variation_from_normal(self, sensor_name: str, 
                                      sensor_values: np.ndarray) -> float:
        """Compute variation from normal operating range"""
        description = SENSOR_DESCRIPTIONS.get(sensor_name, {})
        normal_range = description.get('normal_range', (0, 100))
        
        mean_value = np.mean(sensor_values)
        normal_mean = (normal_range[0] + normal_range[1]) / 2
        normal_span = normal_range[1] - normal_range[0]
        
        if normal_span == 0:
            return 0.0
        
        # Normalized deviation from normal range center
        variation = abs(mean_value - normal_mean) / normal_span
        return float(variation)
    
    def _track_anomaly_propagation(self, gdn_predictions: np.ndarray, threshold: float = 0.5):
        """
        Track how faults propagate across windows using GDN predictions (with threshold).
        
        - Identify root cause sensors (first sensor with GDN score > threshold)
        - Track which sensors become affected in subsequent windows
        - Build fault propagation chains
        
        Only tracks sensors that become faulty for the FIRST TIME after the root sensor.
        
        Args:
            gdn_predictions: (num_windows, 8) array - GDN anomaly scores per sensor per window
            threshold: Threshold for considering a sensor anomalous (default: 0.5)
        """
        num_windows = len(gdn_predictions)
        
        # Find first occurrence of EACH anomalous sensor (based on GDN predictions)
        first_occurrence_all = {}
        for window_idx in range(num_windows):
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                if gdn_predictions[window_idx, sensor_idx] > threshold:
                    if sensor_name not in first_occurrence_all:
                        first_occurrence_all[sensor_name] = window_idx
        
        # Build propagation chains - only for sensors that become faulty FIRST TIME after root
        for root_sensor, root_window in first_occurrence_all.items():
            chain = {
                'root_sensor': root_sensor,
                'root_window': root_window,
                'gdn_score': float(gdn_predictions[root_window, self.sensor_to_idx[root_sensor]]),
                'affected_sensors': [],
                'propagation_timeline': []
            }
            
            # Track sensors that become faulty for FIRST TIME after root sensor
            # Only include sensors whose first occurrence is AFTER root_window
            affected_first_occurrence = {}  # sensor_name -> first_window_after_root
            
            for window_idx in range(root_window + 1, num_windows):  # Start from root_window + 1
                for sensor_idx, sensor_name in enumerate(self.sensor_names):
                    if gdn_predictions[window_idx, sensor_idx] > threshold:
                        # Only track if this sensor's FIRST occurrence is after root_window
                        if sensor_name in first_occurrence_all:
                            sensor_first_window = first_occurrence_all[sensor_name]
                            
                            # Only include if:
                            # 1. Sensor's first occurrence is AFTER root window
                            # 2. We haven't tracked it yet (first time we see it after root)
                            if sensor_first_window > root_window and sensor_name not in affected_first_occurrence:
                                affected_first_occurrence[sensor_name] = sensor_first_window
                                chain['propagation_timeline'].append({
                                    'window': sensor_first_window,
                                    'affected_sensors': [sensor_name],
                                    'gdn_score': float(gdn_predictions[sensor_first_window, sensor_idx])
                                })
            
            # Only add chain if there are affected sensors
            if chain['propagation_timeline']:
                # Sort timeline by window
                chain['propagation_timeline'].sort(key=lambda x: x['window'])
                self.anomaly_propagation_chains.append(chain)
    
    def query_temporal_relationships(self, sensor1: str, sensor2: str,
                                     start_window: int, end_window: int) -> Dict:
        """
        Query how relationship between two sensors evolved across windows.
        
        Returns:
            Dictionary with relationship evolution data
        """
        evolution = {
            'sensor1': sensor1,
            'sensor2': sensor2,
            'windows': [],
            'correlations': [],
            'edge_types': []
        }
        
        for window_idx in range(start_window, min(end_window + 1, len(self.window_graphs))):
            if window_idx in self.window_graphs:
                graph = self.window_graphs[window_idx]
                if graph.has_edge(sensor1, sensor2):
                    edge_data = graph[sensor1][sensor2]
                    evolution['windows'].append(window_idx)
                    evolution['correlations'].append(edge_data.get('correlation', 0))
                    evolution['edge_types'].append(edge_data.get('edge_type', 'unknown'))
        
        return evolution
    
    def get_sensor_descriptions(self) -> Dict[str, Dict]:
        """Return sensor metadata with fault patterns"""
        return SENSOR_DESCRIPTIONS.copy()
    
    def get_fault_documents(self) -> Dict[str, Dict]:
        """Return fault type knowledge base"""
        return FAULT_DOCUMENTS.copy()
    
    def export_for_llm(self, format: str = 'json') -> Dict:
        """
        Export KG in LLM-friendly format with temporal narratives.
        
        Args:
            format: Export format ('json' or 'narrative')
            
        Returns:
            Dictionary with graph structure and narratives
        """
        if format == 'json':
            return self._export_json()
        elif format == 'narrative':
            return self._export_narrative()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_json(self) -> Dict:
        """Export KG as structured JSON"""
        # Convert NetworkX graph to JSON-serializable format
        nodes = []
        for node, data in self.kg.nodes(data=True):
            node_dict = {'id': node, **data}
            nodes.append(node_dict)
        
        edges = []
        for u, v, key, data in self.kg.edges(keys=True, data=True):
            edge_dict = {
                'source': u,
                'target': v,
                'key': key,
                **data
            }
            edges.append(edge_dict)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_windows': len(self.window_graphs),
            'anomaly_propagation_chains': self.anomaly_propagation_chains,
            'sensor_descriptions': self.get_sensor_descriptions(),
            'fault_documents': self.get_fault_documents()
        }
    
    def _export_narrative(self) -> Dict:
        """Export KG as natural language narratives"""
        narratives = []
        
        # Window-by-window narratives
        for window_idx in sorted(self.window_graphs.keys()):
            graph = self.window_graphs[window_idx]
            stats = self.window_stats[window_idx]
            
            narrative = f"Window {window_idx}:\n"
            
            # Describe sensors with high GDN prediction scores (using threshold)
            prediction_threshold = 0.5
            faulty_sensors = [s for s, stat in stats.items() if stat.anomaly_score > prediction_threshold]
            if faulty_sensors:
                narrative += f"  Sensors with high GDN anomaly scores (> {prediction_threshold}): {', '.join(faulty_sensors)}\n"
            
            # Describe relationship violations
            violations = []
            for u, v, data in graph.edges(data=True):
                if data.get('edge_type') == 'violates_expected_relation':
                    violations.append(f"{u} and {v}")
            
            if violations:
                narrative += f"  Relationship violations: {', '.join(violations)}\n"
            
            narratives.append(narrative)
        
        # Anomaly propagation narratives
        propagation_narratives = []
        for chain in self.anomaly_propagation_chains:
            narrative = (
                f"Fault propagation chain:\n"
                f"  Root sensor: {chain['root_sensor']} at window {chain['root_window']}\n"
            )
            
            for timeline_entry in chain['propagation_timeline']:
                narrative += (
                    f"  Window {timeline_entry['window']}: "
                    f"Affected sensors: {', '.join(timeline_entry['affected_sensors'])}\n"
                )
            
            propagation_narratives.append(narrative)
        
        return {
            'window_narratives': narratives,
            'propagation_narratives': propagation_narratives,
            'summary': {
                'total_windows': len(self.window_graphs),
                'total_anomaly_chains': len(self.anomaly_propagation_chains),
                'total_nodes': self.kg.number_of_nodes(),
                'total_edges': self.kg.number_of_edges()
            }
        }


# ============================================================================
# Window Similarity Computation
# ============================================================================

def compute_window_similarity(
    window_embeddings: Dict[int, Dict[str, Any]],
    k: int = 5
) -> List[Tuple[int, int, float, float]]:
    """
    Compute window-to-window similarity based on embeddings.
    
    For each window, finds k most similar windows (excluding self) using
    cosine similarity and euclidean distance in embedding space.
    
    Args:
        window_embeddings: Dictionary mapping window_idx -> {
            'embedding': np.ndarray (hidden_dim,),
            'dist_normal': float,
            'dist_anomalous': float,
            'confidence': float
        }
        k: Number of nearest neighbors to find per window
    
    Returns:
        List of tuples: (window_i, window_j, cosine_similarity, euclidean_distance)
        Sorted by window_i, then by similarity descending
    """
    if len(window_embeddings) == 0:
        return []
    
    # Extract embeddings and window indices
    window_indices = sorted(window_embeddings.keys())
    embeddings_list = [window_embeddings[idx]['embedding'] for idx in window_indices]
    embeddings_array = np.array(embeddings_list)  # (N, hidden_dim)
    
    N = len(window_indices)
    if N < 2:
        return []
    
    # Compute cosine similarity matrix (memory efficient: compute per window)
    similarity_edges = []
    
    for i, window_i in enumerate(window_indices):
        embedding_i = embeddings_array[i:i+1]  # (1, hidden_dim)
        
        # Compute cosine similarity with all other windows
        # Use sklearn's cosine_similarity for efficiency
        similarities = cosine_similarity(embedding_i, embeddings_array)[0]  # (N,)
        
        # Compute euclidean distances
        distances = cdist(embedding_i, embeddings_array, metric='euclidean')[0]  # (N,)
        
        # Find top-k most similar (excluding self)
        # Set self-similarity to -inf to exclude it
        similarities[i] = -np.inf
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Add edges
        for j in top_k_indices:
            if j != i:  # Exclude self
                window_j = window_indices[j]
                cosine_sim = float(similarities[j])
                euclidean_dist = float(distances[j])
                similarity_edges.append((window_i, window_j, cosine_sim, euclidean_dist))
    
    return similarity_edges
