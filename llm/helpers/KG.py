"""
Knowledge Graph Helper Class for constructing semantic knowledge graphs from GDN outputs.

This module provides temporal traversal of windows from gdn.ipynb, building relationships
between sensors across time and tracking anomaly propagation for automotive diagnostics.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


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
    """
    
    def __init__(self, sensor_names: List[str], sensor_embeddings: np.ndarray, 
                 adjacency_matrix: np.ndarray):
        """
        Initialize the Knowledge Graph Builder.
        
        Args:
            sensor_names: List of sensor names (must match order in data)
            sensor_embeddings: Learned sensor embeddings from GDN (num_sensors, embed_dim)
            adjacency_matrix: Adjacency matrix from GDN (num_sensors, num_sensors)
        """
        self.sensor_names = sensor_names
        self.sensor_embeddings = sensor_embeddings
        self.adjacency_matrix = adjacency_matrix
        self.num_sensors = len(sensor_names)
        
        # Create sensor name to index mapping
        self.sensor_to_idx = {name: idx for idx, name in enumerate(sensor_names)}
        
        # Initialize graph structures
        self.kg = nx.MultiDiGraph()  # Main knowledge graph
        self.window_graphs = {}  # Per-window graphs
        self.window_stats = {}  # Per-window statistics
        self.temporal_edges = []  # Temporal edges between windows
        self.anomaly_propagation_chains = []  # Fault propagation chains
        
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
    
    def build_from_gdn_windows(self, X_windows: np.ndarray, sensor_labels: np.ndarray,
                                window_labels: np.ndarray) -> nx.MultiDiGraph:
        """
        Main entry point: Build KG by traversing windows temporally.
        
        Args:
            X_windows: (num_windows, 300, 8) array - sensor data windows
            sensor_labels: (num_windows, 8) array - binary fault labels per sensor
            window_labels: (num_windows,) array - window-level fault labels
            
        Returns:
            Knowledge graph with temporal traversal
        """
        num_windows = len(X_windows)
        
        # Traverse windows sequentially
        for window_idx in range(num_windows):
            window_data = X_windows[window_idx]  # (300, 8)
            window_sensor_labels = sensor_labels[window_idx]  # (8,)
            is_faulty_window = window_labels[window_idx] > 0
            
            # Process this window
            self._process_window(window_idx, window_data, window_sensor_labels, is_faulty_window)
            
            # Build temporal edges to previous window
            if window_idx > 0:
                self._build_temporal_edges(window_idx - 1, window_idx, 
                                         X_windows[window_idx - 1], window_data,
                                         sensor_labels[window_idx - 1], window_sensor_labels)
        
        # Track anomaly propagation after processing all windows
        self._track_anomaly_propagation(sensor_labels)
        
        return self.kg
    
    def _process_window(self, window_idx: int, window_data: np.ndarray,
                       sensor_labels: np.ndarray, is_faulty: bool):
        """
        Process a single window:
        - Compute within-window sensor correlations
        - Compare to expected correlations (from adjacency_matrix)
        - Detect relationship violations
        - Add nodes and edges for this window
        """
        # Compute statistical summaries for each sensor in this window
        window_stats = {}
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            sensor_values = window_data[:, sensor_idx]
            
            stats = WindowStats(
                mean=float(np.mean(sensor_values)),
                std=float(np.std(sensor_values)),
                min=float(np.min(sensor_values)),
                max=float(np.max(sensor_values)),
                variation_from_normal=self._compute_variation_from_normal(
                    sensor_name, sensor_values
                ),
                anomaly_score=float(sensor_labels[sensor_idx])
            )
            window_stats[sensor_name] = stats
        
        self.window_stats[window_idx] = window_stats
        
        # Compute within-window correlation matrix
        correlation_matrix = np.corrcoef(window_data.T)  # (8, 8)
        
        # Build per-window graph
        window_graph = nx.Graph()
        
        # Add sensor nodes with window-specific attributes
        for sensor_name, stats in window_stats.items():
            window_graph.add_node(
                sensor_name,
                window_idx=window_idx,
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                variation_from_normal=stats.variation_from_normal,
                is_faulty=bool(stats.anomaly_score > 0)
            )
        
        # Add edges based on correlations and adjacency matrix
        for i, sensor_i in enumerate(self.sensor_names):
            for j, sensor_j in enumerate(self.sensor_names):
                if i >= j:
                    continue
                
                # Get correlation from window data
                window_corr = correlation_matrix[i, j]
                
                # Get expected correlation from GDN adjacency matrix
                expected_corr = self.adjacency_matrix[i, j]
                
                # Infer semantic edge type
                edge_type, edge_attrs = self._infer_semantic_edge(
                    sensor_i, sensor_j, window_corr, expected_corr, 
                    window_stats[sensor_i], window_stats[sensor_j]
                )
                
                if edge_type:
                    # Ensure edge_type is in edge_attrs, not duplicated
                    edge_data = {
                        'window_idx': window_idx,
                        'edge_type': edge_type,
                        'correlation': float(window_corr),
                        'expected_correlation': float(expected_corr),
                        'correlation_deviation': float(abs(window_corr - expected_corr)),
                        **edge_attrs
                    }
                    # Remove edge_type from edge_attrs if it's already there
                    if 'edge_type' in edge_attrs:
                        edge_data.pop('edge_type', None)
                        edge_data['edge_type'] = edge_type  # Use the one we want
                    
                    window_graph.add_edge(sensor_i, sensor_j, **edge_data)
                    
                    # Add to main KG with temporal context
                    self.kg.add_edge(sensor_i, sensor_j, **edge_data)
        
        self.window_graphs[window_idx] = window_graph
    
    def _infer_semantic_edge(self, sensor_i: str, sensor_j: str, 
                            window_corr: float, expected_corr: float,
                            stats_i: WindowStats, stats_j: WindowStats) -> Tuple[Optional[str], Dict]:
        """
        Convert adjacency weights to semantic edge types for a window.
        
        Returns:
            (edge_type, edge_attributes) or (None, {}) if no edge should be created
        """
        # Check if this pair has expected correlation
        expected_rel = EXPECTED_CORRELATIONS.get((sensor_i, sensor_j)) or \
                       EXPECTED_CORRELATIONS.get((sensor_j, sensor_i))
        
        # Threshold for considering correlation significant
        corr_threshold = 0.3
        deviation_threshold = 0.2
        
        # Check if correlation is significant
        if abs(window_corr) < corr_threshold:
            return None, {}
        
        edge_attrs = {}
        
        # Check for relationship violation
        if expected_rel:
            expected_type = expected_rel['type']
            corr_deviation = abs(window_corr - expected_corr)
            
            if corr_deviation > deviation_threshold:
                # Relationship violated
                edge_type = 'violates_expected_relation'
                edge_attrs['expected_type'] = expected_type
                edge_attrs['violation_strength'] = float(corr_deviation)
            else:
                # Relationship maintained
                edge_type = expected_type
                edge_attrs['expected_type'] = expected_type
        else:
            # General correlation
            if window_corr > 0:
                edge_type = 'correlates_with'
            else:
                edge_type = 'correlates_with'  # Negative correlation still counts
        
        # Check if edge supports or contradicts fault
        if stats_i.anomaly_score > 0 or stats_j.anomaly_score > 0:
            # Check if correlation pattern supports fault hypothesis
            if expected_rel and abs(window_corr - expected_corr) > deviation_threshold:
                edge_attrs['supports_fault'] = True
            else:
                edge_attrs['supports_fault'] = False
        
        edge_attrs['edge_type'] = edge_type
        return edge_type, edge_attrs
    
    def _build_temporal_edges(self, prev_window_idx: int, curr_window_idx: int,
                             prev_window_data: np.ndarray, curr_window_data: np.ndarray,
                             prev_sensor_labels: np.ndarray, curr_sensor_labels: np.ndarray):
        """
        Build temporal edges connecting consecutive windows:
        - Connect same sensor across windows
        - Track value changes
        - Track relationship evolution
        - Track anomaly propagation
        """
        prev_stats = self.window_stats[prev_window_idx]
        curr_stats = self.window_stats[curr_window_idx]
        
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
            
            # Anomaly propagation tracking
            prev_faulty = prev_stat.anomaly_score > 0
            curr_faulty = curr_stat.anomaly_score > 0
            
            if prev_faulty and curr_faulty:
                # Fault persists
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
                # Fault appears
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type='anomaly_propagation',
                    sensor=sensor_name,
                    propagation_type='appears',
                    source_window=prev_window_idx,
                    target_window=curr_window_idx
                )
        
        # Track relationship evolution between sensors
        if prev_window_idx in self.window_graphs and curr_window_idx in self.window_graphs:
            prev_graph = self.window_graphs[prev_window_idx]
            curr_graph = self.window_graphs[curr_window_idx]
            
            for sensor_i in self.sensor_names:
                for sensor_j in self.sensor_names:
                    if sensor_i >= sensor_j:
                        continue
                    
                    if prev_graph.has_edge(sensor_i, sensor_j) and \
                       curr_graph.has_edge(sensor_i, sensor_j):
                        
                        prev_corr = prev_graph[sensor_i][sensor_j].get('correlation', 0)
                        curr_corr = curr_graph[sensor_i][sensor_j].get('correlation', 0)
                        corr_change = curr_corr - prev_corr
                        
                        if abs(corr_change) > 0.1:  # Significant change
                            self.kg.add_edge(
                                f"{sensor_i}@window_{prev_window_idx}",
                                f"{sensor_j}@window_{curr_window_idx}",
                                edge_type='relationship_evolution',
                                source_sensor=sensor_i,
                                target_sensor=sensor_j,
                                correlation_change=float(corr_change),
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
    
    def _track_anomaly_propagation(self, sensor_labels: np.ndarray):
        """
        Track how faults propagate across windows:
        - Identify root cause sensors
        - Track which sensors become affected in subsequent windows
        - Build fault propagation chains
        """
        num_windows = len(sensor_labels)
        
        # Find first occurrence of each faulty sensor
        first_occurrence = {}
        for window_idx in range(num_windows):
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                if sensor_labels[window_idx, sensor_idx] > 0:
                    if sensor_name not in first_occurrence:
                        first_occurrence[sensor_name] = window_idx
        
        # Build propagation chains
        for root_sensor, root_window in first_occurrence.items():
            chain = {
                'root_sensor': root_sensor,
                'root_window': root_window,
                'affected_sensors': [],
                'propagation_timeline': []
            }
            
            # Track which sensors become faulty after root sensor
            for window_idx in range(root_window, num_windows):
                affected_in_window = []
                for sensor_idx, sensor_name in enumerate(self.sensor_names):
                    if sensor_labels[window_idx, sensor_idx] > 0:
                        affected_in_window.append(sensor_name)
                
                if affected_in_window:
                    chain['propagation_timeline'].append({
                        'window': window_idx,
                        'affected_sensors': affected_in_window
                    })
            
            if chain['propagation_timeline']:
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
            
            # Describe faulty sensors
            faulty_sensors = [s for s, stat in stats.items() if stat.anomaly_score > 0]
            if faulty_sensors:
                narrative += f"  Faulty sensors: {', '.join(faulty_sensors)}\n"
            
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
