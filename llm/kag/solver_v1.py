"""
KAG Solver v1 - Deterministic Multi-Step Reasoning

Implements heuristic-based Knowledge-Augmented Generation (KAG) reasoning
without LLM planning. Performs deterministic graph traversal and multi-step
reasoning to identify root cause sensors and affected sensors.

This is a baseline solver for comparison with LLM-planned KAG (Week 4).
"""

from llm.kag.neo4j_queries import Neo4jKAGQueries
from typing import Dict, List, Optional
import numpy as np


class KAGSolverV1:
    """Deterministic KAG solver - multi-step reasoning without LLM planning."""
    
    def __init__(self, neo4j_queries: Neo4jKAGQueries, sensor_names: List[str], 
                 anomaly_threshold: float = 0.7, min_anomalous: int = 1, 
                 confidence_min: float = 0.4):
        """
        Initialize KAG Solver v1.
        
        Args:
            neo4j_queries: Neo4jKAGQueries instance for graph queries
            sensor_names: List of sensor names in order (for label conversion)
            anomaly_threshold: Threshold for anomaly detection (default: 0.7)
                              Sensors with anomaly_score > threshold are considered anomalous
            min_anomalous: Minimum number of anomalous sensors required (default: 1)
            confidence_min: Minimum confidence threshold to accept a fault (default: 0.4)
        """
        self.queries = neo4j_queries
        self.sensor_names = sensor_names
        self.sensor_to_idx = {name: idx for idx, name in enumerate(sensor_names)}
        self.anomaly_threshold = anomaly_threshold
        self.min_anomalous = min_anomalous
        self.confidence_min = confidence_min
        self._anomaly_score_cache = {}  # Cache for anomaly scores: (window_idx, sensor) -> score
    
    def solve(self, window_idx: int) -> Dict:
        """
        Multi-step reasoning to find root cause and affected sensors.
        
        Performs deterministic reasoning steps:
        1. Get anomalous sensors (threshold=self.anomaly_threshold, default=0.3)
        2. Get correlation violations
        3. Compute centrality to find most connected sensor
        4. Deduce root cause (heuristic: anomaly_score * degree_centrality)
        5. Find affected sensors via correlations
        6. Classify fault type based on root cause + affected sensors
        
        Args:
            window_idx: Window index to analyze
            
        Returns:
            Dictionary with keys:
            - 'root_cause_sensor': str or None (base sensor name)
            - 'affected_sensors': List[str] (base sensor names)
            - 'fault_type': str
            - 'reasoning_trace': List[Dict] (step-by-step reasoning)
            - 'confidence': float (0.0-1.0)
            - 'sensor_labels': np.ndarray (binary array, shape=(num_sensors,))
            - 'window_label': int (0 or 1-8, sensor-indexed)
        """
        reasoning_trace = []
        
        # Step 1: Get anomalous sensors
        anomalous = self.queries.get_anomalous_sensors(window_idx, threshold=self.anomaly_threshold)
        reasoning_trace.append({
            'step': 1,
            'operation': 'get_anomalous_sensors',
            'result': f"Found {len(anomalous)} anomalous sensors",
            'data': [s['sensor'] for s in anomalous]
        })
        
        # Check minimum anomalous sensors requirement
        if not anomalous or len(anomalous) < self.min_anomalous:
            reasoning_trace.append({
                'step': 1.5,
                'operation': 'min_anomalous_check',
                'result': f"Failed: Found {len(anomalous)} anomalous sensors, need at least {self.min_anomalous}"
            })
            return self._empty_result(reasoning_trace)
        
        # Step 2: Get violations
        violations = self.queries.get_violations(window_idx, deviation_threshold=0.3)
        reasoning_trace.append({
            'step': 2,
            'operation': 'get_violations',
            'result': f"Found {len(violations)} correlation violations"
        })
        
        # Step 3: Compute centrality to find most connected sensor
        centrality = self.queries.compute_sensor_centrality(window_idx)
        reasoning_trace.append({
            'step': 3,
            'operation': 'compute_centrality',
            'result': f"Top sensor by connections: {centrality[0]['sensor']}" if centrality else "None"
        })
        
        # Step 4: Deduce root cause
        # Heuristic: sensor with highest (anomaly_score * onset_score * degree_centrality)
        root_cause = self._deduce_root_cause(anomalous, centrality, window_idx)
        reasoning_trace.append({
            'step': 4,
            'operation': 'deduce_root_cause',
            'result': f"Root cause: {root_cause}"
        })
        
        # Step 5: Find affected sensors via correlations
        affected = self._find_affected_sensors(root_cause, window_idx, anomalous)
        reasoning_trace.append({
            'step': 5,
            'operation': 'find_affected_sensors',
            'result': f"Affected: {affected}"
        })
        
        # Step 6: Map to fault type
        fault_type = self._classify_fault_type(root_cause, affected, violations)
        
        # Step 7: Compute confidence and check minimum threshold
        confidence = self._compute_confidence(anomalous, violations, centrality)
        reasoning_trace.append({
            'step': 7,
            'operation': 'compute_confidence',
            'result': f"Confidence: {confidence:.3f} (min required: {self.confidence_min})"
        })
        
        # Check confidence threshold
        if confidence < self.confidence_min:
            reasoning_trace.append({
                'step': 7.5,
                'operation': 'confidence_check',
                'result': f"Failed: Confidence {confidence:.3f} below minimum {self.confidence_min}"
            })
            return self._empty_result(reasoning_trace)
        
        return {
            'root_cause_sensor': root_cause,
            'affected_sensors': affected,
            'fault_type': fault_type,
            'reasoning_trace': reasoning_trace,
            'confidence': confidence,
            'sensor_labels': self._to_sensor_labels(root_cause, affected),
            'window_label': self._to_window_label(root_cause, affected)
        }
    
    def _get_anomaly_score(self, sensor: str, window_idx: int) -> float:
        """
        Get anomaly score for a sensor in a specific window.
        Uses cache to avoid repeated Neo4j queries.
        
        Args:
            sensor: Base sensor name
            window_idx: Window index
            
        Returns:
            Anomaly score (0.0-1.0)
        """
        cache_key = (window_idx, sensor)
        if cache_key in self._anomaly_score_cache:
            return self._anomaly_score_cache[cache_key]
        
        # Query Neo4j for anomaly score
        with self.queries.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {base_sensor_name: $sensor, window: $window_idx})
                RETURN s.anomaly_score AS score
                LIMIT 1
            """, sensor=sensor, window_idx=window_idx)
            record = result.single()
            score = float(record['score']) if record else 0.0
        
        self._anomaly_score_cache[cache_key] = score
        return score
    
    def _find_anomaly_onset(self, sensor: str, window_idx: int, lookback: int = 3) -> int:
        """
        Find the earliest window where sensor became anomalous.
        
        Searches backwards from window_idx up to lookback windows to find
        when the sensor first exceeded the anomaly threshold.
        
        Args:
            sensor: Base sensor name
            window_idx: Current window index
            lookback: Maximum number of windows to look back (default: 3)
            
        Returns:
            Earliest window index where anomaly was detected, or window_idx if not found earlier
        """
        earliest = window_idx
        start_window = max(0, window_idx - lookback)
        
        for w in range(start_window, window_idx + 1):
            score = self._get_anomaly_score(sensor, w)
            if score > self.anomaly_threshold:
                earliest = w
                break
        
        return earliest
    
    def _deduce_root_cause(self, anomalous: List[Dict], 
                           centrality: List[Dict], window_idx: int) -> Optional[str]:
        """
        Find sensor with highest score considering anomaly, temporal onset, and centrality.
        
        Scoring formula: anomaly_score * onset_score * degree_centrality
        where onset_score = (window_idx - onset + 1) rewards earlier onset.
        
        Args:
            anomalous: List of anomalous sensor dicts with 'sensor' and 'score'
            centrality: List of centrality dicts with 'sensor' and 'degree'
            window_idx: Current window index (for temporal analysis)
            
        Returns:
            Base sensor name of root cause, or None if no candidates
        """
        if not anomalous:
            return None
        
        # Build centrality lookup
        cent_map = {c['sensor']: c['degree'] for c in centrality}
        
        # Score each anomalous sensor with temporal onset
        scores = []
        for sensor_data in anomalous:
            sensor = sensor_data['sensor']
            anomaly_score = sensor_data['score']
            
            # Find when anomaly started
            onset = self._find_anomaly_onset(sensor, window_idx)
            onset_score = float(window_idx - onset + 1)  # Earlier onset -> larger score
            
            # Get centrality
            degree = max(cent_map.get(sensor, 1), 1)  # Ensure at least 1
            
            # Combined score
            score = anomaly_score * onset_score * degree
            scores.append((sensor, score))
        
        # Return highest scoring sensor
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else None
    
    def _find_affected_sensors(self, root_cause: str, window_idx: int,
                               anomalous: List[Dict]) -> List[str]:
        """
        Find sensors affected by root cause via correlations.
        
        Args:
            root_cause: Base sensor name of root cause
            window_idx: Window index
            anomalous: List of anomalous sensors (for filtering)
            
        Returns:
            List of base sensor names that are affected
        """
        if not root_cause:
            return []
        
        affected = []
        anomalous_names = {s['sensor'] for s in anomalous}
        
        # Get direct correlations from root
        neighbors = self.queries.get_correlated_neighbors(root_cause, window_idx, corr_threshold=0.5)
        
        for n in neighbors:
            neighbor = n['neighbor']
            # Only count as affected if also anomalous
            if neighbor in anomalous_names and neighbor != root_cause:
                affected.append(neighbor)
        
        return affected
    
    def _classify_fault_type(self, root_cause: str, affected: List[str],
                            violations: List[Dict]) -> str:
        """
        Map root cause + affected sensors to fault type.
        
        Uses exact fault types from codebase:
        - VSS_DROPOUT for VEHICLE_SPEED faults
        - COOLANT_DROPOUT for COOLANT_TEMPERATURE faults
        - TPS_STUCK for THROTTLE faults
        - MAF_SCALE_LOW for INTAKE_MANIFOLD_PRESSURE faults
        - RPM_SPEED_DECOUPLE for RPM+SPEED decoupling
        - gradual_drift as default
        
        Args:
            root_cause: Base sensor name of root cause
            affected: List of affected base sensor names
            violations: List of violation dicts (unused but kept for API consistency)
            
        Returns:
            Fault type string
        """
        if not root_cause:
            return None
        
        # Check for RPM_SPEED_DECOUPLE (both sensors affected)
        all_faulty = [root_cause] + affected
        has_rpm = any('ENGINE_RPM' in s for s in all_faulty)
        has_speed = any('VEHICLE_SPEED' in s for s in all_faulty)
        
        if has_rpm and has_speed:
            return 'RPM_SPEED_DECOUPLE'
        
        # Check root cause sensor
        if 'VEHICLE_SPEED' in root_cause:
            return 'VSS_DROPOUT'
        elif 'COOLANT_TEMPERATURE' in root_cause:
            return 'COOLANT_DROPOUT'
        elif 'THROTTLE' in root_cause:
            return 'TPS_STUCK'
        elif 'INTAKE_MANIFOLD_PRESSURE' in root_cause:
            return 'MAF_SCALE_LOW'
        else:
            return 'gradual_drift'
    
    def _to_sensor_labels(self, root_cause: Optional[str], affected: List[str]) -> np.ndarray:
        """
        Convert root cause + affected sensors to binary sensor label array.
        
        Args:
            root_cause: Base sensor name of root cause (or None)
            affected: List of affected base sensor names
            
        Returns:
            Binary array of shape (num_sensors,) where 1 indicates faulty sensor
        """
        labels = np.zeros(len(self.sensor_names), dtype=int)
        
        all_faulty = []
        if root_cause:
            all_faulty.append(root_cause)
        all_faulty.extend(affected)
        
        for sensor_name in all_faulty:
            if sensor_name in self.sensor_to_idx:
                idx = self.sensor_to_idx[sensor_name]
                labels[idx] = 1
        
        return labels
    
    def _to_window_label(self, root_cause: Optional[str], affected: List[str]) -> int:
        """
        Convert root cause to window label (sensor-indexed: 0-8).
        
        Window labels are sensor-indexed:
        - 0 = no fault
        - 1-8 = sensor index (1-indexed: sensor 0 -> label 1, sensor 7 -> label 8)
        
        Args:
            root_cause: Base sensor name of root cause (or None)
            affected: List of affected sensors (unused, root cause determines label)
            
        Returns:
            Window label (0-8)
        """
        if not root_cause:
            return 0
        
        if root_cause in self.sensor_to_idx:
            sensor_idx = self.sensor_to_idx[root_cause]
            return sensor_idx + 1  # 1-indexed
        
        return 0
    
    def _violation_severity(self, violations: List[Dict]) -> float:
        """
        Compute violation severity score based on deviation magnitude.
        
        Counts violations with deviation > 0.5 as "severe" and returns
        a normalized score capped at 1.0.
        
        Args:
            violations: List of violation dicts with 'deviation' key
            
        Returns:
            Severity score (0.0-1.0)
        """
        if not violations:
            return 0.0
        
        severe = [v for v in violations if v.get('deviation', 0.0) > 0.5]
        return min(len(severe) / 3.0, 1.0)
    
    def _compute_confidence(self, anomalous: List[Dict], violations: List[Dict], 
                           centrality: List[Dict]) -> float:
        """
        Confidence heuristic based on evidence strength with violation severity weighting.
        
        Args:
            anomalous: List of anomalous sensors
            violations: List of violations
            centrality: List of centrality results
            
        Returns:
            Confidence score (0.0-1.0)
        """
        score = 0.0
        
        # Anomalous sensors found
        if anomalous:
            score += 0.3
        
        # Violation severity (weighted by deviation magnitude)
        score += 0.4 * self._violation_severity(violations)
        
        # High centrality (well-connected sensor)
        if centrality and centrality[0]['degree'] > 3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _empty_result(self, trace: List[Dict]) -> Dict:
        """
        Return empty result when no anomalies found.
        
        Args:
            trace: Reasoning trace so far
            
        Returns:
            Empty result dictionary
        """
        return {
            'root_cause_sensor': None,
            'affected_sensors': [],
            'fault_type': None,  # No fault type when no fault detected
            'reasoning_trace': trace,
            'confidence': 0.0,
            'sensor_labels': np.zeros(len(self.sensor_names), dtype=int),
            'window_label': 0
        }
