"""
Neo4j Query Wrappers for KAG Reasoning Primitives

Provides reusable query functions that wrap Neo4j graph operations for
Knowledge-Augmented Generation (KAG) reasoning over sensor knowledge graphs.

All queries use base_sensor_name for filtering and return base sensor names
for consistency with the knowledge graph schema.
"""

from typing import List, Dict, Optional
from neo4j import GraphDatabase


class Neo4jKAGQueries:
    """KAG reasoning primitives over sensor knowledge graph."""
    
    def __init__(self, uri: str, user: str, password: str, max_retries: int = 3):
        """
        Initialize Neo4j connection with retry logic.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://127.0.0.1:7687")
            user: Neo4j username
            password: Neo4j password
            max_retries: Maximum number of connection retry attempts
        """
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    uri, 
                    auth=(user, password),
                    max_connection_lifetime=3600,
                    connection_timeout=30.0
                )
                # Test the connection
                with self.driver.session() as session:
                    session.run("RETURN 1").consume()
                print(f"✓ Connected to Neo4j at {uri}")
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"  ⚠️  Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"  Retrying in 2 seconds...")
                    time.sleep(2.0)
                else:
                    raise ConnectionError(
                        f"Failed to connect to Neo4j after {max_retries} attempts. "
                        f"URI: {uri}, User: {user}. "
                        f"Last error: {last_error}. "
                        f"Please ensure Neo4j Desktop is running and the database is started."
                    ) from last_error
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_all_anomaly_scores(self, window_idx: int) -> List[Dict]:
        """
        Get all anomaly scores for a window (for auto-threshold detection).
        
        Args:
            window_idx: Window index to query
            
        Returns:
            List of dictionaries with keys: 'sensor', 'score', 'subsystem'
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {window: $window_idx})
                RETURN s.base_sensor_name AS sensor, 
                       s.anomaly_score AS score,
                       s.subsystem AS subsystem
                ORDER BY s.anomaly_score DESC
            """, window_idx=window_idx)
            return [dict(record) for record in result]
    
    def get_anomalous_sensors(self, window_idx: int, threshold: float = 0.5) -> List[Dict]:
        """
        Retrieval: Find sensors with anomaly_score > threshold in a window.
        
        Args:
            window_idx: Window index to query
            threshold: Anomaly score threshold (default: 0.5)
            
        Returns:
            List of dictionaries with keys:
            - 'sensor': base sensor name
            - 'score': anomaly_score
            - 'subsystem': sensor subsystem
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {window: $window_idx})-[:BELONGS_TO]->(w:Window {label: $window_idx})
                WHERE s.anomaly_score > $threshold
                RETURN s.base_sensor_name AS sensor, 
                       s.anomaly_score AS score,
                       s.subsystem AS subsystem
                ORDER BY s.anomaly_score DESC
            """, window_idx=window_idx, threshold=threshold)
            return [dict(record) for record in result]
    
    def get_all_deviations(self, window_idx: int) -> List[Dict]:
        """
        Get all correlation deviations for a window (for auto-threshold detection).
        
        Args:
            window_idx: Window index to query
            
        Returns:
            List of dictionaries with keys: 'source', 'target', 'actual', 'expected', 'deviation'
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Sensor {window: $window_idx})-[r:CORRELATES_WITH]->(s2:Sensor {window: $window_idx})
                WHERE exists(r.actual_correlation) AND exists(r.expected_correlation)
                RETURN s1.base_sensor_name AS source,
                       s2.base_sensor_name AS target,
                       r.actual_correlation AS actual,
                       r.expected_correlation AS expected,
                       abs(r.actual_correlation - r.expected_correlation) AS deviation
                ORDER BY deviation DESC
            """, window_idx=window_idx)
            return [dict(record) for record in result]
    
    def get_violations(self, window_idx: int, deviation_threshold: float = 0.3) -> List[Dict]:
        """
        Retrieval: Find relationship violations in this window.
        
        Finds CORRELATES_WITH relationships where the deviation between
        actual and expected correlation exceeds the threshold.
        This is the STRONGEST fault signal - violations indicate broken sensor relationships.
        
        Args:
            window_idx: Window index to query
            deviation_threshold: Minimum deviation to consider a violation (default: 0.3)
            
        Returns:
            List of dictionaries with keys:
            - 'source': base sensor name
            - 'target': base sensor name
            - 'actual': actual_correlation value
            - 'expected': expected_correlation value
            - 'deviation': absolute deviation (same as deviation_from_gdn)
            - 'source_anomaly': anomaly_score of source sensor
            - 'target_anomaly': anomaly_score of target sensor
            - 'violates_gdn_expectation': True if deviation > threshold
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Sensor {window: $window_idx})-[r:CORRELATES_WITH]->(s2:Sensor {window: $window_idx})
                WHERE r.deviation > $threshold
                RETURN s1.base_sensor_name AS source,
                       s2.base_sensor_name AS target,
                       r.actual_correlation AS actual,
                       r.expected_correlation AS expected,
                       r.deviation AS deviation,
                       s1.anomaly_score AS source_anomaly,
                       s2.anomaly_score AS target_anomaly,
                       (r.deviation > $threshold) AS violates_gdn_expectation
                ORDER BY deviation DESC
            """, window_idx=window_idx, threshold=deviation_threshold)
            return [dict(record) for record in result]
    
    def get_sensors_with_violations_and_anomaly(
        self, window_idx: int, anomaly_threshold: float = 0.5, min_violations: int = 1
    ) -> List[Dict]:
        """
        Retrieval: Find sensors with high anomaly scores AND many violated correlations.
        
        This combines two strong signals:
        1. High GDN anomaly score (indicates likely fault)
        2. Multiple correlation violations (indicates broken relationships)
        
        Args:
            window_idx: Window index to query
            anomaly_threshold: Minimum anomaly score (default: 0.5)
            min_violations: Minimum number of violations (default: 1)
            
        Returns:
            List of dictionaries with keys:
            - 'sensor': base sensor name
            - 'anomaly_score': GDN anomaly score
            - 'violation_count': Number of violated correlations
            - 'max_deviation': Maximum deviation among violations
            - 'avg_deviation': Average deviation among violations
            - 'subsystem': Sensor subsystem
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {window: $window_idx})
                WHERE s.anomaly_score > $anomaly_threshold
                OPTIONAL MATCH (s)-[r:CORRELATES_WITH]->(other:Sensor {window: $window_idx})
                WHERE r.deviation > 0.3
                WITH s, count(r) AS violation_count,
                     max(r.deviation) AS max_deviation,
                     avg(r.deviation) AS avg_deviation
                WHERE violation_count >= $min_violations
                RETURN s.base_sensor_name AS sensor,
                       s.anomaly_score AS anomaly_score,
                       violation_count,
                       max_deviation,
                       avg_deviation,
                       s.subsystem AS subsystem
                ORDER BY violation_count DESC, anomaly_score DESC
            """, window_idx=window_idx, anomaly_threshold=anomaly_threshold, min_violations=min_violations)
            return [dict(record) for record in result]
    
    def get_correlated_neighbors(self, sensor: str, window_idx: int, 
                                 corr_threshold: float = 0.5) -> List[Dict]:
        """
        Retrieval: Find sensors correlated with given sensor in a window.
        
        Checks both directions of CORRELATES_WITH relationships (since edges
        are stored in one direction but correlations are symmetric).
        
        Args:
            sensor: Base sensor name (e.g., "VEHICLE_SPEED ()")
            window_idx: Window index to query
            corr_threshold: Minimum absolute correlation to include (default: 0.5)
            
        Returns:
            List of dictionaries with keys:
            - 'neighbor': base sensor name of correlated sensor
            - 'correlation': actual_correlation value
            - 'neighbor_anomaly': anomaly_score of the neighbor sensor
        """
        with self.driver.session() as session:
            # Check both outgoing and incoming edges using UNION
            result = session.run("""
                MATCH (s1:Sensor {base_sensor_name: $sensor, window: $window_idx})
                MATCH (s1)-[r:CORRELATES_WITH]->(s2:Sensor {window: $window_idx})
                WHERE abs(r.actual_correlation) > $threshold
                RETURN s2.base_sensor_name AS neighbor,
                       r.actual_correlation AS correlation,
                       s2.anomaly_score AS neighbor_anomaly
                UNION
                MATCH (s1:Sensor {base_sensor_name: $sensor, window: $window_idx})
                MATCH (s3:Sensor {window: $window_idx})-[r:CORRELATES_WITH]->(s1)
                WHERE abs(r.actual_correlation) > $threshold
                RETURN s3.base_sensor_name AS neighbor,
                       r.actual_correlation AS correlation,
                       s3.anomaly_score AS neighbor_anomaly
                ORDER BY abs(correlation) DESC
            """, sensor=sensor, window_idx=window_idx, threshold=corr_threshold)
            return [dict(record) for record in result]
    
    def compute_sensor_centrality(self, window_idx: int) -> List[Dict]:
        """
        Math: Rank sensors by degree centrality (connection count).
        
        Computes the number of CORRELATES_WITH relationships for each sensor
        in the specified window.
        
        Args:
            window_idx: Window index to query
            
        Returns:
            List of dictionaries with keys:
            - 'sensor': base sensor name
            - 'degree': number of CORRELATES_WITH relationships
            - 'score': anomaly_score of the sensor
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {window: $window_idx})
                OPTIONAL MATCH (s)-[r:CORRELATES_WITH]-(other:Sensor {window: $window_idx})
                WITH s.base_sensor_name AS sensor, 
                     count(r) AS degree, 
                     s.anomaly_score AS score
                RETURN sensor, degree, score
                ORDER BY degree DESC, score DESC
            """, window_idx=window_idx)
            return [dict(record) for record in result]
    
    def find_propagation_path(self, root_sensor: str, target_sensor: str, 
                             window_idx: int) -> Dict:
        """
        Find shortest path between sensors (fault propagation analysis).
        
        Uses CORRELATES_WITH relationships to find the shortest path
        between two sensors in the same window.
        
        Args:
            root_sensor: Base sensor name of root sensor
            target_sensor: Base sensor name of target sensor
            window_idx: Window index to query
            
        Returns:
            Dictionary with keys:
            - 'path_nodes': List of base sensor names along the path
            - 'hops': Number of hops (relationship steps) in the path
            Returns empty path if no path exists.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Sensor {base_sensor_name: $root, window: $window_idx})
                MATCH (s2:Sensor {base_sensor_name: $target, window: $window_idx})
                MATCH path = shortestPath((s1)-[:CORRELATES_WITH*]-(s2))
                WHERE ALL(n IN nodes(path) WHERE n.window = $window_idx)
                RETURN [n IN nodes(path) | n.base_sensor_name] AS path_nodes,
                       length(path) AS hops
                LIMIT 1
            """, root=root_sensor, target=target_sensor, window_idx=window_idx)
            record = result.single()
            if record:
                return dict(record)
            else:
                return {'path_nodes': [], 'hops': 0}
    
    def get_temporal_sensor_history(self, window_idx: int, window_range: List[int]) -> List[Dict]:
        """
        Temporal Retrieval: Get sensor anomaly history over multiple time windows.
        
        Retrieves anomaly scores for sensors across a range of windows to identify
        temporal patterns (e.g., sensors that become anomalous early, persistent faults).
        
        Args:
            window_idx: Current window index (center of range)
            window_range: List of window indices to query (e.g., [t-2, t-1, t])
            
        Returns:
            List of dictionaries with keys:
            - 'sensor': base sensor name
            - 'window': window index
            - 'score': anomaly_score in that window
            - 'subsystem': sensor subsystem
            - 'temporal_pattern': 'increasing', 'decreasing', 'persistent', or 'spike'
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor)
                WHERE s.window IN $window_range
                RETURN s.base_sensor_name AS sensor,
                       s.window AS window,
                       s.anomaly_score AS score,
                       s.subsystem AS subsystem
                ORDER BY s.base_sensor_name, s.window
            """, window_range=window_range)
            
            records = [dict(record) for record in result]
            
            # Group by sensor and compute temporal patterns
            sensor_history = {}
            for record in records:
                sensor = record['sensor']
                if sensor not in sensor_history:
                    sensor_history[sensor] = []
                sensor_history[sensor].append({
                    'window': record['window'],
                    'score': record['score'],
                    'subsystem': record['subsystem']
                })
            
            # Analyze temporal patterns
            result_list = []
            for sensor, history in sensor_history.items():
                if len(history) < 2:
                    pattern = 'single_window'
                else:
                    scores = [h['score'] for h in sorted(history, key=lambda x: x['window'])]
                    if all(scores[i] <= scores[i+1] for i in range(len(scores)-1)):
                        pattern = 'increasing'
                    elif all(scores[i] >= scores[i+1] for i in range(len(scores)-1)):
                        pattern = 'decreasing'
                    elif all(s > 0.5 for s in scores):
                        pattern = 'persistent'
                    elif max(scores) > 0.7 and min(scores) < 0.3:
                        pattern = 'spike'
                    else:
                        pattern = 'variable'
                
                # Add all windows for this sensor
                for h in history:
                    result_list.append({
                        'sensor': sensor,
                        'window': h['window'],
                        'score': h['score'],
                        'subsystem': h['subsystem'],
                        'temporal_pattern': pattern
                    })
            
            return result_list
    
    def explore_neighborhood(self, root_sensor: str, window_idx: int, radius: int = 2) -> Dict:
        """
        Exploration: Expand k-hop neighborhood from a root sensor.
        
        Finds all sensors within k hops of the root sensor via CORRELATES_WITH
        relationships, along with their anomaly scores and violations.
        
        Args:
            root_sensor: Base sensor name of root sensor
            window_idx: Window index to query
            radius: Number of hops to expand (default: 2)
            
        Returns:
            Dictionary with keys:
            - 'root_sensor': root sensor name
            - 'neighbors': List of dicts with keys:
                - 'sensor': base sensor name
                - 'hop': distance from root (1, 2, ...)
                - 'score': anomaly_score
                - 'subsystem': sensor subsystem
                - 'violations_count': number of violations involving this sensor
            - 'summary': Dict with 'total_neighbors', 'anomalous_count', 'violations_count'
        """
        with self.driver.session() as session:
            # Find all neighbors within radius hops
            result = session.run("""
                MATCH (root:Sensor {base_sensor_name: $root, window: $window_idx})
                MATCH path = (root)-[:CORRELATES_WITH*1..$radius]-(neighbor:Sensor {window: $window_idx})
                WHERE length(path) <= $radius
                WITH neighbor, length(path) AS hop
                RETURN DISTINCT neighbor.base_sensor_name AS sensor,
                       hop,
                       neighbor.anomaly_score AS score,
                       neighbor.subsystem AS subsystem
                ORDER BY hop, neighbor.anomaly_score DESC
            """, root=root_sensor, window_idx=window_idx, radius=radius)
            
            neighbors = []
            for record in result:
                neighbors.append({
                    'sensor': record['sensor'],
                    'hop': record['hop'],
                    'score': record['score'],
                    'subsystem': record['subsystem']
                })
            
            # Count violations for each neighbor
            neighbor_names = [n['sensor'] for n in neighbors]
            violations_count = {}
            if neighbor_names:
                violations_result = session.run("""
                    MATCH (s1:Sensor {window: $window_idx})-[r:CORRELATES_WITH]->(s2:Sensor {window: $window_idx})
                    WHERE s1.base_sensor_name IN $neighbors 
                       OR s2.base_sensor_name IN $neighbors
                       AND abs(r.actual_correlation - r.expected_correlation) > 0.3
                    WITH s1.base_sensor_name AS sensor1, s2.base_sensor_name AS sensor2
                    UNWIND [sensor1, sensor2] AS sensor
                    WITH sensor, count(*) AS violations
                    RETURN sensor, violations
                """, window_idx=window_idx, neighbors=neighbor_names)
                
                for record in violations_result:
                    violations_count[record['sensor']] = record['violations']
            
            # Add violation counts to neighbors
            for neighbor in neighbors:
                neighbor['violations_count'] = violations_count.get(neighbor['sensor'], 0)
            
            # Compute summary
            anomalous_count = sum(1 for n in neighbors if n['score'] > 0.5)
            total_violations = sum(n['violations_count'] for n in neighbors)
            
            return {
                'root_sensor': root_sensor,
                'neighbors': neighbors,
                'summary': {
                    'total_neighbors': len(neighbors),
                    'anomalous_count': anomalous_count,
                    'violations_count': total_violations
                }
            }