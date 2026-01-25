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
        
        Args:
            window_idx: Window index to query
            deviation_threshold: Minimum deviation to consider a violation (default: 0.3)
            
        Returns:
            List of dictionaries with keys:
            - 'source': base sensor name
            - 'target': base sensor name
            - 'actual': actual_correlation value
            - 'expected': expected_correlation value
            - 'deviation': absolute deviation
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Sensor {window: $window_idx})-[r:CORRELATES_WITH]->(s2:Sensor {window: $window_idx})
                WHERE abs(r.actual_correlation - r.expected_correlation) > $threshold
                RETURN s1.base_sensor_name AS source,
                       s2.base_sensor_name AS target,
                       r.actual_correlation AS actual,
                       r.expected_correlation AS expected,
                       abs(r.actual_correlation - r.expected_correlation) AS deviation
                ORDER BY deviation DESC
            """, window_idx=window_idx, threshold=deviation_threshold)
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
