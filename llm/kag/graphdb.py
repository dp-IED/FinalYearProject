"""
Neo4j Loader for Window Analysis Data

Loads window analysis data from KnowledgeGraphBuilder into Neo4j database
following the specified schema:
- Sensor nodes with metadata and base_correlations (from GDN adjacency matrix)
- Window nodes with labels, fault_type, and faulty_sensor
- HAS_READING relationships (Window -> Sensor) containing:
  * Statistical properties (mean, std, min, max, etc.)
  * Raw time-series arrays (readings, normalized_readings, timesteps)
- CORRELATES_WITH relationships (Sensor -> Sensor) per window:
  * actual_correlation: actual correlation value in this window
  * expected_correlation: expected correlation from GDN adjacency matrix (learned normal)
  * Edges scoped to windows via 'window' property - same sensors can have different correlations per window
  * No threshold - all correlations stored, LLM decides significance at runtime
- Temporal propagation relationships (PRECEDES, PROPAGATES)
"""

import neo4j
from typing import List, Optional
import numpy as np
from pathlib import Path
import sys
import json

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.helpers.KG import KnowledgeGraphBuilder, SENSOR_SUBSYSTEMS, SENSOR_DESCRIPTIONS


class Neo4jLoader:
    """
    Loads window analysis data from KnowledgeGraphBuilder into Neo4j.
    """
    
    def __init__(self, uri: str = "bolt://127.0.0.1:7687", 
                 user: str = "neo4j", password: str = "password"):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
    
    def connect(self):
        """Establish connection to Neo4j."""
        if self.driver is None:
            self.driver = neo4j.GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def create_schema(self):
        """
        Create all constraints and indexes for the Neo4j schema.
        """
        self.connect()
        
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT sensor_name_unique IF NOT EXISTS
                FOR (s:Sensor)
                REQUIRE s.name IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT window_label_unique IF NOT EXISTS
                FOR (w:Window)
                REQUIRE w.label IS UNIQUE
            """)
            
            # Create index on CORRELATES_WITH.window for efficient queries
            session.run("""
                CREATE INDEX corr_window_idx IF NOT EXISTS
                FOR ()-[r:CORRELATES_WITH]-()
                ON (r.window)
            """)
            
            print("✓ Schema created (constraints and indexes)")
    
    def load_sensors(self, sensor_names: List[str], kg_builder: Optional[KnowledgeGraphBuilder] = None):
        """
        Create Sensor nodes with metadata and base correlations from GDN adjacency matrix.
        
        Args:
            sensor_names: List of sensor names
            kg_builder: Optional KnowledgeGraphBuilder instance to extract base correlations
        """
        self.connect()
        
        with self.driver.session() as session:
            for i, sensor_name in enumerate(sensor_names):
                subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, 'Unknown')
                description_data = SENSOR_DESCRIPTIONS.get(sensor_name, {})
                description = description_data.get('description', '')
                
                # Store base correlations from GDN adjacency matrix (learned normal correlations)
                base_correlations = {}
                if kg_builder is not None and kg_builder.adjacency_matrix is not None:
                    for j, other_sensor in enumerate(sensor_names):
                        if i != j:
                            base_correlations[other_sensor] = float(kg_builder.adjacency_matrix[i, j])
                
                if base_correlations:
                    # Convert dictionary to JSON string for Neo4j storage
                    base_correlations_json = json.dumps(base_correlations)
                    session.run("""
                        MERGE (s:Sensor {name: $name})
                        SET s.subsystem = $subsystem,
                            s.description = $description,
                            s.base_correlations = $base_correlations_json
                    """, name=sensor_name, subsystem=subsystem, 
                        description=description, base_correlations_json=base_correlations_json)
                else:
                    session.run("""
                        MERGE (s:Sensor {name: $name})
                        SET s.subsystem = $subsystem,
                            s.description = $description
                    """, name=sensor_name, subsystem=subsystem, description=description)
            
            print(f"✓ Loaded {len(sensor_names)} sensor nodes" + 
                  (" with base correlations" if kg_builder is not None else ""))
    
    def load_windows(self, window_indices: List[int], window_labels: Optional[np.ndarray] = None,
                     sensor_labels: Optional[np.ndarray] = None, sensor_names: Optional[List[str]] = None,
                     batch_size: int = 200):
        """
        Create Window nodes.
        
        Args:
            window_indices: List of window indices (0, 1, 2, ..., N-1)
            window_labels: Optional array of binary fault labels (0 = normal, 1 = faulty)
                          Note: window_idx is stored as label, is_fault as 0/1
            sensor_labels: Optional (num_windows, num_sensors) array - binary fault labels per sensor
            sensor_names: Optional list of sensor names (must match sensor_labels order)
            batch_size: Number of windows to create per transaction
        """
        self.connect()
        
        # Prepare all window data first
        batch_data = []
        for idx, window_idx in enumerate(window_indices):
            # window_idx is the actual window index (0, 1, 2, ...)
            # window_labels[window_idx] is binary fault indicator (0 or 1) - use window_idx to index
            is_faulty = int(window_labels[window_idx]) if window_labels is not None and window_idx < len(window_labels) else 0
            
            # Determine fault_type (0 or 1-indexed sensor index) and faulty_sensor name
            fault_type = 0
            faulty_sensor = None
            
            if is_faulty and sensor_labels is not None and sensor_names is not None:
                # Find first faulty sensor (1-indexed: sensor 0 -> fault_type 1, sensor 7 -> fault_type 8)
                # Use window_idx directly to index sensor_labels (window_idx is the actual window index)
                if window_idx < len(sensor_labels):
                    faulty_indices = np.where(sensor_labels[window_idx] > 0)[0]
                    if len(faulty_indices) > 0:
                        fault_type = int(faulty_indices[0]) + 1  # 1-indexed
                        faulty_sensor = sensor_names[faulty_indices[0]]
            
            batch_data.append({
                'window_idx': window_idx,
                'is_faulty': is_faulty,
                'fault_type': fault_type,
                'faulty_sensor': faulty_sensor
            })
        
        # Batch load in transactions
        total_expected = len(batch_data)
        with self.driver.session() as session:
            total_windows = 0
            
            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i:i + batch_size]
                
                with session.begin_transaction() as tx:
                    for item in batch:
                        tx.run("""
                            MERGE (w:Window {label: $window_idx})
                            SET w.is_fault = $is_faulty,
                                w.fault_type = $fault_type,
                                w.faulty_sensor = $faulty_sensor
                        """, window_idx=item['window_idx'],
                            is_faulty=item['is_faulty'],
                            fault_type=item['fault_type'],
                            faulty_sensor=item['faulty_sensor'])
                    
                    tx.commit()
                    total_windows += len(batch)
                    
                    # Progress update every 10 batches
                    if (i // batch_size + 1) % 10 == 0 or total_windows >= total_expected:
                        percentage = (total_windows / total_expected * 100) if total_expected > 0 else 0
                        print(f"  Progress: {total_windows}/{total_expected} windows ({percentage:.1f}%)...")
            
            print(f"✓ Loaded {total_windows} window nodes")
    
    def load_window_readings(self, kg_builder: KnowledgeGraphBuilder, 
                            window_indices: List[int],
                            subsample_rate: int = 20,
                            batch_size: int = 10):
        """
        Create HAS_READING relationships between Windows and Sensors.
        Includes Layer 2: Statistical summaries as properties.
        Includes Layer 3: Raw time-series readings as array properties (subsampled).
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices to load
            subsample_rate: Keep every Nth point for time-series (20 means 300 -> 15 points)
            batch_size: Number of relationships to create per transaction (reduced for memory)
        """
        self.connect()
        
        has_timeseries = kg_builder.X_windows_unnormalized is not None
        total_expected = sum(len(kg_builder.window_stats.get(w, {})) for w in window_indices)
        total_readings = 0
        
        with self.driver.session() as session:
            # Process windows incrementally to avoid memory buildup
            current_batch = []
            
            for window_idx in window_indices:
                if window_idx not in kg_builder.window_stats:
                    continue
                
                window_stats = kg_builder.window_stats[window_idx]
                window_has_timeseries = has_timeseries and window_idx < len(kg_builder.X_windows_unnormalized)
                
                for sensor_idx, sensor_name in enumerate(kg_builder.sensor_names):
                    if sensor_name not in window_stats:
                        continue
                    
                    stats = window_stats[sensor_name]
                    anomaly_score = float(stats.anomaly_score)
                    is_faulty = anomaly_score > 0.0
                    
                    # Prepare time-series arrays (Layer 3) if available
                    readings_array = None
                    normalized_readings_array = None
                    timesteps_array = None
                    
                    if window_has_timeseries:
                        window_data = kg_builder.X_windows_unnormalized[window_idx]  # (300, 8)
                        window_data_norm = kg_builder.X_windows[window_idx] if kg_builder.X_windows is not None else None
                        
                        full_series = window_data[:, sensor_idx]  # (300,)
                        
                        # Subsample: keep every Nth point
                        subsampled_indices = list(range(0, len(full_series), subsample_rate))
                        subsampled_values = full_series[subsampled_indices]
                        
                        readings_array = [float(v) for v in subsampled_values]
                        timesteps_array = [int(t) for t in subsampled_indices]
                        
                        # Get normalized values if available
                        if window_data_norm is not None:
                            subsampled_norm_values = window_data_norm[subsampled_indices, sensor_idx]
                            normalized_readings_array = [float(v) for v in subsampled_norm_values]
                    
                    # Add to current batch
                    current_batch.append({
                        'window_idx': window_idx,
                        'sensor_name': sensor_name,
                        'anomaly_score': anomaly_score,
                        'is_faulty': is_faulty,
                        'mean': stats.mean,
                        'std': stats.std,
                        'min': stats.min,
                        'max': stats.max,
                        'variance': stats.variance,
                        'num_zeros': stats.num_zeros,
                        'trend': stats.trend,
                        'median': stats.median,
                        'q25': stats.q25,
                        'q75': stats.q75,
                        'readings': readings_array,
                        'normalized_readings': normalized_readings_array,
                        'timesteps': timesteps_array,
                        'has_timeseries': window_has_timeseries and readings_array is not None
                    })
                    
                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        with session.begin_transaction() as tx:
                            for item in current_batch:
                                if item['has_timeseries']:
                                    tx.run("""
                                        MATCH (w:Window {label: $window_idx})
                                        MATCH (s:Sensor {name: $sensor_name})
                                        MERGE (w)-[r:HAS_READING]->(s)
                                        SET r.anomaly_score = $anomaly_score,
                                            r.is_faulty = $is_faulty,
                                            r.mean = $mean,
                                            r.std = $std,
                                            r.min = $min,
                                            r.max = $max,
                                            r.variance = $variance,
                                            r.num_zeros = $num_zeros,
                                            r.trend = $trend,
                                            r.median = $median,
                                            r.q25 = $q25,
                                            r.q75 = $q75,
                                            r.readings = $readings,
                                            r.normalized_readings = $normalized_readings,
                                            r.timesteps = $timesteps
                                    """, window_idx=item['window_idx'],
                                        sensor_name=item['sensor_name'],
                                        anomaly_score=item['anomaly_score'],
                                        is_faulty=item['is_faulty'],
                                        mean=item['mean'], std=item['std'],
                                        min=item['min'], max=item['max'],
                                        variance=item['variance'],
                                        num_zeros=item['num_zeros'],
                                        trend=item['trend'],
                                        median=item['median'],
                                        q25=item['q25'], q75=item['q75'],
                                        readings=item['readings'],
                                        normalized_readings=item['normalized_readings'],
                                        timesteps=item['timesteps'])
                                else:
                                    tx.run("""
                                        MATCH (w:Window {label: $window_idx})
                                        MATCH (s:Sensor {name: $sensor_name})
                                        MERGE (w)-[r:HAS_READING]->(s)
                                        SET r.anomaly_score = $anomaly_score,
                                            r.is_faulty = $is_faulty,
                                            r.mean = $mean,
                                            r.std = $std,
                                            r.min = $min,
                                            r.max = $max,
                                            r.variance = $variance,
                                            r.num_zeros = $num_zeros,
                                            r.trend = $trend,
                                            r.median = $median,
                                            r.q25 = $q25,
                                            r.q75 = $q75
                                    """, window_idx=item['window_idx'],
                                        sensor_name=item['sensor_name'],
                                        anomaly_score=item['anomaly_score'],
                                        is_faulty=item['is_faulty'],
                                        mean=item['mean'], std=item['std'],
                                        min=item['min'], max=item['max'],
                                        variance=item['variance'],
                                        num_zeros=item['num_zeros'],
                                        trend=item['trend'],
                                        median=item['median'],
                                        q25=item['q25'], q75=item['q75'])
                            
                            tx.commit()
                            total_readings += len(current_batch)
                            
                            if total_readings % (batch_size * 10) == 0:
                                percentage = (total_readings / total_expected * 100) if total_expected > 0 else 0
                                print(f"  Progress: {total_readings}/{total_expected} ({percentage:.1f}%)...")
                        
                        # Clear batch to free memory
                        current_batch = []
                        import gc
                        gc.collect()  # Force garbage collection to free memory
            
            # Process remaining items in final batch
            if current_batch:
                with session.begin_transaction() as tx:
                    for item in current_batch:
                        if item['has_timeseries']:
                            tx.run("""
                                MATCH (w:Window {label: $window_idx})
                                MATCH (s:Sensor {name: $sensor_name})
                                MERGE (w)-[r:HAS_READING]->(s)
                                SET r.anomaly_score = $anomaly_score,
                                    r.is_faulty = $is_faulty,
                                    r.mean = $mean,
                                    r.std = $std,
                                    r.min = $min,
                                    r.max = $max,
                                    r.variance = $variance,
                                    r.num_zeros = $num_zeros,
                                    r.trend = $trend,
                                    r.median = $median,
                                    r.q25 = $q25,
                                    r.q75 = $q75,
                                    r.readings = $readings,
                                    r.normalized_readings = $normalized_readings,
                                    r.timesteps = $timesteps
                            """, window_idx=item['window_idx'],
                                sensor_name=item['sensor_name'],
                                anomaly_score=item['anomaly_score'],
                                is_faulty=item['is_faulty'],
                                mean=item['mean'], std=item['std'],
                                min=item['min'], max=item['max'],
                                variance=item['variance'],
                                num_zeros=item['num_zeros'],
                                trend=item['trend'],
                                median=item['median'],
                                q25=item['q25'], q75=item['q75'],
                                readings=item['readings'],
                                normalized_readings=item['normalized_readings'],
                                timesteps=item['timesteps'])
                        else:
                            tx.run("""
                                MATCH (w:Window {label: $window_idx})
                                MATCH (s:Sensor {name: $sensor_name})
                                MERGE (w)-[r:HAS_READING]->(s)
                                SET r.anomaly_score = $anomaly_score,
                                    r.is_faulty = $is_faulty,
                                    r.mean = $mean,
                                    r.std = $std,
                                    r.min = $min,
                                    r.max = $max,
                                    r.variance = $variance,
                                    r.num_zeros = $num_zeros,
                                    r.trend = $trend,
                                    r.median = $median,
                                    r.q25 = $q25,
                                    r.q75 = $q75
                            """, window_idx=item['window_idx'],
                                sensor_name=item['sensor_name'],
                                anomaly_score=item['anomaly_score'],
                                is_faulty=item['is_faulty'],
                                mean=item['mean'], std=item['std'],
                                min=item['min'], max=item['max'],
                                variance=item['variance'],
                                num_zeros=item['num_zeros'],
                                trend=item['trend'],
                                median=item['median'],
                                q25=item['q25'], q75=item['q75'])
                    
                    tx.commit()
                    total_readings += len(current_batch)
            
            # Verify readings were loaded
            sample_check = session.run("""
                MATCH ()-[r:HAS_READING]->()
                WHERE r.readings IS NOT NULL
                RETURN count(r) as with_readings,
                       size(r.readings) as sample_size
                LIMIT 1
            """).single()
            
            if sample_check:
                print(f"✓ Loaded {total_readings} HAS_READING relationships")
                print(f"  - {sample_check['with_readings']} relationships have readings arrays")
                print(f"  - Sample array size: {sample_check['sample_size']} values")
            else:
                print(f"✓ Loaded {total_readings} HAS_READING relationships")
                print(f"  ⚠️  Warning: No readings arrays found on relationships")
    
    def load_correlations(self, kg_builder: KnowledgeGraphBuilder,
                         window_indices: List[int],
                         batch_size: int = 20):
        """
        Create CORRELATES_WITH relationships between Sensors per window.
        Stores actual_correlation and expected_correlation on edges.
        Edges are scoped to windows via the 'window' property, so same sensors can have
        different correlations in different windows without needing unique sensor nodes.
        No threshold - stores ALL correlations, letting LLM decide significance at runtime.
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices to load
            batch_size: Number of relationships to create per transaction
        """
        self.connect()
        
        # Process incrementally to avoid memory buildup
        with self.driver.session() as session:
            total_correlations = 0
            current_batch = []
            
            for window_idx in window_indices:
                # Compute correlation matrix for this window directly from data
                if kg_builder.X_windows is None:
                    continue
                
                if window_idx >= len(kg_builder.X_windows):
                    continue
                
                window_data = kg_builder.X_windows[window_idx]  # (300, 8) normalized
                
                try:
                    correlation_matrix = np.corrcoef(window_data.T)  # (8, 8)
                except Exception:
                    continue
                
                # Create CORRELATES_WITH edges for all sensor pairs (no threshold)
                for i, sensor_i in enumerate(kg_builder.sensor_names):
                    for j, sensor_j in enumerate(kg_builder.sensor_names):
                        if i >= j:  # Only create one direction (i < j)
                            continue
                        
                        # Actual correlation in this window
                        actual_corr = float(correlation_matrix[i, j])
                        
                        # Skip NaN correlations
                        if np.isnan(actual_corr):
                            continue
                        
                        # Expected correlation from GDN adjacency matrix (learned normal)
                        expected_corr = float(kg_builder.adjacency_matrix[i, j])
                        
                        # Ensure consistent edge direction: always from smaller to larger (alphabetically)
                        src_sensor, dst_sensor = sorted([sensor_i, sensor_j])
                        
                        current_batch.append({
                            'src': src_sensor,
                            'dst': dst_sensor,
                            'window': window_idx,
                            'actual_correlation': actual_corr,
                            'expected_correlation': expected_corr
                        })
                        
                        # Process batch when it reaches batch_size
                        if len(current_batch) >= batch_size:
                            with session.begin_transaction() as tx:
                                for item in current_batch:
                                    tx.run("""
                                        MATCH (s1:Sensor {name: $src})
                                        MATCH (s2:Sensor {name: $dst})
                                        MERGE (s1)-[r:CORRELATES_WITH {window: $window}]->(s2)
                                        SET r.actual_correlation = $actual_correlation,
                                            r.expected_correlation = $expected_correlation
                                    """, src=item['src'], dst=item['dst'],
                                        window=item['window'],
                                        actual_correlation=item['actual_correlation'],
                                        expected_correlation=item['expected_correlation'])
                                
                                tx.commit()
                                total_correlations += len(current_batch)
                                
                                if total_correlations % (batch_size * 20) == 0:
                                    print(f"  Progress: {total_correlations} correlations loaded...")
                            
                            # Clear batch to free memory
                            current_batch = []
                            import gc
                            gc.collect()  # Force garbage collection to free memory
            
            # Process remaining items in final batch
            if current_batch:
                with session.begin_transaction() as tx:
                    for item in current_batch:
                        tx.run("""
                            MATCH (s1:Sensor {name: $src})
                            MATCH (s2:Sensor {name: $dst})
                            MERGE (s1)-[r:CORRELATES_WITH {window: $window}]->(s2)
                            SET r.actual_correlation = $actual_correlation,
                                r.expected_correlation = $expected_correlation
                        """, src=item['src'], dst=item['dst'],
                            window=item['window'],
                            actual_correlation=item['actual_correlation'],
                            expected_correlation=item['expected_correlation'])
                    
                    tx.commit()
                    total_correlations += len(current_batch)
            
            print(f"✓ Loaded {total_correlations} CORRELATES_WITH relationships")
    
    def load_temporal_propagation(self, kg_builder: KnowledgeGraphBuilder,
                                 window_indices: List[int]):
        """
        Create PRECEDES relationships between consecutive Windows and
        PROPAGATES relationships between Sensors for anomaly propagation.
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices (should be sorted)
        """
        self.connect()
        
        with self.driver.session() as session:
            # Create PRECEDES relationships between consecutive windows
            total_precedes = 0
            sorted_indices = sorted(window_indices)
            expected_precedes = len(sorted_indices) - 1
            
            print(f"  Creating PRECEDES relationships (expected: {expected_precedes})...")
            for i in range(len(sorted_indices) - 1):
                from_window = sorted_indices[i]
                to_window = sorted_indices[i + 1]
                
                session.run("""
                    MATCH (w1:Window {label: $from_window})
                    MATCH (w2:Window {label: $to_window})
                    MERGE (w1)-[:PRECEDES]->(w2)
                """, from_window=from_window, to_window=to_window)
                
                total_precedes += 1
            
            print(f"  ✓ Loaded {total_precedes} PRECEDES relationships")
            
            # Create PROPAGATES relationships from anomaly propagation chains
            # Note: propagation_timeline now only contains FIRST occurrence of each affected sensor
            total_propagates = 0
            total_chains = len(kg_builder.anomaly_propagation_chains)
            
            if total_chains > 0:
                print(f"  Processing {total_chains} anomaly propagation chains...")
            
            for chain_idx, chain in enumerate(kg_builder.anomaly_propagation_chains):
                root_sensor = chain.get('root_sensor')
                root_window = chain.get('root_window')
                propagation_timeline = chain.get('propagation_timeline', [])
                
                if not root_sensor or root_window is None:
                    continue
                
                # Extract affected sensors with their FIRST occurrence window
                # The timeline now only contains first occurrences (fixed in _track_anomaly_propagation)
                for timeline_entry in propagation_timeline:
                    window = timeline_entry.get('window')
                    affected = timeline_entry.get('affected_sensors', [])
                    
                    # Each timeline entry represents first occurrence of affected sensors
                    if window > root_window:
                        for sensor_name in affected:
                            if sensor_name != root_sensor:
                                result = session.run("""
                                    MATCH (s1:Sensor {name: $source_sensor})
                                    MATCH (s2:Sensor {name: $target_sensor})
                                    MERGE (s1)-[r:PROPAGATES]->(s2)
                                    ON CREATE SET r.from_window = $from_window,
                                                  r.to_window = $to_window
                                    ON MATCH SET r.from_window = CASE 
                                        WHEN r.from_window IS NULL OR r.from_window > $from_window 
                                        THEN $from_window 
                                        ELSE r.from_window 
                                    END,
                                    r.to_window = CASE 
                                        WHEN r.to_window IS NULL OR r.to_window > $to_window 
                                        THEN $to_window 
                                        ELSE r.to_window 
                                    END
                                    RETURN r
                                """, source_sensor=root_sensor,
                                    target_sensor=sensor_name,
                                    from_window=root_window,
                                    to_window=window)
                                
                                # Only count if relationship was created or updated
                                if result.single():
                                    total_propagates += 1
                
                # Progress update every 10 chains
                if (chain_idx + 1) % 10 == 0 or (chain_idx + 1) >= total_chains:
                    percentage = ((chain_idx + 1) / total_chains * 100) if total_chains > 0 else 0
                    print(f"  Progress: {chain_idx + 1}/{total_chains} chains processed ({percentage:.1f}%)...")
            
            print(f"  ✓ Loaded {total_propagates} PROPAGATES relationships")
    
    def load_timeseries_readings(self, kg_builder: KnowledgeGraphBuilder,
                                window_indices: List[int],
                                subsample_rate: int = 20):
        """
        Create Reading nodes with subsampled time-series data (Layer 3).
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices to load
            subsample_rate: Keep every Nth point (20 means 300 -> 15 points)
        """
        self.connect()
        
        if kg_builder.X_windows_unnormalized is None:
            print("⚠️  Warning: Unnormalized windows not available, skipping time-series layer")
            return
        
        with self.driver.session() as session:
            total_readings = 0
            
            for window_idx in window_indices:
                if window_idx >= len(kg_builder.X_windows_unnormalized):
                    continue
                
                window_data = kg_builder.X_windows_unnormalized[window_idx]  # (300, 8)
                window_data_norm = kg_builder.X_windows[window_idx] if kg_builder.X_windows is not None else None
                
                for sensor_idx, sensor_name in enumerate(kg_builder.sensor_names):
                    full_series = window_data[:, sensor_idx]  # (300,)
                    
                    # Subsample: keep every Nth point
                    subsampled_indices = list(range(0, len(full_series), subsample_rate))
                    subsampled_values = full_series[subsampled_indices]
                    
                    # Get normalized values if available
                    if window_data_norm is not None:
                        subsampled_norm_values = window_data_norm[subsampled_indices, sensor_idx]
                    else:
                        subsampled_norm_values = subsampled_values
                    
                    # Create Reading nodes for each subsampled point
                    for i, (timestep, value, norm_value) in enumerate(zip(
                        subsampled_indices, subsampled_values, subsampled_norm_values
                    )):
                        reading_id = f"w{window_idx}_{sensor_name.replace(' ', '_').replace('()', '')}_{i}"
                        
                        session.run("""
                            MATCH (w:Window {label: $window_idx})
                            MATCH (s:Sensor {name: $sensor_name})
                            MERGE (r:Reading {id: $reading_id})
                            SET r.timestep = $timestep,
                                r.value = $value,
                                r.normalized_value = $normalized_value
                            MERGE (w)-[:CONTAINS]->(r)
                            MERGE (r)-[:MEASURES]->(s)
                        """, window_idx=window_idx, sensor_name=sensor_name,
                            reading_id=reading_id, timestep=int(timestep),
                            value=float(value), normalized_value=float(norm_value))
                        
                        total_readings += 1
            
            print(f"✓ Loaded {total_readings} Reading nodes (Layer 3: time-series)")
    
    def load_from_kg_builder(self, kg_builder: KnowledgeGraphBuilder,
                            window_labels: Optional[np.ndarray] = None,
                            sensor_labels: Optional[np.ndarray] = None):
        """
        Main entry point: Load all data from KnowledgeGraphBuilder into Neo4j.
        
        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_labels: Optional array of window labels (ground truth binary: 0 or 1)
            sensor_labels: Optional (num_windows, num_sensors) array - binary fault labels per sensor
        """
        import time
        
        self.connect()
        
        print("=" * 80)
        print("Loading Knowledge Graph into Neo4j")
        print("=" * 80)
        print()
        
        start_time = time.time()
        
        # Get window indices
        window_indices = sorted(kg_builder.window_graphs.keys())
        
        if not window_indices:
            print("⚠️  No windows found in KnowledgeGraphBuilder")
            return
        
        total_windows = len(window_indices)
        total_steps = 5
        
        print("📊 Dataset Overview:")
        print(f"   Windows: {total_windows}")
        print(f"   Sensors: {len(kg_builder.sensor_names)}")
        print(f"   Total steps: {total_steps}")
        print()
        
        # Step 1: Load sensors (with base correlations from GDN adjacency matrix)
        try:
            step_start = time.time()
            print(f"[1/{total_steps}] Loading sensors with base correlations...")
            self.load_sensors(kg_builder.sensor_names, kg_builder)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            raise
        
        # Step 2: Load windows (with fault_type and faulty_sensor)
        try:
            step_start = time.time()
            print(f"[2/{total_steps}] Loading windows...")
            self.load_windows(window_indices, window_labels, sensor_labels, kg_builder.sensor_names)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 2 (windows): {e}")
            raise
        
        # Step 3: Load window readings (HAS_READING) - Statistical summaries and time-series arrays
        try:
            step_start = time.time()
            print(f"[3/{total_steps}] Loading window readings (HAS_READING relationships)...")
            print("   This includes:")
            print("   - Statistical properties")
            print("   - Time-series arrays")
            expected_readings = sum(len(kg_builder.window_stats.get(w, {})) for w in window_indices)
            print(f"   Expected: ~{expected_readings} relationships")
            self.load_window_readings(kg_builder, window_indices, subsample_rate=20)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 3 (window readings): {e}")
            print("   💡 Tip: Try increasing Neo4j heap size or reducing dataset size")
            raise
        
        # Step 4: Load correlations (CORRELATES_WITH) - All correlations, no threshold
        try:
            step_start = time.time()
            print(f"[4/{total_steps}] Loading correlations (CORRELATES_WITH relationships)...")
            print("   Storing actual_correlation and expected_correlation for all sensor pairs")
            print("   Edges scoped to windows - no threshold, LLM decides significance at runtime")
            num_sensors = len(kg_builder.sensor_names)
            expected_correlations = total_windows * (num_sensors * (num_sensors - 1) // 2)
            print(f"   Expected: ~{expected_correlations} relationships")
            self.load_correlations(kg_builder, window_indices)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 4 (correlations): {e}")
            print("   💡 Tip: Try increasing Neo4j heap size or reducing dataset size")
            raise
        
        # Step 5: Load temporal propagation
        try:
            step_start = time.time()
            print(f"[5/{total_steps}] Loading temporal relationships (PRECEDES, PROPAGATES)...")
            self.load_temporal_propagation(kg_builder, window_indices)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 5 (temporal relationships): {e}")
            raise
        
        total_time = time.time() - start_time
        
        print("=" * 80)
        print(f"✓ Loading complete! Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print("=" * 80)
    
    def clear_database(self):
        """
        Clear all nodes and relationships from the database.
        Use with caution!
        """
        self.connect()
        
        with self.driver.session() as session:
            # Drop all constraints first (to avoid constraint violations)
            constraints_to_drop = [
                'window_label_unique',
                'window_id_unique', 
                'sensor_name_unique',
                'reading_id_unique'
            ]
            for constraint_name in constraints_to_drop:
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                except Exception:
                    pass  # Constraint might not exist
            
            # Clear all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")


if __name__ == "__main__":
    # Example usage
    print("Neo4j Loader for Window Analysis")
    print("Use this module programmatically:")
    print()
    print("  from llm.kag.init_neo4j import Neo4jLoader")
    print("  loader = Neo4jLoader()")
    print("  loader.create_schema()")
    print("  loader.load_from_kg_builder(kg_builder)")
    print()