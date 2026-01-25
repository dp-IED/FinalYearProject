"""
Neo4j Loader for Window Analysis Data

Loads window analysis data from KnowledgeGraphBuilder into Neo4j database
following the specified schema:
- Window nodes with labels, fault_type, and faulty_sensor
- Per-window Sensor nodes with composite names (Window_{idx}_Sensor_{name}):
  * Statistical properties (mean, std, min, max, variance, trend, etc.)
  * Raw time-series arrays (readings, normalized_readings, timesteps)
  * Base correlations from GDN adjacency matrix
  * BELONGS_TO relationships to Window nodes
- CORRELATES_WITH relationships (Sensor -> Sensor) within the same window:
  * actual_correlation: actual correlation value in this window
  * expected_correlation: expected correlation from GDN adjacency matrix (learned normal)
  * Correlations only exist between sensors of the same window (no window property on edges)
  * No threshold - all correlations stored, LLM decides significance at runtime
- Temporal relationships: PRECEDES (Window -> Window) for temporal ordering
"""

import neo4j
from typing import List, Optional, Dict, Any, Tuple
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

    def __init__(
        self,
        uri: str = "bolt://127.0.0.1:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
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

    def connect(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Establish connection to Neo4j with retry logic.
        
        Args:
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retries
        """
        if self.driver is None:
            import time
            last_error = None
            for attempt in range(max_retries):
                try:
                    self.driver = neo4j.GraphDatabase.driver(
                        self.uri, 
                        auth=(self.user, self.password),
                        max_connection_lifetime=3600,  # 1 hour
                        connection_timeout=30.0  # 30 seconds timeout
                    )
                    # Test the connection
                    with self.driver.session() as session:
                        session.run("RETURN 1").consume()
                    print(f"✓ Connected to Neo4j at {self.uri}")
                    return
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"  Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise ConnectionError(
                            f"Failed to connect to Neo4j after {max_retries} attempts. "
                            f"URI: {self.uri}, User: {self.user}. "
                            f"Last error: {last_error}. "
                            f"Please ensure Neo4j Desktop is running and the database is started."
                        ) from last_error

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
        Per-window sensor architecture: each window has its own sensor nodes.
        """
        self.connect()

        with self.driver.session() as session:
            # Create constraints
            # Sensor nodes have composite names: "Window_{window_idx}_Sensor_{sensor_name}"
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

            # Create index on Sensor.window for efficient window-based queries
            session.run("""
                CREATE INDEX sensor_window_idx IF NOT EXISTS
                FOR (s:Sensor)
                ON (s.window)
            """)

            # Create index on Sensor.base_sensor_name for querying sensor types across windows
            session.run("""
                CREATE INDEX sensor_base_name_idx IF NOT EXISTS
                FOR (s:Sensor)
                ON (s.base_sensor_name)
            """)

            print("✓ Schema created (constraints and indexes)")

    def load_window_sensors(
        self,
        kg_builder: KnowledgeGraphBuilder,
        window_indices: List[int],
        subsample_rate: int = 20,
        batch_size: int = 10,
    ):
        """
        Create Sensor nodes per window with all statistical and time-series properties.
        Each window gets its own set of sensor nodes with composite names.
        Creates BELONGS_TO relationships from sensors to their windows.

        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices to load
            subsample_rate: Keep every Nth point for time-series (20 means 300 -> 15 points)
            batch_size: Number of sensors to create per transaction
        """
        self.connect()

        has_timeseries = kg_builder.X_windows_unnormalized is not None
        total_expected = sum(
            len(kg_builder.window_stats.get(w, {})) for w in window_indices
        )
        total_sensors = 0

        # Pre-compute base correlations for each sensor type
        base_correlations_map = {}
        if kg_builder.adjacency_matrix is not None:
            for i, sensor_name in enumerate(kg_builder.sensor_names):
                base_correlations = {}
                for j, other_sensor in enumerate(kg_builder.sensor_names):
                    if i != j:
                        base_correlations[other_sensor] = float(
                            kg_builder.adjacency_matrix[i, j]
                        )
                base_correlations_map[sensor_name] = (
                    json.dumps(base_correlations) if base_correlations else None
                )

        with self.driver.session() as session:
            # Process windows incrementally to avoid memory buildup
            current_batch = []

            for window_idx in window_indices:
                if window_idx not in kg_builder.window_stats:
                    continue

                window_stats = kg_builder.window_stats[window_idx]
                window_has_timeseries = has_timeseries and window_idx < len(
                    kg_builder.X_windows_unnormalized
                )

                for sensor_idx, base_sensor_name in enumerate(kg_builder.sensor_names):
                    if base_sensor_name not in window_stats:
                        continue

                    stats = window_stats[base_sensor_name]
                    anomaly_score = float(stats.anomaly_score)
                    is_faulty = anomaly_score > 0.0

                    # Create composite sensor name: "Window_{window_idx}_Sensor_{sensor_name}"
                    composite_sensor_name = (
                        f"Window_{window_idx}_Sensor_{base_sensor_name}"
                    )

                    # Get sensor metadata
                    subsystem = SENSOR_SUBSYSTEMS.get(base_sensor_name, "Unknown")
                    description_data = SENSOR_DESCRIPTIONS.get(base_sensor_name, {})
                    description = description_data.get("description", "")

                    # Prepare time-series arrays (Layer 3) if available
                    readings_array = None
                    normalized_readings_array = None
                    timesteps_array = None

                    if window_has_timeseries:
                        window_data = kg_builder.X_windows_unnormalized[
                            window_idx
                        ]  # (300, 8)
                        window_data_norm = (
                            kg_builder.X_windows[window_idx]
                            if kg_builder.X_windows is not None
                            else None
                        )

                        full_series = window_data[:, sensor_idx]  # (300,)

                        # Subsample: keep every Nth point
                        subsampled_indices = list(
                            range(0, len(full_series), subsample_rate)
                        )
                        subsampled_values = full_series[subsampled_indices]

                        readings_array = [float(v) for v in subsampled_values]
                        timesteps_array = [int(t) for t in subsampled_indices]

                        # Get normalized values if available
                        if window_data_norm is not None:
                            subsampled_norm_values = window_data_norm[
                                subsampled_indices, sensor_idx
                            ]
                            normalized_readings_array = [
                                float(v) for v in subsampled_norm_values
                            ]

                    # Add to current batch
                    current_batch.append(
                        {
                            "composite_name": composite_sensor_name,
                            "window_idx": window_idx,
                            "base_sensor_name": base_sensor_name,
                            "subsystem": subsystem,
                            "description": description,
                            "base_correlations_json": base_correlations_map.get(
                                base_sensor_name
                            ),
                            "anomaly_score": anomaly_score,
                            "is_faulty": is_faulty,
                            "mean": stats.mean,
                            "std": stats.std,
                            "min": stats.min,
                            "max": stats.max,
                            "variance": stats.variance,
                            "num_zeros": stats.num_zeros,
                            "trend": stats.trend,
                            "median": stats.median,
                            "q25": stats.q25,
                            "q75": stats.q75,
                            "readings": readings_array,
                            "normalized_readings": normalized_readings_array,
                            "timesteps": timesteps_array,
                            "has_timeseries": window_has_timeseries
                            and readings_array is not None,
                        }
                    )

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        with session.begin_transaction() as tx:
                            for item in current_batch:
                                if item["has_timeseries"]:
                                    if item["base_correlations_json"]:
                                        tx.run(
                                            """
                                            MATCH (w:Window {label: $window_idx})
                                            MERGE (s:Sensor {name: $composite_name})
                                            SET s.window = $window_idx,
                                                s.base_sensor_name = $base_sensor_name,
                                                s.subsystem = $subsystem,
                                                s.description = $description,
                                                s.base_correlations = $base_correlations_json,
                                                s.anomaly_score = $anomaly_score,
                                                s.is_faulty = $is_faulty,
                                                s.mean = $mean,
                                                s.std = $std,
                                                s.min = $min,
                                                s.max = $max,
                                                s.variance = $variance,
                                                s.num_zeros = $num_zeros,
                                                s.trend = $trend,
                                                s.median = $median,
                                                s.q25 = $q25,
                                                s.q75 = $q75,
                                                s.readings = $readings,
                                                s.normalized_readings = $normalized_readings,
                                                s.timesteps = $timesteps
                                            MERGE (s)-[:BELONGS_TO]->(w)
                                        """,
                                            window_idx=item["window_idx"],
                                            composite_name=item["composite_name"],
                                            base_sensor_name=item["base_sensor_name"],
                                            subsystem=item["subsystem"],
                                            description=item["description"],
                                            base_correlations_json=item[
                                                "base_correlations_json"
                                            ],
                                            anomaly_score=item["anomaly_score"],
                                            is_faulty=item["is_faulty"],
                                            mean=item["mean"],
                                            std=item["std"],
                                            min=item["min"],
                                            max=item["max"],
                                            variance=item["variance"],
                                            num_zeros=item["num_zeros"],
                                            trend=item["trend"],
                                            median=item["median"],
                                            q25=item["q25"],
                                            q75=item["q75"],
                                            readings=item["readings"],
                                            normalized_readings=item[
                                                "normalized_readings"
                                            ],
                                            timesteps=item["timesteps"],
                                        )
                                    else:
                                        tx.run(
                                            """
                                            MATCH (w:Window {label: $window_idx})
                                            MERGE (s:Sensor {name: $composite_name})
                                            SET s.window = $window_idx,
                                                s.base_sensor_name = $base_sensor_name,
                                                s.subsystem = $subsystem,
                                                s.description = $description,
                                                s.anomaly_score = $anomaly_score,
                                                s.is_faulty = $is_faulty,
                                                s.mean = $mean,
                                                s.std = $std,
                                                s.min = $min,
                                                s.max = $max,
                                                s.variance = $variance,
                                                s.num_zeros = $num_zeros,
                                                s.trend = $trend,
                                                s.median = $median,
                                                s.q25 = $q25,
                                                s.q75 = $q75,
                                                s.readings = $readings,
                                                s.normalized_readings = $normalized_readings,
                                                s.timesteps = $timesteps
                                            MERGE (s)-[:BELONGS_TO]->(w)
                                        """,
                                            window_idx=item["window_idx"],
                                            composite_name=item["composite_name"],
                                            base_sensor_name=item["base_sensor_name"],
                                            subsystem=item["subsystem"],
                                            description=item["description"],
                                            anomaly_score=item["anomaly_score"],
                                            is_faulty=item["is_faulty"],
                                            mean=item["mean"],
                                            std=item["std"],
                                            min=item["min"],
                                            max=item["max"],
                                            variance=item["variance"],
                                            num_zeros=item["num_zeros"],
                                            trend=item["trend"],
                                            median=item["median"],
                                            q25=item["q25"],
                                            q75=item["q75"],
                                            readings=item["readings"],
                                            normalized_readings=item[
                                                "normalized_readings"
                                            ],
                                            timesteps=item["timesteps"],
                                        )
                                else:
                                    if item["base_correlations_json"]:
                                        tx.run(
                                            """
                                            MATCH (w:Window {label: $window_idx})
                                            MERGE (s:Sensor {name: $composite_name})
                                            SET s.window = $window_idx,
                                                s.base_sensor_name = $base_sensor_name,
                                                s.subsystem = $subsystem,
                                                s.description = $description,
                                                s.base_correlations = $base_correlations_json,
                                                s.anomaly_score = $anomaly_score,
                                                s.is_faulty = $is_faulty,
                                                s.mean = $mean,
                                                s.std = $std,
                                                s.min = $min,
                                                s.max = $max,
                                                s.variance = $variance,
                                                s.num_zeros = $num_zeros,
                                                s.trend = $trend,
                                                s.median = $median,
                                                s.q25 = $q25,
                                                s.q75 = $q75
                                            MERGE (s)-[:BELONGS_TO]->(w)
                                        """,
                                            window_idx=item["window_idx"],
                                            composite_name=item["composite_name"],
                                            base_sensor_name=item["base_sensor_name"],
                                            subsystem=item["subsystem"],
                                            description=item["description"],
                                            base_correlations_json=item[
                                                "base_correlations_json"
                                            ],
                                            anomaly_score=item["anomaly_score"],
                                            is_faulty=item["is_faulty"],
                                            mean=item["mean"],
                                            std=item["std"],
                                            min=item["min"],
                                            max=item["max"],
                                            variance=item["variance"],
                                            num_zeros=item["num_zeros"],
                                            trend=item["trend"],
                                            median=item["median"],
                                            q25=item["q25"],
                                            q75=item["q75"],
                                        )
                                    else:
                                        tx.run(
                                            """
                                            MATCH (w:Window {label: $window_idx})
                                            MERGE (s:Sensor {name: $composite_name})
                                            SET s.window = $window_idx,
                                                s.base_sensor_name = $base_sensor_name,
                                                s.subsystem = $subsystem,
                                                s.description = $description,
                                                s.anomaly_score = $anomaly_score,
                                                s.is_faulty = $is_faulty,
                                                s.mean = $mean,
                                                s.std = $std,
                                                s.min = $min,
                                                s.max = $max,
                                                s.variance = $variance,
                                                s.num_zeros = $num_zeros,
                                                s.trend = $trend,
                                                s.median = $median,
                                                s.q25 = $q25,
                                                s.q75 = $q75
                                            MERGE (s)-[:BELONGS_TO]->(w)
                                        """,
                                            window_idx=item["window_idx"],
                                            composite_name=item["composite_name"],
                                            base_sensor_name=item["base_sensor_name"],
                                            subsystem=item["subsystem"],
                                            description=item["description"],
                                            anomaly_score=item["anomaly_score"],
                                            is_faulty=item["is_faulty"],
                                            mean=item["mean"],
                                            std=item["std"],
                                            min=item["min"],
                                            max=item["max"],
                                            variance=item["variance"],
                                            num_zeros=item["num_zeros"],
                                            trend=item["trend"],
                                            median=item["median"],
                                            q25=item["q25"],
                                            q75=item["q75"],
                                        )

                            tx.commit()
                            total_sensors += len(current_batch)

                            if total_sensors % (batch_size * 10) == 0:
                                percentage = (
                                    (total_sensors / total_expected * 100)
                                    if total_expected > 0
                                    else 0
                                )
                                print(
                                    f"  Progress: {total_sensors}/{total_expected} ({percentage:.1f}%)..."
                                )

                        # Clear batch to free memory
                        current_batch = []
                        import gc

                        gc.collect()  # Force garbage collection to free memory

            # Process remaining items in final batch
            if current_batch:
                with session.begin_transaction() as tx:
                    for item in current_batch:
                        if item["has_timeseries"]:
                            if item["base_correlations_json"]:
                                tx.run(
                                    """
                                    MATCH (w:Window {label: $window_idx})
                                    MERGE (s:Sensor {name: $composite_name})
                                    SET s.window = $window_idx,
                                        s.base_sensor_name = $base_sensor_name,
                                        s.subsystem = $subsystem,
                                        s.description = $description,
                                        s.base_correlations = $base_correlations_json,
                                        s.anomaly_score = $anomaly_score,
                                        s.is_faulty = $is_faulty,
                                        s.mean = $mean,
                                        s.std = $std,
                                        s.min = $min,
                                        s.max = $max,
                                        s.variance = $variance,
                                        s.num_zeros = $num_zeros,
                                        s.trend = $trend,
                                        s.median = $median,
                                        s.q25 = $q25,
                                        s.q75 = $q75,
                                        s.readings = $readings,
                                        s.normalized_readings = $normalized_readings,
                                        s.timesteps = $timesteps
                                    MERGE (s)-[:BELONGS_TO]->(w)
                                """,
                                    window_idx=item["window_idx"],
                                    composite_name=item["composite_name"],
                                    base_sensor_name=item["base_sensor_name"],
                                    subsystem=item["subsystem"],
                                    description=item["description"],
                                    base_correlations_json=item[
                                        "base_correlations_json"
                                    ],
                                    anomaly_score=item["anomaly_score"],
                                    is_faulty=item["is_faulty"],
                                    mean=item["mean"],
                                    std=item["std"],
                                    min=item["min"],
                                    max=item["max"],
                                    variance=item["variance"],
                                    num_zeros=item["num_zeros"],
                                    trend=item["trend"],
                                    median=item["median"],
                                    q25=item["q25"],
                                    q75=item["q75"],
                                    readings=item["readings"],
                                    normalized_readings=item["normalized_readings"],
                                    timesteps=item["timesteps"],
                                )
                            else:
                                tx.run(
                                    """
                                    MATCH (w:Window {label: $window_idx})
                                    MERGE (s:Sensor {name: $composite_name})
                                    SET s.window = $window_idx,
                                        s.base_sensor_name = $base_sensor_name,
                                        s.subsystem = $subsystem,
                                        s.description = $description,
                                        s.anomaly_score = $anomaly_score,
                                        s.is_faulty = $is_faulty,
                                        s.mean = $mean,
                                        s.std = $std,
                                        s.min = $min,
                                        s.max = $max,
                                        s.variance = $variance,
                                        s.num_zeros = $num_zeros,
                                        s.trend = $trend,
                                        s.median = $median,
                                        s.q25 = $q25,
                                        s.q75 = $q75,
                                        s.readings = $readings,
                                        s.normalized_readings = $normalized_readings,
                                        s.timesteps = $timesteps
                                    MERGE (s)-[:BELONGS_TO]->(w)
                                """,
                                    window_idx=item["window_idx"],
                                    composite_name=item["composite_name"],
                                    base_sensor_name=item["base_sensor_name"],
                                    subsystem=item["subsystem"],
                                    description=item["description"],
                                    anomaly_score=item["anomaly_score"],
                                    is_faulty=item["is_faulty"],
                                    mean=item["mean"],
                                    std=item["std"],
                                    min=item["min"],
                                    max=item["max"],
                                    variance=item["variance"],
                                    num_zeros=item["num_zeros"],
                                    trend=item["trend"],
                                    median=item["median"],
                                    q25=item["q25"],
                                    q75=item["q75"],
                                    readings=item["readings"],
                                    normalized_readings=item["normalized_readings"],
                                    timesteps=item["timesteps"],
                                )
                        else:
                            if item["base_correlations_json"]:
                                tx.run(
                                    """
                                    MATCH (w:Window {label: $window_idx})
                                    MERGE (s:Sensor {name: $composite_name})
                                    SET s.window = $window_idx,
                                        s.base_sensor_name = $base_sensor_name,
                                        s.subsystem = $subsystem,
                                        s.description = $description,
                                        s.base_correlations = $base_correlations_json,
                                        s.anomaly_score = $anomaly_score,
                                        s.is_faulty = $is_faulty,
                                        s.mean = $mean,
                                        s.std = $std,
                                        s.min = $min,
                                        s.max = $max,
                                        s.variance = $variance,
                                        s.num_zeros = $num_zeros,
                                        s.trend = $trend,
                                        s.median = $median,
                                        s.q25 = $q25,
                                        s.q75 = $q75
                                    MERGE (s)-[:BELONGS_TO]->(w)
                                """,
                                    window_idx=item["window_idx"],
                                    composite_name=item["composite_name"],
                                    base_sensor_name=item["base_sensor_name"],
                                    subsystem=item["subsystem"],
                                    description=item["description"],
                                    base_correlations_json=item[
                                        "base_correlations_json"
                                    ],
                                    anomaly_score=item["anomaly_score"],
                                    is_faulty=item["is_faulty"],
                                    mean=item["mean"],
                                    std=item["std"],
                                    min=item["min"],
                                    max=item["max"],
                                    variance=item["variance"],
                                    num_zeros=item["num_zeros"],
                                    trend=item["trend"],
                                    median=item["median"],
                                    q25=item["q25"],
                                    q75=item["q75"],
                                )
                            else:
                                tx.run(
                                    """
                                    MATCH (w:Window {label: $window_idx})
                                    MERGE (s:Sensor {name: $composite_name})
                                    SET s.window = $window_idx,
                                        s.base_sensor_name = $base_sensor_name,
                                        s.subsystem = $subsystem,
                                        s.description = $description,
                                        s.anomaly_score = $anomaly_score,
                                        s.is_faulty = $is_faulty,
                                        s.mean = $mean,
                                        s.std = $std,
                                        s.min = $min,
                                        s.max = $max,
                                        s.variance = $variance,
                                        s.num_zeros = $num_zeros,
                                        s.trend = $trend,
                                        s.median = $median,
                                        s.q25 = $q25,
                                        s.q75 = $q75
                                    MERGE (s)-[:BELONGS_TO]->(w)
                                """,
                                    window_idx=item["window_idx"],
                                    composite_name=item["composite_name"],
                                    base_sensor_name=item["base_sensor_name"],
                                    subsystem=item["subsystem"],
                                    description=item["description"],
                                    anomaly_score=item["anomaly_score"],
                                    is_faulty=item["is_faulty"],
                                    mean=item["mean"],
                                    std=item["std"],
                                    min=item["min"],
                                    max=item["max"],
                                    variance=item["variance"],
                                    num_zeros=item["num_zeros"],
                                    trend=item["trend"],
                                    median=item["median"],
                                    q25=item["q25"],
                                    q75=item["q75"],
                                )

                    tx.commit()
                    total_sensors += len(current_batch)

            # Verify sensors were loaded
            sample_check = session.run("""
                MATCH (s:Sensor)
                WHERE s.readings IS NOT NULL
                RETURN count(s) as with_readings,
                       size(s.readings) as sample_size
                LIMIT 1
            """).single()

            if sample_check:
                print(f"✓ Loaded {total_sensors} sensor nodes")
                print(
                    f"  - {sample_check['with_readings']} sensors have readings arrays"
                )
                print(f"  - Sample array size: {sample_check['sample_size']} values")
            else:
                print(f"✓ Loaded {total_sensors} sensor nodes")
                print(f"  ⚠️  Warning: No readings arrays found on sensors")

    def load_windows(
        self,
        window_indices: List[int],
        window_labels: Optional[np.ndarray] = None,
        sensor_labels: Optional[np.ndarray] = None,
        sensor_names: Optional[List[str]] = None,
        batch_size: int = 200,
    ):
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
            is_faulty = (
                int(window_labels[window_idx])
                if window_labels is not None and window_idx < len(window_labels)
                else 0
            )

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

            batch_data.append(
                {
                    "window_idx": window_idx,
                    "is_faulty": is_faulty,
                    "fault_type": fault_type,
                    "faulty_sensor": faulty_sensor,
                }
            )

        # Batch load in transactions
        total_expected = len(batch_data)
        with self.driver.session() as session:
            total_windows = 0

            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i : i + batch_size]

                with session.begin_transaction() as tx:
                    for item in batch:
                        tx.run(
                            """
                            MERGE (w:Window {label: $window_idx})
                            SET w.is_fault = $is_faulty,
                                w.fault_type = $fault_type,
                                w.faulty_sensor = $faulty_sensor
                        """,
                            window_idx=item["window_idx"],
                            is_faulty=item["is_faulty"],
                            fault_type=item["fault_type"],
                            faulty_sensor=item["faulty_sensor"],
                        )

                    tx.commit()
                    total_windows += len(batch)

                    # Progress update every 10 batches
                    if (
                        i // batch_size + 1
                    ) % 10 == 0 or total_windows >= total_expected:
                        percentage = (
                            (total_windows / total_expected * 100)
                            if total_expected > 0
                            else 0
                        )
                        print(
                            f"  Progress: {total_windows}/{total_expected} windows ({percentage:.1f}%)..."
                        )

            print(f"✓ Loaded {total_windows} window nodes")

    def sigmoid(self, x: float, center: float = 0.5, steepness: float = 10.0) -> float:
        """
        Smooth transition from 0 to 1 around center.
        
        Args:
            x: Input value
            center: Center point of sigmoid (default: 0.5)
            steepness: Steepness of transition (default: 10.0)
            
        Returns:
            Sigmoid value between 0 and 1
        """
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    def select_topk_correlations(
        self,
        actual_corr_matrix: np.ndarray,
        expected_corr_matrix: np.ndarray,
        gdn_anomaly_score: float,
        sensor_names: List[str],
        top_k: int = 10,
        threshold_center: float = 0.5,
        threshold_steepness: float = 10.0,
    ) -> List[dict]:
        """
        Select top-k most informative correlations for a single window.
        
        Uses continuous weighted selection strategy:
        - Normal windows (low GDN score): prioritize strong correlations
        - Anomalous windows (high GDN score): prioritize deviations from expected
        - Borderline windows: balanced blend
        
        Args:
            actual_corr_matrix: (N, N) empirical correlations for this window
            expected_corr_matrix: (N, N) learned adjacency (static, from GDN)
            gdn_anomaly_score: Scalar GDN anomaly score (max/mean sensor prob)
            sensor_names: List of sensor names (for indexing)
            top_k: Number of top correlations to select
            threshold_center: Sigmoid center point (default: 0.5)
            threshold_steepness: Sigmoid steepness (default: 10.0)
            
        Returns:
            List of dicts with keys: sensor_i, sensor_j, sensor_i_name, sensor_j_name,
            actual_corr, expected_corr, deviation, info_score, anomaly_weight
        """
        N = actual_corr_matrix.shape[0]
        
        # Compute anomaly weight (smooth 0→1 transition)
        anomaly_weight = self.sigmoid(gdn_anomaly_score, threshold_center, threshold_steepness)
        
        # Compute informativeness for all edges
        edges = []
        for i in range(N):
            for j in range(i + 1, N):  # Upper triangle (undirected)
                actual = actual_corr_matrix[i, j]
                expected = expected_corr_matrix[i, j]
                
                # Skip NaN correlations
                if np.isnan(actual) or np.isnan(expected):
                    continue
                
                deviation = abs(actual - expected)
                
                # Blend strength and deviation based on anomaly weight
                info_score = (1 - anomaly_weight) * abs(actual) + anomaly_weight * deviation
                
                edges.append({
                    'sensor_i': i,
                    'sensor_j': j,
                    'sensor_i_name': sensor_names[i],
                    'sensor_j_name': sensor_names[j],
                    'actual_corr': float(actual),
                    'expected_corr': float(expected),
                    'deviation': float(deviation),
                    'info_score': float(info_score),
                    'anomaly_weight': float(anomaly_weight),
                })
        
        # Sort by informativeness, take top-k
        edges = sorted(edges, key=lambda e: e['info_score'], reverse=True)[:top_k]
        
        return edges

    def load_correlations(
        self,
        kg_builder: KnowledgeGraphBuilder,
        window_indices: List[int],
        batch_size: int = 20,
        top_k: int = 10,
        threshold_center: float = 0.5,
        threshold_steepness: float = 10.0,
    ):
        """
        Create CORRELATES_WITH relationships between Sensors within the same window.
        Uses continuous weighted selection to load only top-k most informative correlations.
        
        Selection strategy:
        - Normal windows (low GDN score): prioritize strong correlations
        - Anomalous windows (high GDN score): prioritize deviations from expected
        - Borderline windows: balanced blend
        
        Stores actual_correlation, expected_correlation, deviation, info_score, and anomaly_weight on edges.
        Sensors are matched by window property - correlations only exist between sensors of the same window.

        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices to load
            batch_size: Number of relationships to create per transaction
            top_k: Number of top correlations to select per window (default: 10)
            threshold_center: Sigmoid center point for anomaly weight (default: 0.5)
            threshold_steepness: Sigmoid steepness for anomaly weight (default: 10.0)
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

                # Extract GDN anomaly score for this window
                # Use max sensor anomaly score as window-level score
                if window_idx not in kg_builder.window_stats:
                    # Fallback: skip window if no stats available
                    continue
                
                window_stats = kg_builder.window_stats[window_idx]
                gdn_anomaly_score = max(
                    stats.anomaly_score for stats in window_stats.values()
                )

                # Select top-k most informative correlations
                top_edges = self.select_topk_correlations(
                    actual_corr_matrix=correlation_matrix,
                    expected_corr_matrix=kg_builder.adjacency_matrix,
                    gdn_anomaly_score=gdn_anomaly_score,
                    sensor_names=kg_builder.sensor_names,
                    top_k=top_k,
                    threshold_center=threshold_center,
                    threshold_steepness=threshold_steepness,
                )

                # Process only the selected top-k edges
                for edge in top_edges:
                    # Build composite sensor names for this window
                    src_composite_name = (
                        f"Window_{window_idx}_Sensor_{edge['sensor_i_name']}"
                    )
                    dst_composite_name = (
                        f"Window_{window_idx}_Sensor_{edge['sensor_j_name']}"
                    )

                    # Ensure consistent edge direction: always from smaller to larger (alphabetically)
                    if edge['sensor_i_name'] < edge['sensor_j_name']:
                        final_src = src_composite_name
                        final_dst = dst_composite_name
                    else:
                        final_src = dst_composite_name
                        final_dst = src_composite_name

                    current_batch.append(
                        {
                            "src": final_src,
                            "dst": final_dst,
                            "window": window_idx,
                            "actual_correlation": edge["actual_corr"],
                            "expected_correlation": edge["expected_corr"],
                            "deviation": edge["deviation"],
                            "info_score": edge["info_score"],
                            "anomaly_weight": edge["anomaly_weight"],
                        }
                    )

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        with session.begin_transaction() as tx:
                            for item in current_batch:
                                tx.run(
                                    """
                                    MATCH (s1:Sensor {name: $src, window: $window})
                                    MATCH (s2:Sensor {name: $dst, window: $window})
                                    MERGE (s1)-[r:CORRELATES_WITH]->(s2)
                                    SET r.actual_correlation = $actual_correlation,
                                        r.expected_correlation = $expected_correlation,
                                        r.deviation = $deviation,
                                        r.info_score = $info_score,
                                        r.anomaly_weight = $anomaly_weight
                                """,
                                    src=item["src"],
                                    dst=item["dst"],
                                    window=item["window"],
                                    actual_correlation=item["actual_correlation"],
                                    expected_correlation=item["expected_correlation"],
                                    deviation=item["deviation"],
                                    info_score=item["info_score"],
                                    anomaly_weight=item["anomaly_weight"],
                                )

                            tx.commit()
                            total_correlations += len(current_batch)

                            if total_correlations % (batch_size * 20) == 0:
                                print(
                                    f"  Progress: {total_correlations} correlations loaded..."
                                )

                        # Clear batch to free memory
                        current_batch = []
                        import gc

                        gc.collect()  # Force garbage collection to free memory

            # Process remaining items in final batch
            if current_batch:
                with session.begin_transaction() as tx:
                    for item in current_batch:
                        tx.run(
                            """
                            MATCH (s1:Sensor {name: $src, window: $window})
                            MATCH (s2:Sensor {name: $dst, window: $window})
                            MERGE (s1)-[r:CORRELATES_WITH]->(s2)
                            SET r.actual_correlation = $actual_correlation,
                                r.expected_correlation = $expected_correlation,
                                r.deviation = $deviation,
                                r.info_score = $info_score,
                                r.anomaly_weight = $anomaly_weight
                        """,
                            src=item["src"],
                            dst=item["dst"],
                            window=item["window"],
                            actual_correlation=item["actual_correlation"],
                            expected_correlation=item["expected_correlation"],
                            deviation=item["deviation"],
                            info_score=item["info_score"],
                            anomaly_weight=item["anomaly_weight"],
                        )

                    tx.commit()
                    total_correlations += len(current_batch)

            print(f"✓ Loaded {total_correlations} CORRELATES_WITH relationships")

    def load_temporal_propagation(
        self, kg_builder: KnowledgeGraphBuilder, window_indices: List[int]
    ):
        """
        Create PRECEDES relationships between consecutive Windows for temporal ordering.
        PROPAGATES relationships are no longer used - windows are only connected via PRECEDES.

        Args:
            kg_builder: KnowledgeGraphBuilder instance (unused, kept for API compatibility)
            window_indices: List of window indices (should be sorted)
        """
        self.connect()

        with self.driver.session() as session:
            # Create PRECEDES relationships between consecutive windows
            total_precedes = 0
            sorted_indices = sorted(window_indices)
            expected_precedes = len(sorted_indices) - 1

            print(
                f"  Creating PRECEDES relationships (expected: {expected_precedes})..."
            )
            for i in range(len(sorted_indices) - 1):
                from_window = sorted_indices[i]
                to_window = sorted_indices[i + 1]

                session.run(
                    """
                    MATCH (w1:Window {label: $from_window})
                    MATCH (w2:Window {label: $to_window})
                    MERGE (w1)-[:PRECEDES]->(w2)
                """,
                    from_window=from_window,
                    to_window=to_window,
                )

                total_precedes += 1

            print(f"  ✓ Loaded {total_precedes} PRECEDES relationships")

    def load_timeseries_readings(
        self,
        kg_builder: KnowledgeGraphBuilder,
        window_indices: List[int],
        subsample_rate: int = 20,
    ):
        """
        Create Reading nodes with subsampled time-series data (Layer 3).

        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_indices: List of window indices to load
            subsample_rate: Keep every Nth point (20 means 300 -> 15 points)
        """
        self.connect()

        if kg_builder.X_windows_unnormalized is None:
            print(
                "⚠️  Warning: Unnormalized windows not available, skipping time-series layer"
            )
            return

        with self.driver.session() as session:
            total_readings = 0

            for window_idx in window_indices:
                if window_idx >= len(kg_builder.X_windows_unnormalized):
                    continue

                window_data = kg_builder.X_windows_unnormalized[window_idx]  # (300, 8)
                window_data_norm = (
                    kg_builder.X_windows[window_idx]
                    if kg_builder.X_windows is not None
                    else None
                )

                for sensor_idx, sensor_name in enumerate(kg_builder.sensor_names):
                    full_series = window_data[:, sensor_idx]  # (300,)

                    # Subsample: keep every Nth point
                    subsampled_indices = list(
                        range(0, len(full_series), subsample_rate)
                    )
                    subsampled_values = full_series[subsampled_indices]

                    # Get normalized values if available
                    if window_data_norm is not None:
                        subsampled_norm_values = window_data_norm[
                            subsampled_indices, sensor_idx
                        ]
                    else:
                        subsampled_norm_values = subsampled_values

                    # Create Reading nodes for each subsampled point
                    for i, (timestep, value, norm_value) in enumerate(
                        zip(
                            subsampled_indices,
                            subsampled_values,
                            subsampled_norm_values,
                        )
                    ):
                        reading_id = f"w{window_idx}_{sensor_name.replace(' ', '_').replace('()', '')}_{i}"

                        session.run(
                            """
                            MATCH (w:Window {label: $window_idx})
                            MATCH (s:Sensor {name: $sensor_name})
                            MERGE (r:Reading {id: $reading_id})
                            SET r.timestep = $timestep,
                                r.value = $value,
                                r.normalized_value = $normalized_value
                            MERGE (w)-[:CONTAINS]->(r)
                            MERGE (r)-[:MEASURES]->(s)
                        """,
                            window_idx=window_idx,
                            sensor_name=sensor_name,
                            reading_id=reading_id,
                            timestep=int(timestep),
                            value=float(value),
                            normalized_value=float(norm_value),
                        )

                        total_readings += 1

            print(f"✓ Loaded {total_readings} Reading nodes (Layer 3: time-series)")

    def load_from_kg_builder(
        self,
        kg_builder: KnowledgeGraphBuilder,
        window_labels: Optional[np.ndarray] = None,
        sensor_labels: Optional[np.ndarray] = None,
        window_indices_subset: Optional[List[int]] = None,
        top_k_correlations: int = 10,
        correlation_threshold_center: float = 0.5,
        correlation_threshold_steepness: float = 10.0,
    ):
        """
        Main entry point: Load data from KnowledgeGraphBuilder into Neo4j.

        Args:
            kg_builder: KnowledgeGraphBuilder instance with built KG
            window_labels: Optional array of window labels (ground truth binary: 0 or 1)
            sensor_labels: Optional (num_windows, num_sensors) array - binary fault labels per sensor
            window_indices_subset: Optional list of window indices to load (if None, loads all windows)
            top_k_correlations: Number of top correlations to select per window (default: 10)
            correlation_threshold_center: Sigmoid center point for anomaly weight (default: 0.5)
            correlation_threshold_steepness: Sigmoid steepness for anomaly weight (default: 10.0)
        """
        import time

        # Connect with retry logic
        try:
            self.connect(max_retries=3, retry_delay=2.0)
        except ConnectionError as e:
            raise ConnectionError(
                f"Neo4j connection failed. Please check:\n"
                f"  1. Neo4j Desktop is running\n"
                f"  2. The database is started (not just installed)\n"
                f"  3. Connection URI: {self.uri}\n"
                f"  4. Username: {self.user}\n"
                f"  5. Password matches your Neo4j Desktop settings\n"
                f"  6. Port 7687 is not blocked by firewall\n"
                f"\nOriginal error: {e}"
            ) from e

        print("=" * 80)
        print("Loading Knowledge Graph into Neo4j")
        print("=" * 80)
        print()

        start_time = time.time()

        # Get window indices (either subset or all)
        all_window_indices = sorted(kg_builder.window_graphs.keys())
        if window_indices_subset is not None:
            # Filter to only include windows that exist in the KG
            window_indices = sorted(
                [w for w in window_indices_subset if w in kg_builder.window_graphs]
            )
            print(
                f"📋 Loading subset: {len(window_indices)}/{len(all_window_indices)} windows"
            )
        else:
            window_indices = all_window_indices

        if not window_indices:
            print("⚠️  No windows found in KnowledgeGraphBuilder")
            return

        total_windows = len(window_indices)
        total_steps = 4

        print("📊 Dataset Overview:")
        print(f"   Windows: {total_windows}")
        print(f"   Sensors per window: {len(kg_builder.sensor_names)}")
        print(f"   Total steps: {total_steps}")
        print()

        # Step 1: Load windows (with fault_type and faulty_sensor)
        try:
            step_start = time.time()
            print(f"[1/{total_steps}] Loading windows...")
            self.load_windows(
                window_indices, window_labels, sensor_labels, kg_builder.sensor_names
            )
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 1 (windows): {e}")
            raise

        # Step 2: Load sensors per window (with all statistical and time-series properties)
        try:
            step_start = time.time()
            print(f"[2/{total_steps}] Loading sensors per window...")
            print("   This includes:")
            print("   - Statistical properties (mean, std, min, max, etc.)")
            print("   - Time-series arrays (subsampled)")
            print("   - Base correlations from GDN adjacency matrix")
            expected_sensors = sum(
                len(kg_builder.window_stats.get(w, {})) for w in window_indices
            )
            print(f"   Expected: ~{expected_sensors} sensor nodes")
            self.load_window_sensors(kg_builder, window_indices, subsample_rate=20)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 2 (window sensors): {e}")
            print("   💡 Tip: Try increasing Neo4j heap size or reducing dataset size")
            raise

        # Step 3: Load correlations (CORRELATES_WITH) - Top-k selection with continuous weighted strategy
        try:
            step_start = time.time()
            print(
                f"[3/{total_steps}] Loading correlations (CORRELATES_WITH relationships)..."
            )
            print(
                f"   Using top-{top_k_correlations} selection with continuous weighted strategy"
            )
            print(
                "   Storing actual_correlation, expected_correlation, deviation, info_score, anomaly_weight"
            )
            print("   Correlations only exist between sensors within the same window")
            print(
                "   Selection: normal windows prioritize strength, anomalous windows prioritize deviation"
            )
            expected_correlations = total_windows * top_k_correlations
            print(f"   Expected: ~{expected_correlations} relationships (top-{top_k_correlations} per window)")
            self.load_correlations(
                kg_builder,
                window_indices,
                top_k=top_k_correlations,
                threshold_center=correlation_threshold_center,
                threshold_steepness=correlation_threshold_steepness,
            )
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 3 (correlations): {e}")
            print("   💡 Tip: Try increasing Neo4j heap size or reducing dataset size")
            raise

        # Step 4: Load temporal relationships (PRECEDES only)
        try:
            step_start = time.time()
            print(f"[4/{total_steps}] Loading temporal relationships (PRECEDES)...")
            print("   Creating PRECEDES relationships between consecutive windows")
            self.load_temporal_propagation(kg_builder, window_indices)
            step_time = time.time() - step_start
            print(f"   ✓ Completed in {step_time:.2f}s")
            print()
        except Exception as e:
            print(f"   ✗ Failed at step 4 (temporal relationships): {e}")
            raise

        total_time = time.time() - start_time

        print("=" * 80)
        print(
            f"✓ Loading complete! Total time: {total_time:.2f}s ({total_time / 60:.2f} minutes)"
        )
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
                "window_label_unique",
                "window_id_unique",
                "sensor_name_unique",
                "reading_id_unique",
            ]
            for constraint_name in constraints_to_drop:
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                except Exception:
                    pass  # Constraint might not exist

            # Drop all indexes
            indexes_to_drop = [
                "sensor_window_idx",
                "sensor_base_name_idx",
                "corr_window_idx",  # Old index, may not exist
            ]
            for index_name in indexes_to_drop:
                try:
                    session.run(f"DROP INDEX {index_name} IF EXISTS")
                except Exception:
                    pass  # Index might not exist

            # Clear all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared")

    def sync_embeddings_to_neo4j(
        self,
        window_embeddings: Dict[int, Dict[str, Any]],
        center_embeddings: np.ndarray,
        gdn_predictions: Optional[np.ndarray] = None,
        batch_size: int = 100
    ) -> None:
        """
        Sync window embeddings and class centers to Neo4j.
        
        Creates ClassCenter nodes and updates Window nodes with embedding properties.
        Also creates DISTANCE_TO_CENTER relationships.
        
        Args:
            window_embeddings: Dictionary mapping window_idx -> {
                'embedding': np.ndarray (hidden_dim,),
                'dist_normal': float,
                'dist_anomalous': float,
                'confidence': float
            }
            center_embeddings: (2, hidden_dim) array - [normal_center, anomalous_center]
            gdn_predictions: Optional (num_windows, num_sensors) array for predicted_class
            batch_size: Batch size for window updates
        """
        self.connect()
        
        if len(window_embeddings) == 0:
            print("  ⚠️  No window embeddings to sync")
            return
        
        # Compute mean radii for centers
        normal_distances = [data['dist_normal'] for data in window_embeddings.values()]
        anomalous_distances = [data['dist_anomalous'] for data in window_embeddings.values()]
        normal_mean_radius = float(np.mean(normal_distances)) if normal_distances else 0.085
        anomalous_mean_radius = float(np.mean(anomalous_distances)) if anomalous_distances else 0.138
        
        normal_std_radius = float(np.std(normal_distances)) if normal_distances else 0.03
        anomalous_std_radius = float(np.std(anomalous_distances)) if anomalous_distances else 0.04
        
        with self.driver.session() as session:
            # Create ClassCenter nodes (MERGE for idempotency)
            normal_center_embedding = center_embeddings[0].tolist()
            anomalous_center_embedding = center_embeddings[1].tolist()
            
            session.run("""
                MERGE (c_normal:ClassCenter {class: "normal"})
                SET c_normal.embedding = $embedding,
                    c_normal.mean_radius = $mean_radius
            """, {
                "embedding": normal_center_embedding,
                "mean_radius": normal_mean_radius
            })
            
            session.run("""
                MERGE (c_anomalous:ClassCenter {class: "anomalous"})
                SET c_anomalous.embedding = $embedding,
                    c_anomalous.mean_radius = $mean_radius
            """, {
                "embedding": anomalous_center_embedding,
                "mean_radius": anomalous_mean_radius
            })
            
            print(f"  ✓ Created/updated ClassCenter nodes")
            print(f"    Normal center: mean_radius={normal_mean_radius:.4f}")
            print(f"    Anomalous center: mean_radius={anomalous_mean_radius:.4f}")
            
            # Update Window nodes with embedding properties (batch processing)
            window_indices = sorted(window_embeddings.keys())
            num_windows = len(window_indices)
            
            from tqdm import tqdm
            for batch_start in tqdm(range(0, num_windows, batch_size), desc="Syncing embeddings"):
                batch_end = min(batch_start + batch_size, num_windows)
                batch_indices = window_indices[batch_start:batch_end]
                
                batch_data = []
                for window_idx in batch_indices:
                    data = window_embeddings[window_idx]
                    embedding_list = data['embedding'].tolist() if isinstance(data['embedding'], np.ndarray) else data['embedding']
                    
                    # Determine predicted_class
                    predicted_class = "normal"
                    if gdn_predictions is not None and window_idx < len(gdn_predictions):
                        # Use GDN prediction threshold
                        max_score = float(np.max(gdn_predictions[window_idx]))
                        if max_score > 0.5:
                            predicted_class = "anomalous"
                    else:
                        # Use distance-based prediction
                        if data['dist_normal'] > data['dist_anomalous']:
                            predicted_class = "anomalous"
                    
                    batch_data.append({
                        "idx": int(window_idx),
                        "embedding": embedding_list,
                        "dist_normal": float(data['dist_normal']),
                        "dist_anomalous": float(data['dist_anomalous']),
                        "confidence": float(data['confidence']),
                        "predicted_class": predicted_class
                    })
                
                # Batch update windows
                session.run("""
                    UNWIND $batch AS window_data
                    MATCH (w:Window {idx: window_data.idx})
                    SET w.embedding = window_data.embedding,
                        w.dist_normal = window_data.dist_normal,
                        w.dist_anomalous = window_data.dist_anomalous,
                        w.confidence = window_data.confidence,
                        w.predicted_class = window_data.predicted_class
                """, {"batch": batch_data})
                
                if (batch_start // batch_size + 1) % 5 == 0:
                    print(f"  Processed {batch_end}/{num_windows} windows")
            
            print(f"  ✓ Updated {num_windows} Window nodes with embeddings")
            
            # Create DISTANCE_TO_CENTER relationships
            print("  Creating DISTANCE_TO_CENTER relationships...")
            for batch_start in tqdm(range(0, num_windows, batch_size), desc="Creating distance relationships"):
                batch_end = min(batch_start + batch_size, num_windows)
                batch_indices = window_indices[batch_start:batch_end]
                
                batch_relationships = []
                for window_idx in batch_indices:
                    data = window_embeddings[window_idx]
                    
                    # Normal center relationship
                    z_score_normal = (data['dist_normal'] - normal_mean_radius) / normal_std_radius if normal_std_radius > 0 else 0.0
                    batch_relationships.append({
                        "window_idx": int(window_idx),
                        "center_class": "normal",
                        "distance": float(data['dist_normal']),
                        "z_score": float(z_score_normal)
                    })
                    
                    # Anomalous center relationship
                    z_score_anomalous = (data['dist_anomalous'] - anomalous_mean_radius) / anomalous_std_radius if anomalous_std_radius > 0 else 0.0
                    batch_relationships.append({
                        "window_idx": int(window_idx),
                        "center_class": "anomalous",
                        "distance": float(data['dist_anomalous']),
                        "z_score": float(z_score_anomalous)
                    })
                
                # Batch create relationships
                session.run("""
                    UNWIND $batch AS rel_data
                    MATCH (w:Window {idx: rel_data.window_idx})
                    MATCH (c:ClassCenter {class: rel_data.center_class})
                    MERGE (w)-[d:DISTANCE_TO_CENTER]->(c)
                    SET d.distance = rel_data.distance,
                        d.z_score = rel_data.z_score
                """, {"batch": batch_relationships})
            
            print(f"  ✓ Created DISTANCE_TO_CENTER relationships for {num_windows} windows")

    def sync_similarity_edges_to_neo4j(
        self,
        similarity_edges: List[Tuple[int, int, float, float]],
        window_embeddings: Optional[Dict[int, Dict[str, Any]]] = None,
        batch_size: int = 1000
    ) -> None:
        """
        Sync window similarity edges to Neo4j.
        
        Creates SIMILAR_TO relationships between windows based on embedding similarity.
        
        Args:
            similarity_edges: List of tuples (window_i, window_j, cosine_similarity, euclidean_distance)
            window_embeddings: Optional dict for determining same_class property
            batch_size: Batch size for relationship creation
        """
        self.connect()
        
        if len(similarity_edges) == 0:
            print("  ⚠️  No similarity edges to sync")
            return
        
        # Build predicted_class lookup if available
        predicted_class_lookup = {}
        if window_embeddings is not None:
            for window_idx, data in window_embeddings.items():
                # Determine predicted_class from distances
                if data['dist_normal'] > data['dist_anomalous']:
                    predicted_class_lookup[window_idx] = "anomalous"
                else:
                    predicted_class_lookup[window_idx] = "normal"
        
        with self.driver.session() as session:
            num_edges = len(similarity_edges)
            
            from tqdm import tqdm
            for batch_start in tqdm(range(0, num_edges, batch_size), desc="Syncing similarity edges"):
                batch_end = min(batch_start + batch_size, num_edges)
                batch_edges = similarity_edges[batch_start:batch_end]
                
                batch_data = []
                for window_i, window_j, similarity, distance in batch_edges:
                    # Determine same_class
                    same_class = False
                    if window_i in predicted_class_lookup and window_j in predicted_class_lookup:
                        same_class = (predicted_class_lookup[window_i] == predicted_class_lookup[window_j])
                    else:
                        # Try to get from Neo4j if not in lookup
                        # For now, default to False
                        same_class = False
                    
                    batch_data.append({
                        "idx1": int(window_i),
                        "idx2": int(window_j),
                        "similarity": float(similarity),
                        "distance": float(distance),
                        "same_class": bool(same_class)
                    })
                
                # Batch create relationships using UNWIND
                session.run("""
                    UNWIND $batch AS edge
                    MATCH (w1:Window {idx: edge.idx1})
                    MATCH (w2:Window {idx: edge.idx2})
                    MERGE (w1)-[s:SIMILAR_TO]->(w2)
                    SET s.similarity = edge.similarity,
                        s.distance = edge.distance,
                        s.same_class = edge.same_class
                """, {"batch": batch_data})
            
            print(f"  ✓ Created {num_edges} SIMILAR_TO relationships")


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
