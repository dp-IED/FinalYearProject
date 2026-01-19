"""
Neo4j Visualization Utilities

Provides Cypher queries for Neo4j Browser and Python visualization functions
for displaying window analysis data in notebooks.
"""

import neo4j
from typing import Optional, Dict, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Cypher Query Generators (for Neo4j Browser)
# ============================================================================

def get_correlation_network_query(window_idx: int) -> str:
    """
    Generate Cypher query to visualize correlation network for a specific window.
    
    Args:
        window_idx: Window index to visualize
        
    Returns:
        Cypher query string ready to paste into Neo4j Browser
    """
    query = f"""
    // Correlation network for window {window_idx}
    MATCH (w:Window {{label: {window_idx}}})-[:HAS_READING]->(s:Sensor)
    MATCH (s1:Sensor)-[r:CORRELATES_WITH {{window: {window_idx}}}]->(s2:Sensor)
    WHERE s1 IN s OR s2 IN s
    RETURN s1, s2, r, w
    """
    return query.strip()


def get_violations_query(window_idx: int) -> str:
    """
    Generate Cypher query to show relationship violations in a window.
    
    Args:
        window_idx: Window index to visualize
        
    Returns:
        Cypher query string
    """
    query = f"""
    // Relationship violations in window {window_idx}
    MATCH (w:Window {{label: {window_idx}}})
    MATCH (s1:Sensor)-[r:CORRELATES_WITH {{window: {window_idx}}}]->(s2:Sensor)
    WHERE r.is_violation = true
    RETURN w, s1, s2, r
    ORDER BY r.deviation DESC
    """
    return query.strip()


def get_anomaly_propagation_query() -> str:
    """
    Generate Cypher query to visualize anomaly propagation chains across windows.
    
    Returns:
        Cypher query string
    """
    query = """
    // Anomaly propagation chains
    MATCH (s1:Sensor)-[p:PROPAGATES]->(s2:Sensor)
    MATCH (w1:Window {label: p.from_window})
    MATCH (w2:Window {label: p.to_window})
    RETURN s1, s2, p, w1, w2
    ORDER BY p.from_window, p.to_window
    """
    return query.strip()


def get_temporal_evolution_query(sensor1: str, sensor2: str, 
                                 start_window: int, end_window: int) -> str:
    """
    Generate Cypher query to show relationship evolution over time.
    
    Args:
        sensor1: First sensor name
        sensor2: Second sensor name
        start_window: Starting window index
        end_window: Ending window index
        
    Returns:
        Cypher query string
    """
    query = f"""
    // Temporal evolution of relationship between {sensor1} and {sensor2}
    MATCH (s1:Sensor {{name: '{sensor1}'}})
    MATCH (s2:Sensor {{name: '{sensor2}'}})
    MATCH (s1)-[r:CORRELATES_WITH]->(s2)
    WHERE r.window >= {start_window} AND r.window <= {end_window}
    MATCH (w:Window)
    WHERE w.label = r.window
    RETURN s1, s2, r, w
    ORDER BY r.window
    """
    return query.strip()


def get_window_summary_query(window_idx: int) -> str:
    """
    Generate Cypher query for complete window view with sensors, correlations, violations.
    
    Args:
        window_idx: Window index to visualize
        
    Returns:
        Cypher query string
    """
    query = f"""
    // Complete window {window_idx} summary
    MATCH (w:Window {{label: {window_idx}}})
    OPTIONAL MATCH (w)-[hr:HAS_READING]->(s:Sensor)
    OPTIONAL MATCH (s1:Sensor)-[r:CORRELATES_WITH {{window: {window_idx}}}]->(s2:Sensor)
    RETURN w, s, hr, s1, s2, r
    """
    return query.strip()


def get_faulty_sensors_query(window_idx: int) -> str:
    """
    Generate Cypher query to highlight faulty sensors and their relationships.
    
    Args:
        window_idx: Window index to visualize
        
    Returns:
        Cypher query string
    """
    query = f"""
    // Faulty sensors in window {window_idx}
    MATCH (w:Window {{label: {window_idx}}})
    MATCH (w)-[hr:HAS_READING]->(s:Sensor)
    WHERE hr.is_faulty = true
    OPTIONAL MATCH (s)-[r:CORRELATES_WITH {{window: {window_idx}}}]->(s2:Sensor)
    OPTIONAL MATCH (s3:Sensor)-[r2:CORRELATES_WITH {{window: {window_idx}}}]->(s)
    RETURN w, s, hr, s2, r, s3, r2
    """
    return query.strip()


# ============================================================================
# Python Visualization Functions (for notebook display)
# ============================================================================

def visualize_window_network(window_idx: int, driver: neo4j.Driver,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize a window's correlation network using NetworkX and Matplotlib.
    
    Args:
        window_idx: Window index to visualize
        driver: Neo4j driver instance
        figsize: Figure size (width, height)
    """
    with driver.session() as session:
        # Query window data
        result = session.run("""
            MATCH (w:Window {label: $window_idx})
            MATCH (w)-[hr:HAS_READING]->(s:Sensor)
            OPTIONAL MATCH (s1:Sensor)-[r:CORRELATES_WITH {window: $window_idx}]->(s2:Sensor)
            RETURN s, hr, s1, s2, r
        """, window_idx=window_idx)
        
        # Build NetworkX graph
        G = nx.Graph()
        sensor_colors = {}
        sensor_sizes = {}
        
        for record in result:
            # Add sensors
            sensor = record.get('s')
            if sensor:
                sensor_name = sensor['name']
                is_faulty = record.get('hr', {}).get('is_faulty', False)
                anomaly_score = record.get('hr', {}).get('anomaly_score', 0.0)
                
                G.add_node(sensor_name)
                sensor_colors[sensor_name] = 'red' if is_faulty else 'lightblue'
                sensor_sizes[sensor_name] = 500 + (anomaly_score * 1000)
            
            # Add correlations
            s1 = record.get('s1')
            s2 = record.get('s2')
            r = record.get('r')
            
            if s1 and s2 and r:
                s1_name = s1['name']
                s2_name = s2['name']
                is_violation = r.get('is_violation', False)
                correlation = r.get('correlation', 0.0)
                
                if not G.has_edge(s1_name, s2_name):
                    G.add_edge(s1_name, s2_name, 
                             weight=abs(correlation),
                             is_violation=is_violation,
                             correlation=correlation)
        
        # Create visualization
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        
        # Draw edges
        violation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_violation', False)]
        normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_violation', False)]
        
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, 
                             edge_color='gray', alpha=0.3, width=1)
        nx.draw_networkx_edges(G, pos, edgelist=violation_edges,
                             edge_color='orange', alpha=0.7, width=2, style='dashed')
        
        # Draw nodes
        node_colors = [sensor_colors.get(node, 'lightgray') for node in G.nodes()]
        node_sizes = [sensor_sizes.get(node, 300) for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Draw edge labels for violations
        violation_labels = {(u, v): f"{d.get('correlation', 0):.2f}" 
                           for u, v, d in G.edges(data=True) 
                           if d.get('is_violation', False)}
        nx.draw_networkx_edge_labels(G, pos, violation_labels, font_size=7)
        
        plt.title(f"Window {window_idx} Correlation Network\n"
                 f"Red = Faulty, Orange Dashed = Violations", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def visualize_propagation_timeline(propagation_chains: List[Dict],
                                  figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Visualize anomaly propagation timeline.
    
    Args:
        propagation_chains: List of propagation chain dictionaries from KG
        figsize: Figure size
    """
    if not propagation_chains:
        print("No propagation chains to visualize")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = 0
    colors = plt.cm.Set3(np.linspace(0, 1, len(propagation_chains)))
    
    for idx, chain in enumerate(propagation_chains):
        root_sensor = chain.get('root_sensor', 'Unknown')
        root_window = chain.get('root_window', 0)
        timeline = chain.get('propagation_timeline', [])
        
        # Draw root sensor
        ax.scatter(root_window, y_pos, s=500, c=[colors[idx]], 
                  marker='o', edgecolors='black', linewidths=2, zorder=3)
        ax.text(root_window, y_pos, f"  {root_sensor}\n  (root)", 
               fontsize=9, verticalalignment='center')
        
        # Draw propagation timeline
        for entry in timeline:
            window = entry.get('window', 0)
            affected = entry.get('affected_sensors', [])
            
            if window > root_window:
                # Draw line from root to this window
                ax.plot([root_window, window], [y_pos, y_pos], 
                       color=colors[idx], linewidth=2, alpha=0.5, zorder=1)
                
                # Draw affected sensors
                for i, sensor in enumerate(affected):
                    if sensor != root_sensor:
                        offset = (i + 1) * 0.3
                        ax.scatter(window, y_pos + offset, s=300, 
                                 c=[colors[idx]], marker='s', 
                                 edgecolors='black', linewidths=1, zorder=2)
                        ax.text(window, y_pos + offset, f"  {sensor}", 
                               fontsize=8, verticalalignment='center')
        
        y_pos += 1.5
    
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Propagation Chain', fontsize=12)
    ax.set_title('Anomaly Propagation Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_correlation_heatmap(window_idx: int, driver: neo4j.Driver,
                                 figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize correlation heatmap for a window.
    
    Args:
        window_idx: Window index
        driver: Neo4j driver instance
        figsize: Figure size
    """
    with driver.session() as session:
        # Query correlations
        result = session.run("""
            MATCH (s1:Sensor)-[r:CORRELATES_WITH {window: $window_idx}]->(s2:Sensor)
            RETURN s1.name AS sensor1, s2.name AS sensor2, 
                   r.correlation AS correlation,
                   r.is_violation AS is_violation
            ORDER BY s1.name, s2.name
        """, window_idx=window_idx)
        
        # Build correlation matrix
        sensors = set()
        correlations = {}
        
        for record in result:
            s1 = record['sensor1']
            s2 = record['sensor2']
            corr = record['correlation']
            is_violation = record['is_violation']
            
            sensors.add(s1)
            sensors.add(s2)
            correlations[(s1, s2)] = {'correlation': corr, 'is_violation': is_violation}
        
        sensors = sorted(list(sensors))
        n = len(sensors)
        matrix = np.zeros((n, n))
        violation_mask = np.zeros((n, n), dtype=bool)
        
        for i, s1 in enumerate(sensors):
            for j, s2 in enumerate(sensors):
                if (s1, s2) in correlations:
                    matrix[i, j] = correlations[(s1, s2)]['correlation']
                    violation_mask[i, j] = correlations[(s1, s2)]['is_violation']
                elif i == j:
                    matrix[i, j] = 1.0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Overlay violation markers
        violation_positions = np.where(violation_mask)
        for i, j in zip(violation_positions[0], violation_positions[1]):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                      fill=False, edgecolor='orange', 
                                      linewidth=3))
        
        # Set ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sensors, rotation=45, ha='right')
        ax.set_yticklabels(sensors)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        ax.set_title(f'Correlation Heatmap - Window {window_idx}\n'
                    f'Orange boxes = Violations', fontsize=14)
        plt.tight_layout()
        plt.show()


def visualize_violations_over_time(start_window: int, end_window: int,
                                   driver: neo4j.Driver,
                                   figsize: Tuple[int, int] = (14, 6)) -> None:
    """
    Visualize violation patterns across windows.
    
    Args:
        start_window: Starting window index
        end_window: Ending window index
        driver: Neo4j driver instance
        figsize: Figure size
    """
    with driver.session() as session:
        # Query violations over time
        result = session.run("""
            MATCH (s1:Sensor)-[r:CORRELATES_WITH]->(s2:Sensor)
            WHERE r.window >= $start AND r.window <= $end AND r.is_violation = true
            RETURN r.window AS window, 
                   COUNT(*) AS violation_count,
                   AVG(r.deviation) AS avg_deviation,
                   MAX(r.deviation) AS max_deviation
            ORDER BY r.window
        """, start=start_window, end=end_window)
        
        windows = []
        counts = []
        avg_deviations = []
        max_deviations = []
        
        for record in result:
            windows.append(record['window'])
            counts.append(record['violation_count'])
            avg_deviations.append(record['avg_deviation'])
            max_deviations.append(record['max_deviation'])
        
        if not windows:
            print(f"No violations found in windows {start_window} to {end_window}")
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Violation count over time
        ax1.bar(windows, counts, color='orange', alpha=0.7, edgecolor='darkorange')
        ax1.set_ylabel('Number of Violations', fontsize=11)
        ax1.set_title('Relationship Violations Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Deviation magnitude over time
        ax2.plot(windows, avg_deviations, 'o-', label='Average Deviation', 
                color='red', linewidth=2, markersize=6)
        ax2.plot(windows, max_deviations, 's-', label='Max Deviation', 
                color='darkred', linewidth=2, markersize=6)
        ax2.set_xlabel('Window Index', fontsize=11)
        ax2.set_ylabel('Correlation Deviation', fontsize=11)
        ax2.set_title('Violation Severity Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def print_cypher_queries(window_idx: int):
    """
    Convenience function to print all Cypher queries for a window.
    
    Args:
        window_idx: Window index
    """
    print("=" * 80)
    print(f"Cypher Queries for Window {window_idx}")
    print("=" * 80)
    print()
    
    print("1. Correlation Network:")
    print("-" * 80)
    print(get_correlation_network_query(window_idx))
    print()
    
    print("2. Violations:")
    print("-" * 80)
    print(get_violations_query(window_idx))
    print()
    
    print("3. Window Summary:")
    print("-" * 80)
    print(get_window_summary_query(window_idx))
    print()
    
    print("4. Faulty Sensors:")
    print("-" * 80)
    print(get_faulty_sensors_query(window_idx))
    print()
    
    print("5. Anomaly Propagation:")
    print("-" * 80)
    print(get_anomaly_propagation_query())
    print()