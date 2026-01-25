"""Diagnostic utilities to identify why KAG v2 tools return empty results."""

import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from llm.kag.neo4j_queries import Neo4jKAGQueries


class KAGDiagnostics:
    """Diagnose tool execution failures in KAG pipeline."""
    
    def __init__(self, neo4j_queries: Neo4jKAGQueries, sensor_names: List[str], 
                 dataset_path: Optional[str] = None):
        """
        Initialize KAG Diagnostics.
        
        Args:
            neo4j_queries: Neo4jKAGQueries instance for graph queries
            sensor_names: List of sensor names (for label conversion)
            dataset_path: Optional path to dataset for ground truth access
        """
        self.queries = neo4j_queries
        self.sensor_names = sensor_names
        self.dataset_path = dataset_path
        self._dataset_cache = None
    
    def diagnose_window(self, window_idx: int) -> Dict[str, Any]:
        """
        Comprehensive diagnostic for a single window.
        
        Returns:
            dict with keys: kg_exists, gdn_scores, neo4j_data, 
                           tool_results, ground_truth, root_cause
        """
        report = {
            'window_idx': window_idx,
            'timestamp': str(datetime.now())
        }
        
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC REPORT: Window {window_idx}")
        print(f"{'='*70}\n")
        
        # 1. Check if KG exists for this window
        try:
            kg_exists = self._check_kg_exists(window_idx)
            report['kg_exists'] = kg_exists
            print(f"✓ KG exists: {kg_exists}")
        except Exception as e:
            report['kg_exists'] = False
            report['kg_error'] = str(e)
            print(f"✗ KG check failed: {e}")
            return report
        
        # 2. Get raw GDN anomaly scores
        try:
            gdn_scores = self._get_gdn_scores(window_idx)
            report['gdn_scores'] = gdn_scores
            print(f"\n2. GDN Anomaly Scores:")
            if gdn_scores:
                for sensor, score in sorted(gdn_scores.items(), 
                                           key=lambda x: x[1], reverse=True):
                    flag = "🔴" if score > 0.6 else "🟡" if score > 0.4 else "🟢"
                    print(f"   {flag} {sensor}: {score:.4f}")
                
                max_score = max(gdn_scores.values())
                print(f"   Max score: {max_score:.4f}")
                
                if max_score < 0.5:
                    print(f"   ⚠️  All scores below 0.5 - likely NORMAL window")
            else:
                print(f"   ⚠️  No scores found")
            
        except Exception as e:
            report['gdn_error'] = str(e)
            print(f"✗ GDN scores failed: {e}")
            return report
        
        # 3. Query Neo4j directly
        try:
            neo4j_data = self._query_neo4j(window_idx)
            report['neo4j_data'] = neo4j_data
            print(f"\n3. Neo4j Graph Data:")
            print(f"   Nodes: {neo4j_data['node_count']}")
            print(f"   Relationships: {neo4j_data['relationship_count']}")
            if neo4j_data['sample_rels']:
                print(f"   Sample relationships:")
                for rel in neo4j_data['sample_rels'][:5]:
                    print(f"      {rel['source']} --[{rel['type']}]--> {rel['target']}")
            
            if neo4j_data['node_count'] == 0:
                print(f"   ✗ CRITICAL: No nodes in Neo4j for window {window_idx}")
                report['root_cause'] = "Neo4j graph not populated"
                
        except Exception as e:
            report['neo4j_error'] = str(e)
            print(f"✗ Neo4j query failed: {e}")
        
        # 4. Test each tool with different thresholds
        try:
            tool_results = self._test_all_tools(window_idx)
            report['tool_results'] = tool_results
            print(f"\n4. Tool Execution Tests:")
            
            for tool_name, results in tool_results.items():
                print(f"\n   {tool_name}:")
                for threshold, data in results.items():
                    count = len(data) if isinstance(data, list) else 0
                    print(f"      threshold={threshold}: {count} results")
                    if count > 0 and count <= 3:
                        for item in data[:3]:
                            if isinstance(item, dict):
                                if 'sensor' in item:
                                    print(f"         - {item.get('sensor', '?')}: {item.get('score', 0):.3f}")
                                elif 'source' in item:
                                    print(f"         - {item.get('source', '?')} -> {item.get('target', '?')}: dev={item.get('deviation', 0):.3f}")
                            else:
                                print(f"         - {item}")
            
        except Exception as e:
            report['tool_error'] = str(e)
            print(f"✗ Tool testing failed: {e}")
        
        # 5. Compare to ground truth
        try:
            ground_truth = self._get_ground_truth(window_idx)
            if ground_truth:
                report['ground_truth'] = ground_truth
                print(f"\n5. Ground Truth:")
                print(f"   Fault type: {ground_truth.get('fault_type', 'N/A')}")
                print(f"   Faulty sensors: {ground_truth.get('faulty_sensors', [])}")
        except Exception as e:
            report['ground_truth_error'] = str(e)
            print(f"✗ Ground truth retrieval failed: {e}")
        
        # 6. Root cause analysis
        root_cause = self._analyze_root_cause(report)
        report['root_cause'] = root_cause
        print(f"\n6. ROOT CAUSE ANALYSIS:")
        print(f"   {root_cause}")
        print(f"\n{'='*70}\n")
        
        return report
    
    def _check_kg_exists(self, window_idx: int) -> bool:
        """Check if KG was built for this window."""
        try:
            with self.queries.driver.session() as session:
                result = session.run("""
                    MATCH (s:Sensor)
                    WHERE s.window = $window_idx
                    RETURN count(s) as count
                """, window_idx=window_idx)
                record = result.single()
                return record['count'] > 0 if record else False
        except Exception:
            return False
    
    def _get_gdn_scores(self, window_idx: int) -> Dict[str, float]:
        """Get raw GDN anomaly scores."""
        try:
            with self.queries.driver.session() as session:
                result = session.run("""
                    MATCH (s:Sensor)
                    WHERE s.window = $window_idx
                    RETURN s.base_sensor_name AS sensor, s.anomaly_score AS score
                    ORDER BY s.anomaly_score DESC
                """, window_idx=window_idx)
                return {record['sensor']: float(record['score']) for record in result}
        except Exception as e:
            print(f"   Error querying GDN scores: {e}")
            return {}
    
    def _query_neo4j(self, window_idx: int) -> Dict[str, Any]:
        """Query Neo4j for graph structure."""
        try:
            with self.queries.driver.session() as session:
                # Count nodes
                node_result = session.run("""
                    MATCH (s:Sensor)
                    WHERE s.window = $window_idx
                    RETURN count(s) as count
                """, window_idx=window_idx)
                node_record = node_result.single()
                node_count = node_record['count'] if node_record else 0
                
                # Count relationships
                rel_result = session.run("""
                    MATCH (s:Sensor {window: $window_idx})-[r:CORRELATES_WITH]-(t:Sensor {window: $window_idx})
                    RETURN count(r) as count
                """, window_idx=window_idx)
                rel_record = rel_result.single()
                rel_count = rel_record['count'] if rel_record else 0
                
                # Sample relationships
                sample_result = session.run("""
                    MATCH (s:Sensor {window: $window_idx})-[r:CORRELATES_WITH]->(t:Sensor {window: $window_idx})
                    RETURN s.base_sensor_name AS source, 
                           type(r) AS type, 
                           t.base_sensor_name AS target,
                           r.actual_correlation AS actual,
                           r.expected_correlation AS expected
                    LIMIT 10
                """, window_idx=window_idx)
                sample_rels = [dict(record) for record in sample_result]
                
                return {
                    'node_count': node_count,
                    'relationship_count': rel_count,
                    'sample_rels': sample_rels
                }
        except Exception as e:
            return {
                'node_count': 0,
                'relationship_count': 0,
                'sample_rels': [],
                'error': str(e)
            }
    
    def _test_all_tools(self, window_idx: int) -> Dict[str, Dict]:
        """Test all tools with different thresholds."""
        results = {}
        
        # Test get_anomalous_sensors with different thresholds
        results['get_anomalous_sensors'] = {}
        for threshold in [0.3, 0.5, 0.6, 0.7, 0.8]:
            try:
                data = self.queries.get_anomalous_sensors(window_idx, threshold=threshold)
                results['get_anomalous_sensors'][threshold] = data
            except Exception as e:
                results['get_anomalous_sensors'][threshold] = f"Error: {e}"
        
        # Test get_violations
        results['get_violations'] = {}
        for threshold in [0.2, 0.3, 0.4]:
            try:
                data = self.queries.get_violations(window_idx, deviation_threshold=threshold)
                results['get_violations'][threshold] = data
            except Exception as e:
                results['get_violations'][threshold] = f"Error: {e}"
        
        return results
    
    def _get_ground_truth(self, window_idx: int) -> Optional[Dict[str, Any]]:
        """Get ground truth labels for this window."""
        if not self.dataset_path:
            return None
        
        try:
            # Lazy load dataset
            if self._dataset_cache is None:
                import numpy as np
                data = np.load(self.dataset_path, allow_pickle=True)
                self._dataset_cache = {
                    'sensor_labels': data.get('sensor_labels'),
                    'window_labels': data.get('window_labels'),
                    'fault_types': data.get('fault_types', None)
                }
            
            sensor_labels = self._dataset_cache['sensor_labels']
            fault_types = self._dataset_cache.get('fault_types')
            
            if sensor_labels is None or window_idx >= len(sensor_labels):
                return None
            
            window_label = sensor_labels[window_idx]
            faulty_sensors = [
                self.sensor_names[i] 
                for i, label in enumerate(window_label) 
                if label == 1
            ]
            
            fault_type = None
            if fault_types is not None and window_idx < len(fault_types):
                fault_type = fault_types[window_idx]
            
            return {
                'window_label': int(np.sum(window_label > 0)),
                'fault_type': fault_type,
                'faulty_sensors': faulty_sensors
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_root_cause(self, report: Dict) -> str:
        """Analyze diagnostic report to identify root cause."""
        
        # Check 1: KG not built
        if not report.get('kg_exists', False):
            return "KG not built for this window. Check process_for_kg() execution."
        
        # Check 2: No Neo4j data
        neo4j_data = report.get('neo4j_data', {})
        if neo4j_data.get('node_count', 0) == 0:
            return "Neo4j graph empty. KG build succeeded but data not persisted."
        
        # Check 3: GDN scores all low
        gdn_scores = report.get('gdn_scores', {})
        if gdn_scores:
            max_score = max(gdn_scores.values())
            if max_score < 0.5:
                gt = report.get('ground_truth', {})
                if gt.get('fault_type') == 'NORMAL' or gt.get('window_label', 1) == 0:
                    return "Correctly identified NORMAL window (all GDN scores < 0.5)"
                else:
                    return f"GDN failed to detect fault. Ground truth: {gt.get('fault_type', 'Unknown')}"
        
        # Check 4: Tools return empty despite high scores
        tool_results = report.get('tool_results', {})
        anomalous_sensors_05 = tool_results.get('get_anomalous_sensors', {}).get(0.5, [])
        
        if gdn_scores and max(gdn_scores.values()) > 0.6:
            if not anomalous_sensors_05 or len(anomalous_sensors_05) == 0:
                return "Threshold mismatch: GDN scores high but tools use threshold > max score"
        
        # Check 5: No violations despite anomalies
        violations = tool_results.get('get_violations', {}).get(0.3, [])
        if anomalous_sensors_05 and len(anomalous_sensors_05) > 0 and (not violations or len(violations) == 0):
            return "Anomalies detected but no correlation violations. Check violation detection logic."
        
        return "Tools returning data correctly. Issue may be in LLM synthesis step."
    
    def batch_diagnose(self, window_indices: List[int], output_path: Optional[str] = None):
        """Run diagnostics on multiple windows and save report."""
        reports = []
        
        for idx in window_indices:
            report = self.diagnose_window(idx)
            reports.append(report)
        
        # Aggregate statistics
        summary = self._create_summary(reports)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'detailed_reports': reports
                }, f, indent=2)
            print(f"Diagnostic report saved to {output_path}")
        
        return summary, reports
    
    def _create_summary(self, reports: List[Dict]) -> Dict:
        """Create summary statistics from multiple diagnostic reports."""
        total = len(reports)
        
        kg_exists_count = sum(1 for r in reports if r.get('kg_exists', False))
        empty_tools_count = sum(1 for r in reports if self._has_empty_tools(r))
        low_scores_count = sum(1 for r in reports if self._has_low_scores(r))
        
        root_causes = {}
        for report in reports:
            cause = report.get('root_cause', 'Unknown')
            root_causes[cause] = root_causes.get(cause, 0) + 1
        
        return {
            'total_windows': total,
            'kg_exists': kg_exists_count,
            'empty_tool_results': empty_tools_count,
            'low_gdn_scores': low_scores_count,
            'root_cause_distribution': root_causes
        }
    
    def _has_empty_tools(self, report: Dict) -> bool:
        """Check if report shows empty tool results."""
        tool_results = report.get('tool_results', {})
        anomalous = tool_results.get('get_anomalous_sensors', {}).get(0.5, [])
        return not anomalous or len(anomalous) == 0
    
    def _has_low_scores(self, report: Dict) -> bool:
        """Check if all GDN scores are low."""
        scores = report.get('gdn_scores', {})
        return scores and max(scores.values()) < 0.5
