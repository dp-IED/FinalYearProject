"""
Show an example of what the KG-enhanced LLM (KAG) sees in the prompt.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add paths for imports (matching evaluate_gdn_kg_llm.py)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'anomaly-detection'))
sys.path.insert(0, str(project_root))

from gdn_processor import GDNPredictor
from llm.helpers.KG import KnowledgeGraphBuilder
from llm.evaluation.evaluate_gdn_kg_llm import (
    extract_window_kg_context,
    format_kg_context_for_llm,
    format_window_with_kg_for_llm
)

def main():
    print("="*80)
    print("EXAMPLE: What the KG-Enhanced LLM (KAG) Sees")
    print("="*80)
    print()

    # Load dataset
    dataset_path = Path('llm/evaluation/shared_dataset/test_50.npz')
    data = np.load(dataset_path, allow_pickle=True)

    with open('llm/evaluation/shared_dataset/test_50_metadata.json', 'r') as f:
        metadata = json.load(f)

    sensor_names = metadata['dataset_info']['sensor_names']
    normalized_windows = data['normalized_windows']
    unnormalized_windows = data['unnormalized_windows']
    statistical_features = data['statistical_features'] if 'statistical_features' in data else None

    # Pick a window that has faults (window 0 should have VSS_DRO)
    window_idx = 0
    print(f"Analyzing Window {window_idx} (should have VSS_DRO fault)")
    print()

    # Initialize GDN Predictor
    print("Building KG...")
    model_path = Path('anomaly-detection/best_center_loss_gdn.pt')
    predictor = GDNPredictor(
        model_path=model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=32,
        top_k=3,
        hidden_dim=32,
        device='cpu'
    )

    # Process data for KG
    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows[:50],
        sensor_labels=data['sensor_labels'][:50],
        window_labels=data['window_labels'][:50],
        batch_size=32
    )

    # Build Knowledge Graph
    kg_builder = KnowledgeGraphBuilder(
        sensor_names=kg_data['sensor_names'],
        sensor_embeddings=kg_data['sensor_embeddings'],
        adjacency_matrix=kg_data['adjacency_matrix']
    )

    # Build KG from GDN windows
    kg = kg_builder.build_from_gdn_windows(
        kg_data['X_windows'],
        kg_data['sensor_labels'],
        kg_data['window_labels']
    )

    print("✓ KG built")
    print()

    # Extract KG context for this window
    kg_context = extract_window_kg_context(kg_builder, window_idx, temporal_context_windows=2)

    # Show KG context structure
    print("="*80)
    print("KG CONTEXT STRUCTURE:")
    print("="*80)
    print(f"Faulty entities: {len([e for e in kg_context['entities'] if e.get('is_faulty')])}")
    print(f"Total relationships: {len(kg_context['relationships'])}")
    print(f"Violations: {len(kg_context['violations'])}")
    print(f"Temporal context windows: {len(kg_context['temporal_context'])}")
    print(f"Anomaly propagation chains: {len(kg_context['anomaly_propagation'])}")
    print()

    # Format KG context
    kg_section = format_kg_context_for_llm(kg_context, window_idx, kg_builder)

    print("="*80)
    print("KG CONTEXT SECTION (as formatted for LLM):")
    print("="*80)
    print()
    print(kg_section)
    print()

    # Get window data
    window_data = unnormalized_windows[window_idx]
    stats = statistical_features[window_idx] if statistical_features is not None else None

    # Format complete prompt
    full_prompt = format_window_with_kg_for_llm(
        window_data, sensor_names, kg_context, window_idx, kg_builder,
        stats, use_statistical_features=True
    )

    print("="*80)
    print("COMPLETE PROMPT SENT TO LLM:")
    print("="*80)
    print()
    print(full_prompt)
    print()
    print("="*80)
    print(f"Total prompt length: {len(full_prompt)} characters")
    print(f"Approximate tokens: ~{len(full_prompt.split())} words")
    print("="*80)

if __name__ == '__main__':
    main()
