"""
CLI script to populate ChromaDB with window descriptions and GDN embeddings.

Usage:
    python llm/rag/populate_chromadb.py \
        --dataset llm/evaluation/shared_dataset/test.npz \
        --gdn-model anomaly-detection/best_focal_multilabel_gdn.pt \
        --collection-name window_descriptions \
        --persist-dir chromadb
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import json
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.rag.chromadb_setup import (
    create_chromadb_collection,
    add_windows_to_chromadb,
    get_collection_stats,
)
from llm.rag.rule_based_summarizer import generate_all_descriptions

# Add anomaly-detection to path for GDN processor
sys.path.insert(0, str(project_root / "anomaly-detection"))
from gdn_processor import GDNPredictor


def extract_gdn_embeddings(
    dataset_path: Path,
    gdn_model_path: Path,
    batch_size: int = 32,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> np.ndarray:
    """
    Extract GDN embeddings for all windows in the dataset.

    Args:
        dataset_path: Path to .npz dataset file
        gdn_model_path: Path to GDN model checkpoint
        batch_size: Batch size for inference
        device: Device to run on ('cuda' or 'cpu')
        limit: Optional limit on number of windows to process

    Returns:
        numpy array of embeddings (num_windows, embed_dim)
    """
    print("=" * 80)
    print("Extracting GDN Embeddings")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {gdn_model_path}")
    print()

    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        # Fallback to default sensor names
        sensor_names = [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]

    num_windows = normalized_windows.shape[0]
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Load GDN model
    print("Loading GDN model...")
    try:
        predictor = GDNPredictor(
            model_path=gdn_model_path,
            sensor_names=sensor_names,
            device=device,
        )
        print("  ✓ Model loaded successfully")
        print()
    except Exception as e:
        raise RuntimeError(f"Failed to load GDN model: {e}")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = predictor.get_corr_embedding(normalized_windows, batch_size=batch_size)

    print(f"  ✓ Extracted embeddings: shape {embeddings.shape}")
    print()

    return embeddings


def prepare_metadata(
    dataset_path: Path, limit: Optional[int] = None
) -> Dict[int, Dict[str, any]]:
    """
    Prepare metadata for windows from dataset.

    Args:
        dataset_path: Path to .npz dataset file
        limit: Optional limit on number of windows

    Returns:
        Dictionary mapping window_idx to metadata dict
    """
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels = data.get("sensor_labels", None)
    window_labels = data.get("window_labels", None)
    fault_types = data.get("fault_types", None)

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]

    num_windows = len(sensor_labels) if sensor_labels is not None else 0
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)

    metadata_dict = {}
    for window_idx in range(num_windows):
        window_metadata = {
            "sensor_names": ",".join(sensor_names),
        }

        if sensor_labels is not None and window_idx < len(sensor_labels):
            # Find faulty sensors
            faulty_sensors = np.where(sensor_labels[window_idx] > 0)[0]
            if len(faulty_sensors) > 0:
                window_metadata["has_fault"] = True
                window_metadata["faulty_sensor_indices"] = ",".join(
                    [str(i) for i in faulty_sensors]
                )
                window_metadata["faulty_sensor_names"] = ",".join(
                    [sensor_names[i] for i in faulty_sensors]
                )
            else:
                window_metadata["has_fault"] = False

        if window_labels is not None and window_idx < len(window_labels):
            window_metadata["window_label"] = int(window_labels[window_idx])

        if fault_types is not None and window_idx < len(fault_types):
            fault_type = fault_types[window_idx]
            if fault_type is not None:
                window_metadata["fault_type"] = str(fault_type)

        metadata_dict[window_idx] = window_metadata

    return metadata_dict


def main():
    parser = argparse.ArgumentParser(
        description="Populate ChromaDB with window descriptions and GDN embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to .npz dataset file",
    )
    parser.add_argument(
        "--gdn-model",
        type=str,
        required=True,
        help="Path to GDN model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="window_descriptions",
        help="Name of ChromaDB collection (default: window_descriptions)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Directory to persist ChromaDB (default: ./chromadb)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GDN inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on ('cuda' or 'cpu', default: cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Skip generating descriptions (use existing)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip extracting GDN embeddings",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    gdn_model_path = Path(args.gdn_model)
    if not gdn_model_path.exists():
        raise FileNotFoundError(f"GDN model not found: {gdn_model_path}")

    persist_dir = Path(args.persist_dir) if args.persist_dir else None

    print("=" * 80)
    print("Populating ChromaDB")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {gdn_model_path}")
    print(f"Collection: {args.collection_name}")
    if persist_dir:
        print(f"Persist Directory: {persist_dir}")
    print()

    # Step 1: Generate window descriptions
    descriptions = {}
    if not args.skip_descriptions:
        print("Step 1: Generating window descriptions...")
        descriptions = generate_all_descriptions(
            dataset_path=dataset_path,
            output_dir=None,  # Don't save to disk, just get descriptions
            save_index=False,
        )
        if args.limit is not None and args.limit > 0:
            # Limit descriptions
            descriptions = {
                idx: desc for idx, desc in descriptions.items() if idx < args.limit
            }
        print(f"  ✓ Generated {len(descriptions)} descriptions")
        print()
    else:
        print("Step 1: Skipping description generation")
        print()

    # Step 2: Prepare metadata
    print("Step 2: Preparing metadata...")
    metadata = prepare_metadata(dataset_path=dataset_path, limit=args.limit)
    print(f"  ✓ Prepared metadata for {len(metadata)} windows")
    print()

    # Step 3: Create ChromaDB collection and add data
    print("Step 3: Adding to ChromaDB...")
    collection = create_chromadb_collection(
        collection_name=args.collection_name,
        persist_directory=persist_dir,
    )

    # Ensure we have descriptions for all windows
    if not descriptions:
        # Try to load from existing descriptions directory
        descriptions_dir = dataset_path.parent / "descriptions" / "descriptions"
        if descriptions_dir.exists():
            print(f"  Loading descriptions from {descriptions_dir}...")
            for window_idx in sorted(metadata.keys()):
                desc_file = descriptions_dir / f"window_{window_idx:05d}.txt"
                if desc_file.exists():
                    with open(desc_file, "r") as f:
                        descriptions[window_idx] = f.read()
            print(f"  ✓ Loaded {len(descriptions)} descriptions from disk")
        else:
            raise ValueError(
                "No descriptions available. Run without --skip-descriptions or ensure descriptions exist."
            )

    add_windows_to_chromadb(
        collection=collection,
        descriptions=descriptions,
        metadata=metadata,
    )

    # Step 4: Print statistics
    print()
    print("Step 4: Collection statistics...")
    stats = get_collection_stats(collection)
    print(
        f"  ✓ Collection '{stats['collection_name']}' contains {stats['num_windows']} windows"
    )
    print()

    print("=" * 80)
    print("✓ ChromaDB population complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
