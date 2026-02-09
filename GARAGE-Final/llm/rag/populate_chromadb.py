"""
CLI script to populate ChromaDB with window descriptions and GDN embeddings.

CONSOLIDATED: Merges chromadb_setup functions and uses GDN-enhanced descriptions.

Usage:
    python llm/rag/populate_chromadb.py \
        --dataset data/shared_dataset/test.npz \
        --gdn-model checkpoints/stage2_best.pt \
        --collection-name window_descriptions \
        --persist-dir chromadb
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings

from llm.rag.rule_based_summarizer import generate_all_descriptions
from llm.gdn_processor import GDNPredictor

# Optional: Import KnowledgeGraph for enhanced context
try:
    from kg.create_kg import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None


# ============================================================================
# ChromaDB Setup Functions (merged from chromadb_setup.py)
# ============================================================================

def create_chromadb_collection(
    collection_name: str = "window_descriptions",
    persist_directory: Optional[Path] = None,
    embedding_function=None,
) -> chromadb.Collection:
    """
    Create or get a ChromaDB collection for window data.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist the database (default: ./chromadb)
        embedding_function: Optional embedding function for text (default: uses ChromaDB default)

    Returns:
        ChromaDB Collection object
    """
    if persist_directory is None:
        persist_directory = project_root / "chromadb"
    else:
        persist_directory = Path(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)

    # Create ChromaDB client with persistence
    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )

    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"  ✓ Found existing collection: {collection_name}")
    except Exception:
        # Collection doesn't exist, create it
        if embedding_function is None:
            # Use default embedding function
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Window descriptions and GDN embeddings for RAG"
                },
            )
        else:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={
                    "description": "Window descriptions and GDN embeddings for RAG"
                },
            )
        print(f"  ✓ Created new collection: {collection_name}")

    return collection


def add_windows_to_chromadb(
    collection: chromadb.Collection,
    descriptions: Dict[int, str],
    metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    batch_size: int = 100,
) -> None:
    """
    Add windows to ChromaDB collection.

    Args:
        collection: ChromaDB collection to add to
        descriptions: Dictionary mapping window_idx to description text
        metadata: Optional dictionary mapping window_idx to metadata dict
        batch_size: Batch size for adding documents (default: 100)
    """
    num_windows = len(descriptions)
    print(f"  Adding {num_windows} windows to ChromaDB...")

    # Prepare data in batches
    all_ids = []
    all_documents = []
    all_metadatas = []

    for window_idx in sorted(descriptions.keys()):
        window_id = f"window_{window_idx}"
        description = descriptions[window_idx]

        all_ids.append(window_id)
        all_documents.append(description)

        # Prepare metadata
        window_metadata = {"window_idx": window_idx}
        if metadata and window_idx in metadata:
            window_metadata.update(metadata[window_idx])
        all_metadatas.append(window_metadata)

    # Add in batches
    num_batches = (num_windows + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_windows)

        batch_ids = all_ids[start_idx:end_idx]
        batch_documents = all_documents[start_idx:end_idx]
        batch_metadatas = all_metadatas[start_idx:end_idx]

        # Use text-only (ChromaDB will generate embeddings from text)
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(
                f"    Added batch {batch_idx + 1}/{num_batches} ({end_idx}/{num_windows} windows)"
            )

    print(f"  ✓ Added {num_windows} windows to ChromaDB")


def query_similar_windows(
    collection: chromadb.Collection,
    query_text: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for similar windows.

    Args:
        collection: ChromaDB collection to query
        query_text: Text query (will be embedded by ChromaDB)
        query_embedding: Optional pre-computed embedding vector
        top_k: Number of results to return
        where: Optional metadata filter (e.g., {"window_idx": 0})

    Returns:
        List of dictionaries with keys: id, document, metadata, distance
    """
    if query_text is None and query_embedding is None:
        raise ValueError("Either query_text or query_embedding must be provided")

    if query_text is not None:
        # Text-based query
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )
    else:
        # Embedding-based query
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
        )

    # Format results
    formatted_results = []
    if results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i]
                if results["documents"]
                else None,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i]
                if results["distances"]
                else None,
            }
            formatted_results.append(result)

    return formatted_results


def get_collection_stats(collection: chromadb.Collection) -> Dict[str, Any]:
    """
    Get statistics about a ChromaDB collection.

    Args:
        collection: ChromaDB collection

    Returns:
        Dictionary with collection statistics
    """
    count = collection.count()
    return {
        "num_windows": count,
        "collection_name": collection.name,
    }


# ============================================================================
# GDN Embedding Extraction
# ============================================================================

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
            model_path=str(gdn_model_path),
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


# ============================================================================
# Metadata Preparation
# ============================================================================

def prepare_metadata(
    dataset_path: Path, limit: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
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


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Populate ChromaDB with GDN-enhanced window descriptions"
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
        "--kg-path",
        type=str,
        default=None,
        help="Optional path to KG pickle file for enhanced context",
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
    print("Populating ChromaDB with GDN-Enhanced Descriptions")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {gdn_model_path}")
    print(f"Collection: {args.collection_name}")
    if persist_dir:
        print(f"Persist Directory: {persist_dir}")
    print()

    # Load dataset to get sensor names
    data = np.load(dataset_path, allow_pickle=True)
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
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]

    # Step 1: Load GDN model and get predictions
    print("Step 1: Loading GDN model and generating predictions...")
    predictor = GDNPredictor(
        model_path=str(gdn_model_path),
        sensor_names=sensor_names,
        device=args.device,
    )
    
    # Get normalized windows for predictions
    normalized_windows = data["normalized_windows"]
    if args.limit is not None and args.limit > 0:
        normalized_windows = normalized_windows[:args.limit]
    
    gdn_predictions = predictor.predict(normalized_windows, batch_size=args.batch_size)
    print(f"  ✓ Generated GDN predictions: shape {gdn_predictions.shape}")
    print()

    # Step 2: Get distribution thresholds (if KG available)
    distribution_thresholds = None
    kg = None
    if args.kg_path and Path(args.kg_path).exists():
        try:
            import pickle
            with open(args.kg_path, 'rb') as f:
                kg = pickle.load(f)
            if hasattr(kg, 'distribution_thresholds'):
                distribution_thresholds = kg.distribution_thresholds
            print(f"  ✓ Loaded KG from {args.kg_path}")
            if distribution_thresholds:
                print(f"  ✓ Using distribution thresholds from KG")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load KG: {e}")
    print()

    # Step 3: Generate window descriptions (GDN-enhanced)
    descriptions = {}
    if not args.skip_descriptions:
        print("Step 2: Generating GDN-enhanced window descriptions...")
        descriptions = generate_all_descriptions(
            dataset_path=dataset_path,
            gdn_predictions=gdn_predictions,
            distribution_thresholds=distribution_thresholds,
            kg=kg,
            output_dir=None,  # Don't save to disk, just get descriptions
            save_index=False,
        )
        if args.limit is not None and args.limit > 0:
            # Limit descriptions
            descriptions = {
                idx: desc for idx, desc in descriptions.items() if idx < args.limit
            }
        print(f"  ✓ Generated {len(descriptions)} GDN-enhanced descriptions")
        print()
    else:
        print("Step 2: Skipping description generation")
        print()

    # Step 4: Prepare metadata
    print("Step 3: Preparing metadata...")
    metadata = prepare_metadata(dataset_path=dataset_path, limit=args.limit)
    print(f"  ✓ Prepared metadata for {len(metadata)} windows")
    print()

    # Step 5: Create ChromaDB collection and add data
    print("Step 4: Adding to ChromaDB...")
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

    # Step 6: Print statistics
    print()
    print("Step 5: Collection statistics...")
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
