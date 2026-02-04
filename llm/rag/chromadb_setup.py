"""
ChromaDB setup and management for storing window descriptions and embeddings.

This module provides functions to create ChromaDB collections, add window data
(text descriptions and GDN embeddings), and query similar windows for RAG.
"""

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import json
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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

        # Use text-only embeddings (ChromaDB will generate from text)

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


def delete_collection(
    collection_name: str,
    persist_directory: Optional[Path] = None,
) -> None:
    """
    Delete a ChromaDB collection.

    Args:
        collection_name: Name of the collection to delete
        persist_directory: Directory where ChromaDB is persisted
    """
    if persist_directory is None:
        persist_directory = project_root / "chromadb"
    else:
        persist_directory = Path(persist_directory)

    client = chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(name=collection_name)
        print(f"  ✓ Deleted collection: {collection_name}")
    except Exception as e:
        print(f"  ⚠️  Error deleting collection: {e}")
