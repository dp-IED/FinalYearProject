"""
RAG (Retrieval-Augmented Generation) module for rule-based window descriptions.

This module provides rule-based summarization of sensor windows for use in
RAG pipelines and vector database setups.
"""

from llm.rag.rule_based_summarizer import (
    compute_window_features,
    map_features_to_labels,
    generate_window_description,
    generate_all_descriptions,
)
from llm.rag.sensor_config import get_sensor_config, get_normal_range

__all__ = [
    "compute_window_features",
    "map_features_to_labels",
    "generate_window_description",
    "generate_all_descriptions",
    "get_sensor_config",
    "get_normal_range",
]
