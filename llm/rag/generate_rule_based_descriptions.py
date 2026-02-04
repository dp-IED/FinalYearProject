"""
CLI script to generate rule-based window descriptions for RAG/vector DB setup.

Usage:
    python llm/rag/generate_rule_based_descriptions.py \
        --dataset llm/evaluation/shared_dataset/test.npz \
        --output-dir llm/evaluation/shared_dataset/descriptions \
        --save-index
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.rag.rule_based_summarizer import generate_all_descriptions


def main():
    parser = argparse.ArgumentParser(
        description="Generate rule-based window descriptions for RAG/vector DB setup"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to .npz dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: dataset_path.parent / 'descriptions')",
    )
    parser.add_argument(
        "--save-index",
        action="store_true",
        default=True,
        help="Also save JSON index file with metadata (default: True)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    print("=" * 80)
    print("Generating Rule-Based Window Descriptions")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"Save index: {args.save_index}")
    print()

    descriptions = generate_all_descriptions(
        dataset_path=dataset_path,
        output_dir=output_dir,
        save_index=args.save_index,
    )

    print()
    print("=" * 80)
    print(f"✓ Successfully generated {len(descriptions)} descriptions")
    print("=" * 80)


if __name__ == "__main__":
    main()
