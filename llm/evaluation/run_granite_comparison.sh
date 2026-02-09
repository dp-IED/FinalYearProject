#!/bin/bash
# Run Granite Models Comparison for Serialised KG Method
# Runs two Serialised KG evals (current vs new model) then eval_llm (markdown + JSON).

set -e

DATASET="${1:-llm/evaluation/shared_dataset/test.npz}"
MODEL_PATH="${2:-checkpoints/stage2_sensor_only_0130_002721.pt}"
LIMIT="${3:-40}"
OUTPUT_DIR="${4:-results}"
CURRENT_MODEL="${5:-granite-4.0-h-micro}"
NEW_MODEL="${6:-granite-4.0-1b-base}"
DEVICE="${7:-cpu}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "Granite Models Comparison - Serialised KG Method"
echo "================================================================================"
echo "Dataset: $DATASET"
echo "GDN Model: $MODEL_PATH"
echo "Limit: $LIMIT samples"
echo "Current Model: $CURRENT_MODEL"
echo "New Model: $NEW_MODEL"
echo "Device: $DEVICE"
echo "================================================================================"
echo ""

[ -f "$DATASET" ] || { echo -e "${RED}Error: Dataset not found: $DATASET${NC}"; exit 1; }
[ -f "$MODEL_PATH" ] || {
    if [ -f "checkpoints/$MODEL_PATH" ]; then MODEL_PATH="checkpoints/$MODEL_PATH"; else
    echo -e "${RED}Error: Model not found: $MODEL_PATH${NC}"; exit 1; fi
}

mkdir -p "$OUTPUT_DIR"

echo "Checking LM Studio..."
if ! curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: LM Studio HTTP server not responding${NC}"
    echo "Ensure LM Studio is running with HTTP server enabled and models loaded."
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Inferred result paths (eval_llm convention: serialised_kg_<model_sanitized>.json)
CUR_SANITIZED=$(echo "$CURRENT_MODEL" | tr '-' '_')
NEW_SANITIZED=$(echo "$NEW_MODEL" | tr '-' '_')
CUR_JSON="$OUTPUT_DIR/serialised_kg_${CUR_SANITIZED}.json"
NEW_JSON="$OUTPUT_DIR/serialised_kg_${NEW_SANITIZED}.json"

echo ""
echo "Running evaluation: $CURRENT_MODEL"
echo "--------------------------------------------------------------------------------"
python llm/evaluation/evaluate_gdn_kg_llm.py \
    --dataset "$DATASET" \
    --model-path "$MODEL_PATH" \
    --output "$CUR_JSON" \
    --model-repo "$CURRENT_MODEL" \
    --limit "$LIMIT" \
    --device "$DEVICE" \
    --no-neo4j-sync

echo ""
echo "Running evaluation: $NEW_MODEL"
echo "--------------------------------------------------------------------------------"
python llm/evaluation/evaluate_gdn_kg_llm.py \
    --dataset "$DATASET" \
    --model-path "$MODEL_PATH" \
    --output "$NEW_JSON" \
    --model-repo "$NEW_MODEL" \
    --limit "$LIMIT" \
    --device "$DEVICE" \
    --no-neo4j-sync

echo ""
echo "Comparing (markdown + JSON)..."
echo "--------------------------------------------------------------------------------"
python evaluations/eval_llm.py compare \
    --method "KG->LLM,$CURRENT_MODEL" \
    --method "KG->LLM,$NEW_MODEL" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Generating HTML chart..."
    echo "--------------------------------------------------------------------------------"
    python llm/evaluation/generate_comparison_html.py "$OUTPUT_DIR/compare.json" "$OUTPUT_DIR/compare.html"
    
    echo ""
    echo -e "${GREEN}================================================================================"
    echo "✓ Comparison complete!"
    echo "================================================================================"
    echo -e "${NC}"
    echo "Results: $OUTPUT_DIR/compare.md  $OUTPUT_DIR/compare.json  $OUTPUT_DIR/compare.html"
else
    echo -e "${RED}Comparison failed with exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
