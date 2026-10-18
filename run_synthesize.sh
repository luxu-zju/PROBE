#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
: "${PSEUDO_LABEL_PATH:?Set PSEUDO_LABEL_PATH to the output JSONL path.}"

DATASET="${DATASET:-hotpotqa}"
DATA_ROOT="${DATA_ROOT:-data}"
MODEL_PATH="${MODEL_PATH:-facebook/contriever-msmarco}"
GENERATION_MODEL_PATH="${GENERATION_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
RERANKER_MODEL_PATH="${RERANKER_MODEL_PATH:-BAAI/bge-reranker-v2-m3}"
SYNTHESIS_NAME="${SYNTHESIS_NAME:-probe_offline_synthesis}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export CUDA_VISIBLE_DEVICES

python -m source.run.synthesize_training_data \
    --dataset "$DATASET" \
    --output-path "$PSEUDO_LABEL_PATH" \
    --running-name "$SYNTHESIS_NAME" \
    --data-root "$DATA_ROOT" \
    --index-root "$DATA_ROOT/database/contriever_msmarco" \
    --generation-model-name "$GENERATION_MODEL_PATH" \
    --retrieval-model-name-or-path "$MODEL_PATH" \
    --reranker-model-name-or-path "$RERANKER_MODEL_PATH" \
    --batch-size 16 \
    --generation-batch-size-per-device 16 \
    --retrieval-batch-size 32 \
    --reranker-batch-size 32 \
    --generation-max-new-tokens 128 \
    --retrieval-count 8 \
    --retrieval-buffer-size 32 \
    --max-num-thought 6 \
    --retrieval-gpu-id 0 \
    --generation-gpu-ids 1 2 \
    "$@"
