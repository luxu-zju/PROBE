#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
: "${PSEUDO_LABEL_PATH:?Set PSEUDO_LABEL_PATH to the relevance-scored training JSONL file.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-probe_margin_mse_gap_0.0001}"
MODEL_PATH="${MODEL_PATH:-facebook/contriever-msmarco}"
GENERATION_MODEL_PATH="${GENERATION_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" python -m source.run.train_retriever \
    --pseudo-label-path "$PSEUDO_LABEL_PATH" \
    --model-path "$MODEL_PATH" \
    --lr 1e-5 \
    --epochs 20 \
    --batch-size 32 \
    --gradient-accumulation-steps 8 \
    --num-workers 8 \
    --min-gap 0.0001 \
    --seed 2026 \
    --output-dir "$OUTPUT_ROOT/$EXPERIMENT_NAME"

if [[ "${RUN_EVALUATION:-1}" == "1" ]]; then
    RETRIEVER_CHECKPOINT="$OUTPUT_ROOT/$EXPERIMENT_NAME/best_model" \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    GENERATION_MODEL_PATH="$GENERATION_MODEL_PATH" \
    DATA_ROOT="$DATA_ROOT" \
    bash run_evaluate.sh
fi
