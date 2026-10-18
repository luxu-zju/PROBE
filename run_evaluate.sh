#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
: "${RETRIEVER_CHECKPOINT:?Set RETRIEVER_CHECKPOINT to a trained query encoder.}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-probe}"
DATA_ROOT="${DATA_ROOT:-data}"
PREDICTION_ROOT="${PREDICTION_ROOT:-predictions}"
GENERATION_MODEL_PATH="${GENERATION_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
GENERATION_BACKEND="${GENERATION_BACKEND:-transformers}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2}"

evaluation_args=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then
    evaluation_args+=(--limit "$EVAL_LIMIT")
fi
if [[ "${FAISS_USE_GPU:-0}" == "1" ]]; then
    evaluation_args+=(--faiss-use-gpu)
fi
if [[ "${FAISS_GPU_USE_FLOAT16:-0}" == "1" ]]; then
    evaluation_args+=(--faiss-gpu-use-float16)
fi

for dataset in musique hotpotqa 2wikimultihopqa; do
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" python -m source.run.inference \
        --method probe \
        --dataset "$dataset" \
        --dataset-split test \
        --data-root "$DATA_ROOT" \
        --index-root "$DATA_ROOT/database/contriever_msmarco" \
        --output-root "$PREDICTION_ROOT" \
        --generation-model-name "$GENERATION_MODEL_PATH" \
        --generation-backend "$GENERATION_BACKEND" \
        --retrieval-gpu-id 0 \
        --generation-gpu-ids 1 2 \
        --generation-max-new-tokens 128 \
        --max-num-thought 6 \
        --retrieval-query-model-name-or-path "$RETRIEVER_CHECKPOINT" \
        --running-name "${EXPERIMENT_NAME}_${dataset}" \
        "${evaluation_args[@]}"
done
