#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
DATA_ROOT="${DATA_ROOT:-data}"
MODEL_PATH="${MODEL_PATH:-facebook/contriever-msmarco}"
NUM_SHARDS="${NUM_SHARDS:-1}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"

python -m preprocess.questions \
    --dataset all \
    --raw-root "$DATA_ROOT/raw_data" \
    --output-root "$DATA_ROOT/processed_data"

for dataset in hotpotqa musique 2wikimultihopqa; do
    python -m preprocess.subsample \
        --dataset "$dataset" \
        --processed-root "$DATA_ROOT/processed_data"

    passages="$DATA_ROOT/embed_ready_data/$dataset.tsv"
    index_dir="$DATA_ROOT/database/contriever_msmarco/$dataset"
    python -m preprocess.passages \
        --dataset "$dataset" \
        --raw-root "$DATA_ROOT/raw_data" \
        --output-path "$passages"

    for ((shard_id=0; shard_id<NUM_SHARDS; shard_id++)); do
        python -m source.run.generate_passage_embeddings \
            --model-path "$MODEL_PATH" \
            --passages "$passages" \
            --output-dir "$index_dir" \
            --shard-id "$shard_id" \
            --num-shards "$NUM_SHARDS" \
            --batch-size "$EMBED_BATCH_SIZE"
    done
    python -m source.run.build_index --embedding-dir "$index_dir"
done
