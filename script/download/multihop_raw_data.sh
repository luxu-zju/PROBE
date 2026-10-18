#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
DATA_ROOT="${DATA_ROOT:-data}"
RAW_ROOT="$DATA_ROOT/raw_data"
TEMP_ROOT="${TMPDIR:-/tmp}/probe_download"
mkdir -p "$RAW_ROOT" "$TEMP_ROOT"

mkdir -p "$RAW_ROOT/hotpotqa"
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json \
    -O "$RAW_ROOT/hotpotqa/hotpot_train_v1.1.json"
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \
    -O "$RAW_ROOT/hotpotqa/hotpot_dev_distractor_v1.json"
wget -c https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 \
    -O "$TEMP_ROOT/hotpot_wikipedia.tar.bz2"
if [[ ! -d "$RAW_ROOT/hotpotqa/wikipedia-paragraphs" ]]; then
    tar -xjf "$TEMP_ROOT/hotpot_wikipedia.tar.bz2" -C "$RAW_ROOT/hotpotqa"
    mv "$RAW_ROOT/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts" \
        "$RAW_ROOT/hotpotqa/wikipedia-paragraphs"
fi

mkdir -p "$RAW_ROOT/2wikimultihopqa"
wget -c "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1" \
    -O "$TEMP_ROOT/2wikimultihopqa.zip"
unzip -joq "$TEMP_ROOT/2wikimultihopqa.zip" -d "$RAW_ROOT/2wikimultihopqa"

mkdir -p "$RAW_ROOT/musique"
gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h" -O "$TEMP_ROOT/musique.zip"
unzip -joq "$TEMP_ROOT/musique.zip" -d "$RAW_ROOT/musique"

echo "Raw datasets downloaded to $RAW_ROOT"
