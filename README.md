# PROBE

[EMNLP 2026] Official implementation of "PROBE: Prospective Retrieval with Oracle-Based End-Goal Distillation for Multi-Hop Question Answering"

PROBE is a multi-hop question answering pipeline with offline retriever
distillation. This repository includes data preparation, pseudo-label synthesis,
retriever training, iterative inference, and official answer evaluation on
HotpotQA, MuSiQue, and 2WikiMultiHopQA.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Offline pseudo-label synthesis](#offline-pseudo-label-synthesis)
- [Retriever training](#retriever-training)
- [Inference and evaluation](#inference-and-evaluation)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

The workflow has three stages:

1. Prepare benchmark questions and passage collections, then build dense indexes
   with Contriever.
2. Generate offline multi-hop trajectories and score retrieved candidates with
   a BGE reranker; train the query encoder from this supervision.
3. Run iterative retrieval and generation with the trained encoder, then report
   official answer EM/F1 and supporting-title recall.

The release includes source code, prompt templates, and scripts for these
stages. Datasets, model weights, generated indexes, pseudo-labels, and trained
checkpoints are obtained or created by following the instructions below; they
are not bundled in the repository.

## Installation

```bash
git clone https://github.com/luxu-zju/PROBE.git
cd PROBE
conda create -n probe python=3.10
conda activate probe
pip install -r requirements.txt "transformers==4.57.1"
```

Install a CUDA-compatible PyTorch build for your hardware. The download scripts
also require `wget`, `tar`, and `unzip`. You need Hugging Face access to
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
After obtaining access to the gated Llama model, authenticate on the machine
that will download the weights:

```bash
hf auth login
```

All model paths below accept either a Hugging Face model ID or a local directory.

Transformers is the default generation backend. The standard synthesis and
evaluation setup uses three visible GPUs: GPU 0 for the query encoder (and the
reranker during synthesis), and GPUs 1 and 2 for two full replicas of the
generation model. Each generation GPU must fit a complete model and its KV
cache. Retriever training uses one GPU. FAISS search runs on CPU by default.

## Data Preparation

```bash
export DATA_ROOT="$PWD/data"
export MODEL_PATH=facebook/contriever-msmarco

# Download questions, passage corpora, and official evaluation gold files.
bash script/download/multihop_raw_data.sh

# Preprocess questions and passages, encode passages, and build FAISS indexes.
CUDA_VISIBLE_DEVICES=0 bash script/download/build.sh
```

The preprocessing scripts create a fixed 100-question development subset and
a disjoint 500-question evaluation subset for each benchmark using seed
`13370`. Both are sampled from the public development data; the 500-question
subset is stored as `test_subsampled.jsonl` and evaluated against the official
development gold answers. Passage collections follow each benchmark's corpus
construction, including unlabeled test passages where applicable.

The resulting layout is:

```text
data/
  raw_data/<dataset>/
  processed_data/<dataset>/
    train.jsonl
    dev.jsonl
    dev_subsampled.jsonl
    test_subsampled.jsonl
  embed_ready_data/<dataset>.tsv
  database/contriever_msmarco/<dataset>/
    index.faiss
    docstore.db
    faiss_id_to_docstore_id.pkl
```

Set `EMBED_BATCH_SIZE` to reduce passage-encoding memory use. `NUM_SHARDS`
controls the number of sequentially encoded shards. Building the full HotpotQA
Wikipedia index requires substantial RAM and disk space. The full HotpotQA
FAISS index alone is approximately 16 GB; embedding shards, the document store,
and raw corpora require additional space. Set `DATA_ROOT` to a sufficiently
large volume before downloading or building the data. Public downloads require
network access to the original hosting services.

## Offline Pseudo-Label Synthesis

```bash
export DATA_ROOT="$PWD/data"
export MODEL_PATH=facebook/contriever-msmarco
export GENERATION_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export RERANKER_MODEL_PATH=BAAI/bge-reranker-v2-m3
export PSEUDO_LABEL_PATH="$DATA_ROOT/probe_supervision_15000.jsonl"

CUDA_VISIBLE_DEVICES=0,1,2 bash run_synthesize.sh \
  --start-index 0 --limit 15000
```

Synthesis selects the first 15,000 sorted HotpotQA training question IDs and
shuffles their processing order with seed `100`. The retriever and reranker
remain in evaluation mode, and generation uses greedy decoding
(`temperature=0`, `do_sample=False`). The prompt templates are
[`answer_gen.txt`](prompts/prompt_set__1/answer_gen.txt) and
[`thought_gen_new.txt`](prompts/prompt_set__1/thought_gen_new.txt).

For each retrieval hop, the query consists of the original question and the
accumulated thought/next-hop-query pairs. Contriever retrieves 128 candidates;
the BGE cross-encoder reranks them and retains 32. The top 8 passages are
provided to the generation model, and all 32 candidates and their BGE scores
are saved for distillation.

`--max-num-thought 6` means at most six thought-generation rounds **after the
initial retrieval**, allowing at most seven retrieval/answer rounds. A
non-Unknown answer ends the trajectory early once at least one thought round
has been performed, as required by the default `probe` method. Each JSONL record stores
`question_id`, `step_idx`, `query`, `thoughts`, `next_hop_queries`,
`model_answer`, `is_final_step`, and the scored candidate list `details`.

To continue an interrupted run, use the same selection, models, settings, and
output path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash run_synthesize.sh \
  --start-index 0 --limit 15000 --resume
```

Completed questions are skipped. Records from incomplete questions are
removed atomically before those questions are regenerated, avoiding duplicate
hop records. Use a new output file when changing synthesis settings.

## Retriever Training

```bash
export DATA_ROOT="$PWD/data"
export MODEL_PATH=facebook/contriever-msmarco
export GENERATION_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export PSEUDO_LABEL_PATH="$DATA_ROOT/probe_supervision_15000.jsonl"
export EXPERIMENT_NAME=probe_margin_mse_gap_0.0001

export TRAIN_CUDA_VISIBLE_DEVICES=0
export EVAL_CUDA_VISIBLE_DEVICES=0,1,2
bash run_train.sh
```

Training keeps non-Unknown final-answer supervision and discards unresolved
intermediate-hop records. A candidate set is retained only if the gap between
its two highest BGE scores is at least `0.0001`. The query encoder is optimized
with pairwise Margin-MSE over the entire saved candidate set; the passage
encoder remains frozen in evaluation mode.

The settings in `run_train.sh` are:

| Parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Epochs | `20` |
| Batch size | `32` |
| Gradient accumulation | `8` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| Maximum gradient norm | `1.0` |
| BGE confidence gap | `0.0001` |

The trainer preserves the per-epoch warmup/cosine scheduler and unscaled
gradient accumulation used in the implementation. `best_model` is selected by
the lowest cumulative mean training loss. Checkpoints are written under
`checkpoints/<experiment>/`, or under `OUTPUT_ROOT` when set.

After training, the script evaluates `best_model` on all three benchmarks.
To train only:

```bash
RUN_EVALUATION=0 bash run_train.sh
```

## Inference and Evaluation

Evaluate an existing query-encoder checkpoint on all three benchmarks:

```bash
export DATA_ROOT="$PWD/data"
export GENERATION_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export EXPERIMENT_NAME=probe_margin_mse_gap_0.0001
export RETRIEVER_CHECKPOINT="$PWD/checkpoints/$EXPERIMENT_NAME/best_model"
export EVAL_CUDA_VISIBLE_DEVICES=0,1,2
bash run_evaluate.sh
```

To check the evaluation setup on one question per benchmark before a full run:

```bash
EVAL_LIMIT=1 bash run_evaluate.sh
```

For a single benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m source.run.inference \
  --method probe \
  --dataset hotpotqa \
  --dataset-split test \
  --data-root "$DATA_ROOT" \
  --index-root "$DATA_ROOT/database/contriever_msmarco" \
  --retrieval-query-model-name-or-path "$RETRIEVER_CHECKPOINT" \
  --generation-model-name "$GENERATION_MODEL_PATH" \
  --generation-backend transformers \
  --retrieval-gpu-id 0 \
  --generation-gpu-ids 1 2 \
  --generation-max-new-tokens 128 \
  --max-num-thought 6 \
  --running-name probe_hotpotqa
```

Evaluation uses the trained retriever without the cross-encoder. The top 8
retrieved passages are used for generation. At the thought limit, if no known
answer was generated, the inference pipeline returns the accumulated thoughts
and next-hop queries as its fallback prediction.

Outputs are saved to `predictions/<dataset>/<running-name>/`:

- `test_prediction.json`: final predictions.
- `test_logging.json`: per-question retrieval and generation logs.
- `test_official_evaluation.json`: official answer EM and token-level F1.
- `test_official_evaluation_with_mhr.json`: answer metrics plus cumulative
  supporting-title recall at 8 passages per hop.

Answer EM/F1 are macro-averaged over the selected questions and reported on a
0–1 scale. The official 2Wiki and MuSiQue evaluators include answer aliases.
These are answer metrics; supporting-fact and joint scores are not reported.

`PREDICTION_ROOT` changes the output directory. `EVAL_LIMIT` limits the number
of questions when using the shell scripts. For single-GPU Transformers
inference, use `--retrieval-gpu-id 0 --generation-gpu-ids 0` and reduce the batch
size to fit the retriever and generator on the same device.

### Optional GPU FAISS

Install a CUDA-compatible FAISS build with GPU support to move index search
onto a GPU. For the evaluation scripts:

```bash
FAISS_USE_GPU=1 bash run_evaluate.sh
```

For a direct inference or synthesis command, pass `--faiss-use-gpu`.
`--faiss-gpu-id` selects the GPU; the default is the retrieval GPU.
`--faiss-gpu-use-float16` optionally reduces index memory use.

### Optional vLLM Inference

```bash
pip install -r requirements-vllm.txt
GENERATION_BACKEND=vllm bash run_evaluate.sh
```

For direct inference, use `--generation-backend vllm`.
`--tensor-parallel-size` and `--gpu-memory-utilization` configure vLLM.
Offline synthesis uses Transformers.

The vLLM backend currently forwards prompts without truncation. A prompt
longer than `--generation-max-total-tokens` (4096 by default) raises an error.
When using this optional backend, ensure the prompt fits the context window
and leaves room for generation; use a direct inference command to configure
a larger model-supported context when needed. The default Transformers
backend truncates its input to the configured input limit.

## Repository Structure

```text
preprocess/                     Question conversion, corpus preparation, and subsets
prompts/prompt_set__1/           Answer and thought/query prompt templates
script/download/                Raw-data download and index construction
source/run/                     Synthesis, retriever training, and inference entry points
source/pipeline/                Iterative retrieval and generation controller
source/module/                  Retriever, reranker, generator, and index implementations
source/evaluation/              Official benchmark evaluation adapters
run_synthesize.sh               Offline supervision generation
run_train.sh                    Retriever training and automatic evaluation
run_evaluate.sh                 Evaluation on all three benchmarks
```

## Citation

If you use PROBE in your research, please cite:

```bibtex
@inproceedings{lu2026probe,
  title = {{PROBE}: Prospective Retrieval with Oracle-Based End-Goal Distillation for Multi-Hop Question Answering},
  author = {Lu, Xu and Zang, Zhengqing and Yao, Binhui and Yang, Luwei and Chen, Ruimin and Wang, Zeng and Wang, Haobo},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026}
}
```

## Acknowledgements

We acknowledge [Contriever](https://github.com/facebookresearch/contriever),
[DPR](https://github.com/facebookresearch/DPR),
[IRCoT](https://github.com/StonyBrookNLP/ircot),
[BGE](https://github.com/FlagOpen/FlagEmbedding),
[Transformers](https://github.com/huggingface/transformers), and
[FAISS](https://github.com/facebookresearch/faiss). We also acknowledge the
benchmark datasets and official evaluation implementations from
[HotpotQA](https://github.com/hotpotqa/hotpot),
[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), and
[MuSiQue](https://github.com/StonyBrookNLP/musique).

## License

This code is released under the [Apache License 2.0](LICENSE). Refer to the
original providers for the licenses of datasets and pretrained model weights.
