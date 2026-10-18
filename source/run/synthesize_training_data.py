"""Generate per-hop BGE relevance supervision with iterative retrieval and an LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch

from source.module.generate.transformers import (
    ParallelTransformersGenerator,
    TransformersGeneratorConfig,
)
from source.module.index.index import Indexer, IndexerConfig
from source.module.retrieve.cross_encoder import CrossEncoder, CrossEncoderConfig
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import QuestionState
from source.pipeline.step.end import EndStep
from source.pipeline.step.generation import (
    AnswerGenerateOutputParser,
    AnswerGeneratePromptGenerator,
    GenerationStep,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.retrieval import RetrievalStep
from source.pipeline.step.synthesis import SaveSynthesisStep
from source.utility.data_utils import load_data_from_jsonl


def first_answer(answer) -> str:
    while isinstance(answer, (list, tuple)) and answer:
        answer = answer[0]
    return str(answer)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--running-name", default="offline_synthesis")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--index-root", type=Path, default=Path("data/database/contriever_msmarco"))
    parser.add_argument("--run-output-root", type=Path, default=Path("synthesis_runs"))
    parser.add_argument("--generation-model-name", required=True)
    parser.add_argument("--retrieval-model-name-or-path", default="facebook/contriever-msmarco")
    parser.add_argument("--reranker-model-name-or-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size-per-device", type=int, default=16)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--generation-max-total-tokens", type=int, default=4096)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--retrieval-count", type=int, default=8)
    parser.add_argument("--retrieval-buffer-size", type=int, default=32)
    parser.add_argument(
        "--max-num-thought",
        "--max-thought-rounds",
        dest="max_thought_rounds",
        type=int,
        default=6,
        help=(
            "T_max: thought/next-hop-query rounds after initial retrieval. "
            "T_max=6 permits at most 7 retrieval hops."
        ),
    )
    parser.add_argument("--retrieval-gpu-id", type=int, default=0)
    parser.add_argument(
        "--faiss-use-gpu",
        action="store_true",
        help="Move the FAISS index to GPU; CPU search is the default.",
    )
    parser.add_argument(
        "--faiss-gpu-id",
        type=int,
        default=None,
        help="FAISS GPU ID (default: the retrieval GPU ID).",
    )
    parser.add_argument(
        "--faiss-gpu-use-float16",
        action="store_true",
        help="Store/search the optional GPU FAISS index in float16.",
    )
    parser.add_argument("--generation-gpu-ids", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=15000,
        help="Number of sorted training examples to synthesize (default: 15000).",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed questions and regenerate incomplete ones without duplicate hops.",
    )
    return parser.parse_args()


def completed_question_ids(output_path: Path, max_thought_rounds: int) -> set[str]:
    """Return questions whose last saved answer ended synthesis successfully."""

    last_records: dict[str, dict] = {}
    with output_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question_id = str(record["question_id"])
                step_index = int(record.get("step_idx", -1))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Cannot resume: malformed record at {output_path}:{line_number}: {error}"
                ) from error
            previous = last_records.get(question_id)
            if previous is None or step_index >= int(previous.get("step_idx", -1)):
                last_records[question_id] = record

    completed = set()
    for question_id, record in last_records.items():
        answer = str(record.get("model_answer", "")).strip().lower().rstrip(".")
        step_index = int(record.get("step_idx", -1))
        if answer not in {"", "unknown", "unanswerable", "no answer", "not found", "i don't know", "none"}:
            completed.add(question_id)
        elif step_index >= max_thought_rounds:
            completed.add(question_id)
    return completed


def remove_incomplete_records(output_path: Path, incomplete_ids: set[str]) -> int:
    """Atomically drop records for questions that will be regenerated."""

    if not incomplete_ids:
        return 0
    temporary_path = None
    removed = 0
    try:
        with output_path.open(encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.resume-",
                suffix=".tmp",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                for line in source:
                    if line.strip() and str(json.loads(line)["question_id"]) in incomplete_ids:
                        removed += 1
                    else:
                        target.write(line)
                if removed:
                    target.flush()
                    os.fsync(target.fileno())
        if removed:
            shutil.copymode(output_path, temporary_path)
            os.replace(temporary_path, output_path)
        return removed
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {args.output_path}. Move it, choose a new path, "
            "or pass --resume."
        )
    cfg = PipelineConfig(
        running_name=args.running_name,
        dataset=args.dataset,
        dataset_split="train",
        data_root=args.data_root,
        index_root=args.index_root,
        output_root=args.run_output_root,
        batch_size=args.batch_size,
        seed=args.seed,
        retrieval_count=args.retrieval_count,
        retrieval_buffer_size=args.retrieval_buffer_size,
        max_thought_rounds=args.max_thought_rounds,
    )
    questions, ground_truths = load_data_from_jsonl(str(cfg.data_file_path))
    question_ids = sorted(questions)[args.start_index :]
    if args.limit is not None:
        question_ids = question_ids[: args.limit]
    if not question_ids:
        raise ValueError("No training questions were selected")
    # Match the shuffled training DataLoader used by reference synthesis.
    random.Random(args.seed).shuffle(question_ids)
    if args.resume and args.output_path.exists():
        selected_ids = set(question_ids)
        completed_ids = completed_question_ids(args.output_path, args.max_thought_rounds)
        unexpected_ids = completed_ids - selected_ids
        if unexpected_ids:
            raise ValueError(
                f"Cannot resume: output contains {len(unexpected_ids)} completed question "
                "IDs outside the currently selected --start-index/--limit range"
            )
        question_ids = [question_id for question_id in question_ids if question_id not in completed_ids]
        print(
            f"Resume: {len(completed_ids)} selected questions are complete; "
            f"{len(question_ids)} questions remain."
        )
        if not question_ids:
            print(f"Synthesis is already complete: {args.output_path}")
            return

    assigned_devices = (args.retrieval_gpu_id, *args.generation_gpu_ids)
    if len(set(assigned_devices)) != 3:
        raise ValueError("Retrieval and the two generation models must use three different GPUs")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(assigned_devices):
        raise RuntimeError(
            "Offline synthesis requires three visible GPUs: GPU 0 for retrieval/reranking "
            "and GPUs 1-2 for Transformers generation"
        )
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            query_model_name_or_path=args.retrieval_model_name_or_path,
            batch_size=args.retrieval_batch_size,
            training_strategy=None,
            use_fp16=False,
            device_id=args.retrieval_gpu_id,
        )
    )
    # Pseudo-label synthesis is inference-only. Disable dropout so retrieval
    # supervision is deterministic for a fixed model, input order, and seed.
    retriever.query_model.eval()
    reranker = CrossEncoder(
        CrossEncoderConfig(
            model_name_or_path=args.reranker_model_name_or_path,
            batch_size=args.reranker_batch_size,
            use_fp16=False,
            device_id=args.retrieval_gpu_id,
        )
    )
    generator = ParallelTransformersGenerator(
        TransformersGeneratorConfig(
            model_name=args.generation_model_name,
            device_ids=tuple(args.generation_gpu_ids),
            batch_size_per_device=args.generation_batch_size_per_device,
            max_total_tokens=args.generation_max_total_tokens,
            max_new_tokens=args.generation_max_new_tokens,
        )
    )
    indexer = Indexer.load_local(
        IndexerConfig(
            database_path=str(cfg.database_path),
            use_gpu=args.faiss_use_gpu,
            gpu_id=(
                args.retrieval_gpu_id
                if args.faiss_gpu_id is None
                else args.faiss_gpu_id
            ),
            gpu_use_float16=args.faiss_gpu_use_float16,
        )
    )
    controller = PipelineController(
        pipeline=[
            RetrievalStep(cfg, retriever, indexer, reranker=reranker),
            GenerationStep(
                cfg,
                generator,
                AnswerGeneratePromptGenerator(cfg),
                AnswerGenerateOutputParser(cfg),
            ),
            SaveSynthesisStep(args.output_path, cfg.max_thought_rounds),
            EndStep(cfg, limit_retrieval_rounds=True),
            GenerationStep(
                cfg,
                generator,
                ThoughtGeneratePromptGenerator(cfg),
                ThoughtGenerateOutputParser(cfg),
            ),
        ],
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
        retain_history=False,
    )
    if args.resume and args.output_path.exists():
        removed = remove_incomplete_records(args.output_path, set(question_ids))
        if removed:
            print(f"Resume: removed {removed} incomplete hop records before regeneration.")
    controller.run(
        [
            QuestionState(
                question_id=question_id,
                question=questions[question_id],
                ground_truth=first_answer(ground_truths[question_id]),
            )
            for question_id in question_ids
        ],
        batch_size=cfg.batch_size,
    )
    print(f"Saved offline supervision to {args.output_path}")


if __name__ == "__main__":
    main()
