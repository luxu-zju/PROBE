"""Run iterative multi-hop QA inference and report answer EM/F1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from source.evaluation.evaluate import official_evaluate
from source.module.generate.transformers import (
    ParallelTransformersGenerator,
    TransformersGenerator,
    TransformersGeneratorConfig,
)
from source.module.index.index import Indexer, IndexerConfig
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig
from source.pipeline.config import PipelineConfig
from source.pipeline.controller import PipelineController
from source.pipeline.state import DocumentState, QuestionState
from source.pipeline.step.end import EndStep
from source.pipeline.step.generation import (
    AnswerGenerateOutputParser,
    AnswerGeneratePromptGenerator,
    GenerationStep,
    ThoughtGenerateOutputParser,
    ThoughtGeneratePromptGenerator,
)
from source.pipeline.step.retrieval import RetrievalStep
from source.utility.data_utils import load_data_from_jsonl


def calculate_mhr(controller, contexts: dict, k: int) -> dict[str, float]:
    """Compute cumulative multi-hop supporting-title recall after each hop."""

    supporting_titles = {
        str(question_id): {
            str(context.get("title", "")).strip().lower()
            for context in question_contexts
            if context.get("is_supporting") and context.get("title")
        }
        for question_id, question_contexts in contexts.items()
    }
    recalls_by_question: list[list[float]] = []
    maximum_hops = 0
    for end_state_id in controller.end_state_ids:
        path = controller.get_path(end_state_id)
        question_id = str(path[0].question_id)
        gold_titles = supporting_titles.get(question_id, set())
        if not gold_titles:
            continue
        retrieved_titles: set[str] = set()
        hop_recalls: list[float] = []
        for state in path:
            if not isinstance(state, DocumentState):
                continue
            for document in (state.documents or [])[:k]:
                title = str(document.metadata.get("title") or document.id).strip().lower()
                for gold_title in gold_titles:
                    if (
                        title == gold_title
                        or title.startswith(gold_title + "_")
                        or title.startswith(gold_title + "|||")
                    ):
                        retrieved_titles.add(gold_title)
            hop_recalls.append(len(retrieved_titles) / len(gold_titles))
        if hop_recalls:
            recalls_by_question.append(hop_recalls)
            maximum_hops = max(maximum_hops, len(hop_recalls))

    if not recalls_by_question:
        return {f"MHR_1@{k}": 0.0}
    return {
        f"MHR_{hop + 1}@{k}": sum(
            recalls[hop] if hop < len(recalls) else recalls[-1]
            for recalls in recalls_by_question
        )
        / len(recalls_by_question)
        for hop in range(maximum_hops)
    }


def run_pipeline(cfg, generator, retriever, indexer, limit: int | None = None):
    cfg.prediction_file_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()
    questions, ground_truths, contexts = load_data_from_jsonl(
        file_path=str(cfg.data_file_path), return_contexts=True
    )
    selected_ids = sorted(questions)
    if limit is not None:
        selected_ids = selected_ids[:limit]
    questions = {question_id: questions[question_id] for question_id in selected_ids}
    ground_truths = {question_id: ground_truths[question_id] for question_id in selected_ids}
    contexts = {question_id: contexts[question_id] for question_id in selected_ids}

    controller = PipelineController(
        pipeline=[
            RetrievalStep(cfg, retriever, indexer),
            GenerationStep(
                cfg,
                generator,
                AnswerGeneratePromptGenerator(cfg),
                AnswerGenerateOutputParser(cfg),
            ),
            EndStep(
                cfg,
                combine_next_hop_query=True,
                limit_retrieval_rounds=True,
            ),
            GenerationStep(
                cfg,
                generator,
                ThoughtGeneratePromptGenerator(cfg),
                ThoughtGenerateOutputParser(cfg),
            ),
        ],
        logging_file_path=cfg.logging_file_path,
        prediction_file_path=cfg.prediction_file_path,
    )
    predictions = controller.run(
        [
            QuestionState(question_id=question_id, question=question)
            for question_id, question in questions.items()
        ],
        batch_size=cfg.batch_size,
    )
    # The fixed paper test subset is sampled from benchmark development data.
    evaluation_split = "dev" if cfg.dataset_split == "test" else cfg.dataset_split
    metrics = official_evaluate(
        dataset=cfg.dataset,
        predictions=predictions,
        raw_root=cfg.data_root / "raw_data",
        split=evaluation_split,
    )
    cfg.ground_truth_file_path.write_text(
        json.dumps(ground_truths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cfg.official_evaluation_file_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    metrics_with_mhr = {**metrics, **calculate_mhr(controller, contexts, cfg.retrieval_count)}
    combined_metrics_path = cfg.official_evaluation_file_path.with_name(
        cfg.official_evaluation_file_path.stem + "_with_mhr.json"
    )
    combined_metrics_path.write_text(
        json.dumps(metrics_with_mhr, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics_with_mhr, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--running-name", required=True)
    parser.add_argument("--method", choices=["base", "probe"], default="probe")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--index-root", type=Path, default=Path("data/database/contriever_msmarco"))
    parser.add_argument("--output-root", type=Path, default=Path("predictions"))
    parser.add_argument("--dataset-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--generation-model-name", required=True)
    parser.add_argument(
        "--generation-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Generation backend. Transformers is the reference implementation.",
    )
    parser.add_argument("--retrieval-query-model-name-or-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
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
    parser.add_argument(
        "--generation-gpu-ids",
        type=int,
        nargs="+",
        default=[1, 2],
        help="One or two GPU IDs for Transformers generation (default: 1 2).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate all selected questions by default; set a positive value for a smoke test.",
    )
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit == 0:
        args.limit = None
    elif args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    cfg = PipelineConfig(
        running_name=args.running_name,
        dataset=args.dataset,
        data_root=args.data_root,
        index_root=args.index_root,
        output_root=args.output_root,
        dataset_split=args.dataset_split,
        method=args.method,
        batch_size=args.batch_size,
        seed=args.seed,
        retrieval_count=args.retrieval_count,
        retrieval_buffer_size=args.retrieval_buffer_size,
        max_thought_rounds=args.max_thought_rounds,
    )
    if args.generation_backend == "transformers":
        generation_config = TransformersGeneratorConfig(
            model_name=args.generation_model_name,
            device_ids=tuple(args.generation_gpu_ids),
            batch_size_per_device=args.generation_batch_size,
            max_total_tokens=args.generation_max_total_tokens,
            max_new_tokens=args.generation_max_new_tokens,
        )
        if len(args.generation_gpu_ids) == 1:
            generator = TransformersGenerator(
                generation_config, device_id=args.generation_gpu_ids[0]
            )
        elif len(args.generation_gpu_ids) == 2:
            generator = ParallelTransformersGenerator(generation_config)
        else:
            raise ValueError("Transformers inference supports one or two generation GPUs")
    else:
        # Keep vLLM optional: importing this module requires the optional package.
        from source.module.generate.llama import LlamaGenerator, LlamaGeneratorConfig

        generator = LlamaGenerator(
            LlamaGeneratorConfig(
                model_name=args.generation_model_name,
                batch_size=args.generation_batch_size,
                max_total_tokens=args.generation_max_total_tokens,
                max_new_tokens=args.generation_max_new_tokens,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )
    retriever = DenseRetriever(
        DenseRetrieverConfig(
            query_model_name_or_path=args.retrieval_query_model_name_or_path,
            batch_size=args.retrieval_batch_size,
            training_strategy=None,
            use_fp16=False,
            device_id=args.retrieval_gpu_id,
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
    run_pipeline(cfg, generator, retriever, indexer, args.limit)


if __name__ == "__main__":
    main()
