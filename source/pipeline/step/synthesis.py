"""Persist per-hop BGE relevance supervision for retriever training."""

from __future__ import annotations

import json
from pathlib import Path

from source.pipeline.state import (
    AnswerState,
    BaseState,
    DocumentState,
    QuestionState,
    ThoughtState,
)
from source.pipeline.utils import parse_path, prepare_retrieval_query, retrieval_thoughts


UNKNOWN_ANSWERS = {
    "unknown",
    "unanswerable",
    "no answer",
    "not found",
    "i don't know",
    "none",
}


class SaveSynthesisStep:
    def __init__(self, output_path: str | Path, max_thought_rounds: int) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_thought_rounds = max_thought_rounds

    def __call__(self, paths: list[list[BaseState]]) -> list[BaseState]:
        records = []
        for path in paths:
            question_state = next(state for state in path if isinstance(state, QuestionState))
            answer_state = path[-1]
            document_state = next(
                state for state in reversed(path[:-1]) if isinstance(state, DocumentState)
            )
            _, question, _, _ = parse_path(path)
            reasoning_states = [state for state in path if isinstance(state, ThoughtState)]
            thoughts = [str(state.thought or "") for state in reasoning_states]
            next_hop_queries = [str(state.query or "") for state in reasoning_states]
            training_query = prepare_retrieval_query(
                question,
                retrieval_thoughts(path),
                "full",
            )
            answer = answer_state.answer if isinstance(answer_state, AnswerState) else "unknown"
            normalized_answer = str(answer or "unknown").strip().lower().rstrip(".")
            is_final = normalized_answer not in UNKNOWN_ANSWERS
            if len(thoughts) >= self.max_thought_rounds:
                is_final = True

            details = []
            for document in document_state.documents or []:
                if document.relevance_score is None:
                    raise ValueError("Offline synthesis requires BGE reranker scores")
                details.append(
                    {"doc_text": str(document), "bge_score": document.relevance_score}
                )
            records.append(
                {
                    "question_id": question_state.question_id,
                    "query": training_query,
                    "question": question,
                    "ground_truth": question_state.ground_truth,
                    "model_answer": answer,
                    "is_final_step": is_final,
                    "reasoning_stage": (
                        "final"
                        if is_final
                        else "initial"
                        if not thoughts
                        else "intermediate"
                    ),
                    "step_idx": len(thoughts),
                    "thoughts": thoughts,
                    "next_hop_queries": next_hop_queries,
                    "details": details,
                }
            )

        if records:
            with self.output_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return [path[-1] for path in paths]
