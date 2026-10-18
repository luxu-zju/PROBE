"""Small helpers shared by retrieval and generation steps."""

from typing import List, Literal, Tuple

from source.module.index.docstore import Document
from source.pipeline.constants import QUESTION_THOUGHT_DELIM, THOUGHT_THOUGHT_DELIM
from source.pipeline.state import BaseState, DocumentState, QuestionState, ThoughtState


def parse_path(path: List[BaseState]) -> Tuple[str, str, List[str], List[List[Document]]]:
    question_id = ""
    question = ""
    thoughts: List[str] = []
    document_groups: List[List[Document]] = []
    for state in path:
        if isinstance(state, QuestionState):
            question_id = state.question_id or ""
            question = state.question or ""
        elif isinstance(state, ThoughtState) and state.thought:
            thoughts.append(state.thought)
        elif isinstance(state, DocumentState):
            document_groups.append(state.documents or [])
    return question_id, question, thoughts, document_groups


def select_prompt_documents(
    document_groups: List[List[Document]],
    source: Literal["last_only", "full"],
    retrieval_count: int,
    maximum_count: int,
) -> List[Document]:
    groups = document_groups[-1:] if source == "last_only" else document_groups
    selected: List[Document] = []
    seen_ids = set()
    for group in groups:
        for document in group[:retrieval_count]:
            if document.id in seen_ids:
                continue
            seen_ids.add(document.id)
            selected.append(document)
            if len(selected) >= maximum_count:
                return selected
    return selected


def filter_documents(
    documents: List[Document],
    seen_ids: set,
    remove_duplicates: bool,
) -> List[Document]:
    if not remove_duplicates:
        return documents
    return [document for document in documents if document.id not in seen_ids]


def prepare_retrieval_query(
    question: str,
    thoughts: List[str],
    query_type: Literal["last_only", "full"],
) -> str:
    if query_type == "last_only" and thoughts:
        return thoughts[-1]
    # The reference full-query formatter always appends the thought segment,
    # including the empty segment on the first retrieval hop.
    parts = [question, THOUGHT_THOUGHT_DELIM.join(thoughts)]
    return QUESTION_THOUGHT_DELIM.join(parts)


def retrieval_thoughts(path: List[BaseState]) -> List[str]:
    """Combine each reasoning summary with its explicit next-hop query."""
    return [
        f"{state.thought or ''} {state.query or ''}".strip()
        for state in path
        if isinstance(state, ThoughtState) and (state.thought or state.query)
    ]


def clean_json_value(text: str, key: str) -> str:
    return text.replace("{", "").replace("}", "").replace(key, "").replace(":", "").strip()
