"""Dense retrieval step for iterative question answering."""

from typing import List

import torch

from source.pipeline.state import BaseState, DocumentState
from source.pipeline.utils import (
    filter_documents,
    parse_path,
    prepare_retrieval_query,
    retrieval_thoughts,
)


class RetrievalStep:
    def __init__(self, cfg, retriever, indexer, reranker=None) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.indexer = indexer
        self.reranker = reranker

    def __call__(self, paths: List[List[BaseState]]) -> List[BaseState]:
        queries = []
        previous_document_ids = []
        parent_ids = []

        for path in paths:
            _, question, _, document_groups = parse_path(path)
            thoughts = retrieval_thoughts(path)
            queries.append(
                prepare_retrieval_query(question, thoughts, self.cfg.retrieval_query_type)
            )
            previous_document_ids.append(
                {document.id for group in document_groups for document in group}
            )
            parent_ids.append(path[-1].state_id)

        embeddings = self.retriever.embed(queries, input_type="query")
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().float().cpu().numpy()

        results = self.indexer.search(
            query_embeddings=embeddings,
            k=self.cfg.retrieval_buffer_size * (4 if self.reranker else 1),
        )
        if self.reranker:
            reranked_documents = self.reranker.rerank(
                queries,
                [result.documents for result in results],
                top_k=self.cfg.retrieval_buffer_size,
            )
            for result, documents in zip(results, reranked_documents):
                result.documents = documents
        return [
            DocumentState(
                parent_state_id=parent_id,
                documents=filter_documents(
                    result.documents,
                    seen_ids,
                    remove_duplicates=self.cfg.retrieval_no_duplicates,
                ),
            )
            for result, seen_ids, parent_id in zip(results, previous_document_ids, parent_ids)
        ]
