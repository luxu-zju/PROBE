"""FAISS inner-product index paired with a SQLite document store."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .docstore import Docstore, Document


@dataclass
class IndexerConfig:
    database_path: str
    embedding_size: int = 768
    index_file_name: str = "index.faiss"
    use_gpu: bool = False
    gpu_id: int = 0
    gpu_use_float16: bool = False


@dataclass
class IndexerOutput:
    documents: list[Document]
    scores: list[float]


class Indexer:
    def __init__(
        self,
        cfg: IndexerConfig,
        faiss_index=None,
        docstore: Docstore | None = None,
        document_ids: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.database_path = Path(cfg.database_path)
        self.faiss_index = faiss_index or faiss.IndexFlatIP(cfg.embedding_size)
        self.docstore = docstore or Docstore(self.database_path)
        self.document_ids = document_ids or []
        self._gpu_resources = None

    @classmethod
    def load_local(cls, cfg: IndexerConfig) -> "Indexer":
        database_path = Path(cfg.database_path)
        index = faiss.read_index(str(database_path / cfg.index_file_name))
        if cfg.use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("GPU FAISS was requested, but faiss-gpu is not installed")
            resources = faiss.StandardGpuResources()
            options = faiss.GpuClonerOptions()
            options.useFloat16 = cfg.gpu_use_float16
            index = faiss.index_cpu_to_gpu(resources, cfg.gpu_id, index, options)
        else:
            resources = None

        with (database_path / "faiss_id_to_docstore_id.pkl").open("rb") as handle:
            document_ids = pickle.load(handle)
        docstore = Docstore.load(database_path)
        if index.ntotal != len(document_ids) or index.ntotal != docstore.count():
            raise ValueError("FAISS index, ID mapping, and document store have different sizes")
        instance = cls(cfg, index, docstore, document_ids)
        instance._gpu_resources = resources
        return instance

    def search(self, query_embeddings: np.ndarray, k: int = 8) -> list[IndexerOutput]:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        scores, indices = self.faiss_index.search(queries, min(k, self.faiss_index.ntotal))
        outputs = []
        for row_scores, row_indices in zip(scores, indices):
            documents = [
                self.docstore.search(self.document_ids[int(index)]) for index in row_indices
            ]
            outputs.append(
                IndexerOutput(
                    documents=[document for document in documents if document is not None],
                    scores=[float(score) for score in row_scores],
                )
            )
        return outputs
