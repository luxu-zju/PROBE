"""Build a FAISS index and SQLite document store from embedding shards."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from source.module.index.docstore import Document


def document_fields(document, fallback_id: str):
    if isinstance(document, Document):
        return document.id, document.content or "", document.metadata or {}
    if isinstance(document, dict):
        return (
            str(document.get("id") or document.get("doc_id") or fallback_id),
            str(document.get("content") or document.get("text") or ""),
            document,
        )
    return fallback_id, str(document), {}


def build_index(embedding_dir: Path) -> int:
    shard_paths = sorted(embedding_dir.glob("embeddings_[0-9]*"))
    if not shard_paths:
        raise FileNotFoundError(f"No embedding shards found in {embedding_dir}")

    index = None
    document_ids = []
    seen_ids = set()
    database_path = embedding_dir / "docstore.db"
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE documents (id TEXT PRIMARY KEY, content TEXT, metadata BLOB)"
    )

    for shard_path in tqdm(shard_paths, desc="Indexing shards"):
        with shard_path.open("rb") as handle:
            embeddings, documents = pickle.load(handle)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) != len(documents):
            raise ValueError(f"Embedding/document count mismatch in {shard_path}")
        if index is None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
        elif embeddings.shape[1] != index.d:
            raise ValueError(f"Embedding dimension mismatch in {shard_path}")

        valid_indices = []
        rows = []
        for position, document in enumerate(documents):
            fallback_id = f"{shard_path.stem}_{position}"
            document_id, content, metadata = document_fields(document, fallback_id)
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            document_ids.append(document_id)
            valid_indices.append(position)
            rows.append((document_id, content, json.dumps(metadata, ensure_ascii=False)))

        if valid_indices:
            index.add(np.ascontiguousarray(embeddings[valid_indices]))
            connection.executemany(
                "INSERT INTO documents (id, content, metadata) VALUES (?, ?, ?)", rows
            )
            connection.commit()

    connection.close()
    if index is None or index.ntotal == 0:
        raise ValueError(f"No embeddings were indexed from {embedding_dir}")
    faiss.write_index(index, str(embedding_dir / "index.faiss"))
    with (embedding_dir / "faiss_id_to_docstore_id.pkl").open("wb") as handle:
        pickle.dump(document_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Indexed {index.ntotal} unique documents in {embedding_dir}")
    return int(index.ntotal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    args = parser.parse_args()
    build_index(args.embedding_dir)


if __name__ == "__main__":
    main()
