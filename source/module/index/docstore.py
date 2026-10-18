"""SQLite-backed document storage used by the FAISS index."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    id: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float | None = None

    def __str__(self, max_num_words: int = 350) -> str:
        title = str(self.metadata.get("title", "")).strip()
        text = str(self.metadata.get("text") or self.content).strip()
        text = " ".join(text.split()[:max_num_words])
        return f"Wikipedia Title: {title}\n{text}" if title else text


class Docstore:
    def __init__(
        self,
        database_path: str | Path,
        sqlite_file_name: str = "docstore.db",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.sqlite_path = self.database_path / sqlite_file_name
        if connection is None:
            self.database_path.mkdir(parents=True, exist_ok=True)
            if self.sqlite_path.exists():
                raise FileExistsError(f"Database already exists: {self.sqlite_path}")
            connection = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            connection.execute(
                "CREATE TABLE documents (id TEXT PRIMARY KEY, content TEXT, metadata TEXT)"
            )
            connection.commit()
        self.connection = connection

    @classmethod
    def load(
        cls, database_path: str | Path, sqlite_file_name: str = "docstore.db"
    ) -> "Docstore":
        sqlite_path = Path(database_path) / sqlite_file_name
        if not sqlite_path.is_file():
            raise FileNotFoundError(sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread=False)
        return cls(database_path, sqlite_file_name, connection)

    def add(self, documents: Document | list[Document]) -> None:
        if isinstance(documents, Document):
            documents = [documents]
        rows = [
            (document.id, document.content, json.dumps(document.metadata, ensure_ascii=False))
            for document in documents
        ]
        self.connection.executemany(
            "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)", rows
        )
        self.connection.commit()

    def search(self, document_id: str) -> Document | None:
        row = self.connection.execute(
            "SELECT id, content, metadata FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(id=row[0], content=row[1] or "", metadata=json.loads(row[2] or "{}"))

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def close(self) -> None:
        self.connection.close()
