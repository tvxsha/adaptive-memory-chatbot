"""SQLite-backed memory store. Embeddings are stored as JSON-encoded lists
for simplicity — fine at this project's scale. If your evaluation set grows
large enough that similarity search gets slow, swap this for a proper vector
index (e.g. chromadb or faiss) without changing the public methods below.
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

from memory_engine.config import MEMORY_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    text TEXT NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL,
    embedding TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation ON memory_items(conversation_id);
"""


class MemoryStore:
    def __init__(self, db_path: str = MEMORY_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add(self, conversation_id: str, text: str, category: str, score: float,
            embedding: np.ndarray) -> str:
        item_id = str(uuid.uuid4())
        now = time.time()
        self.conn.execute(
            "INSERT INTO memory_items (id, conversation_id, text, category, score, "
            "embedding, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, conversation_id, text, category, score,
             json.dumps(embedding.tolist()), now, now),
        )
        self.conn.commit()
        return item_id

    def replace(self, item_id: str, text: str, category: str, score: float,
                embedding: np.ndarray) -> None:
        """Used when contradiction detection says the new statement updates
        or contradicts an old one — overwrite instead of duplicating.
        """
        self.conn.execute(
            "UPDATE memory_items SET text=?, category=?, score=?, embedding=?, "
            "updated_at=? WHERE id=?",
            (text, category, score, json.dumps(embedding.tolist()), time.time(), item_id),
        )
        self.conn.commit()

    def get_all(self, conversation_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, text, category, score, embedding, created_at, updated_at "
            "FROM memory_items WHERE conversation_id=?",
            (conversation_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "category": r[2],
                "score": r[3],
                "embedding": np.array(json.loads(r[4])),
                "created_at": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    def delete_conversation(self, conversation_id: str) -> None:
        self.conn.execute(
            "DELETE FROM memory_items WHERE conversation_id=?", (conversation_id,)
        )
        self.conn.commit()

    def export_snapshot(self, conversation_id: str) -> dict:
        """Model-agnostic export format — embeddings excluded since they're
        only useful internally for retrieval, not for another model to read.
        """
        items = self.get_all(conversation_id)
        return {
            "conversation_id": conversation_id,
            "exported_at": time.time(),
            "memory": [
                {"text": i["text"], "category": i["category"], "score": i["score"]}
                for i in sorted(items, key=lambda x: x["score"], reverse=True)
            ],
        }
