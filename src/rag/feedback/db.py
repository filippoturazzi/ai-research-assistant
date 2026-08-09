import json
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  query TEXT NOT NULL,
  rewritten_query TEXT NOT NULL,
  answer TEXT NOT NULL,
  sources TEXT NOT NULL,
  model TEXT NOT NULL,
  latency_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  interaction_id INTEGER NOT NULL REFERENCES interactions(id),
  rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
  comment TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class FeedbackDB:
    def __init__(self, path: Path | str):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def log_interaction(self, query: str, rewritten_query: str, answer: str,
                        sources: list[dict], model: str, latency_ms: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO interactions (query, rewritten_query, answer, sources, model, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (query, rewritten_query, answer, json.dumps(sources, ensure_ascii=False),
                 model, latency_ms),
            )
            self._conn.commit()
            return cur.lastrowid

    def add_feedback(self, interaction_id: int, rating: int, comment: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback (interaction_id, rating, comment) VALUES (?, ?, ?)",
                (interaction_id, rating, comment),
            )
            self._conn.commit()

    def metrics(self) -> dict:
        with self._lock:
            one = lambda sql, *args: self._conn.execute(sql, args).fetchone()

            totals = one("SELECT COUNT(*) AS n, AVG(latency_ms) AS lat FROM interactions")
            fb = one("SELECT COUNT(*) AS n, AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END) AS rate"
                     " FROM feedback")
            fb7 = one("SELECT AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END) AS rate FROM feedback"
                      " WHERE created_at >= datetime('now', '-7 days')")

            negatives = [
                {"interaction_id": r["id"], "query": r["query"], "answer": r["answer"],
                 "sources": json.loads(r["sources"]), "created_at": r["created_at"]}
                for r in self._conn.execute(
                    "SELECT i.* FROM interactions i JOIN feedback f ON f.interaction_id = i.id"
                    " WHERE f.rating = -1 ORDER BY f.created_at DESC LIMIT 50")
            ]
            top_documents = [
                {"doc_title": r["doc_title"], "citations": r["c"]}
                for r in self._conn.execute(
                    "SELECT json_extract(je.value, '$.doc_title') AS doc_title, COUNT(*) AS c"
                    " FROM interactions i, json_each(i.sources) je"
                    " GROUP BY doc_title ORDER BY c DESC LIMIT 10")
            ]
            return {
                "total_questions": totals["n"],
                "feedback_count": fb["n"],
                "approval_rate": fb["rate"],
                "approval_rate_7d": fb7["rate"],
                "avg_latency_ms": totals["lat"],
                "negatives": negatives,
                "top_documents": top_documents,
            }
