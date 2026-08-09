import pytest
import threading

from rag.feedback.db import FeedbackDB


@pytest.fixture
def db():
    d = FeedbackDB(":memory:")
    yield d
    d.close()


def _log(db, query="q"):
    return db.log_interaction(
        query=query, rewritten_query=query + " rw", answer="a",
        sources=[{"doc_title": "Paper A", "page": 1, "text": "t", "score": 0.9}],
        model="m", latency_ms=120,
    )


def test_log_and_metrics_counts(db):
    _log(db)
    _log(db)
    m = db.metrics()
    assert m["total_questions"] == 2
    assert m["feedback_count"] == 0
    assert m["approval_rate"] is None
    assert m["avg_latency_ms"] == 120.0


def test_feedback_and_approval_rate(db):
    i1, i2 = _log(db), _log(db)
    db.add_feedback(i1, 1)
    db.add_feedback(i2, -1, comment="errou a fonte")
    m = db.metrics()
    assert m["feedback_count"] == 2
    assert m["approval_rate"] == 0.5
    assert len(m["negatives"]) == 1
    assert m["negatives"][0]["interaction_id"] == i2


def test_invalid_rating_rejected(db):
    i = _log(db)
    with pytest.raises(Exception):
        db.add_feedback(i, 0)


def test_top_documents(db):
    _log(db)
    _log(db)
    m = db.metrics()
    assert m["top_documents"][0] == {"doc_title": "Paper A", "citations": 2}


def test_concurrent_writes_are_safe(db):
    num_threads = 8
    calls_per_thread = 5
    interaction_ids = []
    lock = threading.Lock()

    def worker():
        for _ in range(calls_per_thread):
            interaction_id = _log(db)
            with lock:
                interaction_ids.append(interaction_id)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    m = db.metrics()
    assert m["total_questions"] == num_threads * calls_per_thread
    assert len(interaction_ids) == num_threads * calls_per_thread
    assert len(set(interaction_ids)) == num_threads * calls_per_thread
