from pathlib import Path

DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "feedback.db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = "llama-3.3-70b-versatile"
REWRITE_MODEL = "llama-3.1-8b-instant"
SUGGESTION_MODEL = "llama-3.1-8b-instant"

CHUNK_WORDS = 600      # ~800 tokens
OVERLAP_WORDS = 80    # ~150 tokens
CANDIDATES_PER_INDEX = 20 # 20 p/bm25 & 20 p/vector search
RERANK_CANDIDATES = 30 # 40 -> 30
TOP_K = 5
RRF_K = 60
