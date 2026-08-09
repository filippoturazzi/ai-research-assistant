# AI Research Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sistema RAG avançado de perguntas e respostas sobre documentos (papers de IA), com busca híbrida FAISS+BM25, re-ranking, citações, feedback e dashboard — Streamlit + FastAPI + núcleo Python puro.

**Architecture:** Três camadas: núcleo RAG (`src/rag/`, biblioteca pura), API fina (`src/api/`, FastAPI) e UI (`src/app/`, Streamlit falando HTTP com a API). Índices FAISS + BM25 com fusão RRF e cross-encoder; geração via Groq; interações/feedback em SQLite.

**Tech Stack:** Python 3.11+, faiss-cpu, sentence-transformers, rank-bm25, pypdf, groq, FastAPI, Streamlit, SQLite (stdlib), pytest (+ httpx, fpdf2 em dev).

**Spec:** `docs/superpowers/specs/2026-08-09-ai-research-assistant-design.md`

## Global Constraints

- Python >= 3.11; instalação editável (`pip install -e ".[dev]"`); imports absolutos (`from rag...`), src-layout.
- Modelos fixos (constantes em `src/rag/config.py`): embeddings `sentence-transformers/all-MiniLM-L6-v2` (384 dim, normalizado), re-ranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, geração `llama-3.3-70b-versatile`, reescrita `llama-3.1-8b-instant`.
- Groq exclusivamente via `GROQ_API_KEY` (`.env`, nunca commitado). Nenhum teste chama Groq real — sempre fakes.
- Testes padrão rodam offline; testes que baixam modelos levam marker `integration` (`pytest -m "not integration"` roda sem rede).
- Chunking: ~600 palavras (~800 tokens) com overlap de ~110 palavras (~150 tokens).
- Busca: top-20 por índice, RRF com k=60, re-rank dos candidatos fundidos, top-5 final.
- Erros nunca vazam stacktrace para o usuário final (API retorna JSON de erro claro; UI mostra mensagem amigável).
- Artefatos gerados (`data/index/`, `data/documents/*.pdf`, `data/feedback.db`, `.env`) ficam fora do git.
- Mensagens de commit em inglês, convenção `feat:`/`test:`/`chore:`/`docs:`.

---

### Task 1: Scaffolding do projeto

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/rag/__init__.py`, `src/rag/config.py`, `src/rag/errors.py`
- Create: `src/api/__init__.py`, `src/app/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Produces: pacotes `rag`, `api`, `app` importáveis; constantes de `rag.config` (nomes abaixo — todas as tasks usam); exceções `ExtractionError`, `GenerationError`, `IndexNotFoundError`, `DuplicateDocumentError` em `rag.errors`.

- [ ] **Step 1: Criar `pyproject.toml`**

```toml
[project]
name = "ai-research-assistant"
version = "0.1.0"
description = "Advanced RAG research assistant (FAISS + BM25 + RRF + cross-encoder re-ranking + Groq)"
requires-python = ">=3.11"
dependencies = [
  "faiss-cpu>=1.8",
  "sentence-transformers>=3.0",
  "rank-bm25>=0.2.2",
  "pypdf>=4.0",
  "groq>=0.11",
  "fastapi>=0.110",
  "uvicorn>=0.29",
  "python-multipart>=0.0.9",
  "streamlit>=1.35",
  "requests>=2.31",
  "python-dotenv>=1.0",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27", "fpdf2>=2.7"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "integration: baixa modelos reais (precisa de rede na primeira execucao)",
]
```

- [ ] **Step 2: Criar `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.env
data/index/
data/documents/*.pdf
data/feedback.db
*.egg-info/
.pytest_cache/
```

- [ ] **Step 3: Criar `src/rag/config.py`**

```python
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

CHUNK_WORDS = 600      # ~800 tokens
OVERLAP_WORDS = 110    # ~150 tokens
CANDIDATES_PER_INDEX = 20
RERANK_CANDIDATES = 30
TOP_K = 5
RRF_K = 60
```

- [ ] **Step 4: Criar `src/rag/errors.py`**

```python
class ExtractionError(Exception):
    """PDF sem texto extraível ou corrompido."""


class GenerationError(Exception):
    """Falha ao chamar o LLM após esgotar as tentativas."""


class IndexNotFoundError(Exception):
    """Índice ausente em disco — rodar scripts/build_index.py."""


class DuplicateDocumentError(Exception):
    """Documento com mesmo doc_id já indexado."""
```

- [ ] **Step 5: Criar `__init__.py` vazios** — `src/rag/__init__.py`, `src/api/__init__.py`, `src/app/__init__.py`, `tests/__init__.py`.

- [ ] **Step 6: Escrever teste de fumaça `tests/test_scaffolding.py`**

```python
from rag import config, errors


def test_config_constants():
    assert config.EMBEDDING_DIM == 384
    assert config.TOP_K == 5


def test_errors_are_exceptions():
    assert issubclass(errors.IndexNotFoundError, Exception)
    assert issubclass(errors.DuplicateDocumentError, Exception)
```

- [ ] **Step 7: Instalar e rodar**

Run: `pip install -e ".[dev]"` e depois `pytest tests/test_scaffolding.py -v`
Expected: 2 PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: project scaffolding with config and error types"
```

---

### Task 2: Modelo Chunk + chunker

**Files:**
- Create: `src/rag/models.py`
- Create: `src/rag/ingestion/__init__.py`, `src/rag/ingestion/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `rag.config.CHUNK_WORDS`, `rag.config.OVERLAP_WORDS`
- Produces: `@dataclass Chunk(chunk_id: str, doc_id: str, doc_title: str, page: int, position: int, text: str)` em `rag.models`; `chunk_pages(pages: list[tuple[int, str]], doc_id: str, doc_title: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[Chunk]` em `rag.ingestion.chunker`.

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_chunker.py
from rag.ingestion.chunker import chunk_pages


def test_short_doc_single_chunk():
    chunks = chunk_pages([(1, "Um parágrafo curto.")], doc_id="doc1", doc_title="Doc 1")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "doc1:0"
    assert (c.doc_id, c.doc_title, c.page, c.position) == ("doc1", "Doc 1", 1, 0)
    assert "parágrafo curto" in c.text


def test_long_text_splits_with_overlap():
    words = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_pages([(1, words)], doc_id="d", doc_title="D",
                         chunk_words=10, overlap_words=3)
    assert len(chunks) == 3
    # overlap: últimas palavras do chunk 0 reaparecem no início do chunk 1
    tail = chunks[0].text.split()[-3:]
    assert chunks[1].text.split()[:3] == tail


def test_page_attribution():
    chunks = chunk_pages(
        [(1, " ".join(f"a{i}" for i in range(10))), (2, " ".join(f"b{i}" for i in range(10)))],
        doc_id="d", doc_title="D", chunk_words=10, overlap_words=2,
    )
    assert chunks[0].page == 1
    assert chunks[1].page == 2


def test_empty_pages_no_chunks():
    assert chunk_pages([(1, "   ")], doc_id="d", doc_title="D") == []
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/models.py`**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    position: int
    text: str
```

- [ ] **Step 4: Implementar `src/rag/ingestion/chunker.py`**

```python
from rag.config import CHUNK_WORDS, OVERLAP_WORDS
from rag.models import Chunk


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    doc_title: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    # unidades (página, parágrafo); parágrafos maiores que o limite são fatiados
    units: list[tuple[int, str]] = []
    for page, text in pages:
        for para in _split_paragraphs(text):
            words = para.split()
            if len(words) <= chunk_words:
                units.append((page, para))
            else:
                for i in range(0, len(words), chunk_words):
                    units.append((page, " ".join(words[i:i + chunk_words])))

    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    current_words = 0
    overlap_text = ""

    def emit() -> None:
        nonlocal current, current_words, overlap_text
        if not current:
            return
        page = current[0][0]
        body = "\n\n".join(p for _, p in current)
        text = f"{overlap_text}\n\n{body}" if overlap_text else body
        position = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}:{position}", doc_id=doc_id, doc_title=doc_title,
            page=page, position=position, text=text,
        ))
        overlap_text = " ".join(body.split()[-overlap_words:])
        current, current_words = [], 0

    for page, para in units:
        n = len(para.split())
        if current and current_words + n > chunk_words:
            emit()
        current.append((page, para))
        current_words += n
    emit()
    return chunks
```

- [ ] **Step 5: Rodar e ver passar**

Run: `pytest tests/test_chunker.py -v`
Expected: 4 PASS

Nota: no `test_long_text_splits_with_overlap`, o overlap entra como prefixo mas as *unidades* seguem consumindo 10 palavras por chunk (30 palavras → 3 chunks). O início do texto do chunk 1 são as 3 palavras de overlap.

- [ ] **Step 6: Commit**

```bash
git add src/rag/models.py src/rag/ingestion tests/test_chunker.py
git commit -m "feat: chunk model and paragraph-aware chunker with overlap"
```

---

### Task 3: Extração de PDF

**Files:**
- Create: `src/rag/ingestion/pdf_extractor.py`
- Create: `tests/conftest.py`
- Test: `tests/test_pdf_extractor.py`

**Interfaces:**
- Consumes: `rag.errors.ExtractionError`
- Produces: `extract_pages(path: Path) -> list[tuple[int, str]]` (página 1-based, texto; páginas vazias omitidas; levanta `ExtractionError` se nada extraível) em `rag.ingestion.pdf_extractor`.

- [ ] **Step 1: Fixture de PDF em `tests/conftest.py`** (fpdf2, dev-only)

```python
import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 10, "Transformers use attention mechanisms.")
    pdf.add_page()
    pdf.multi_cell(0, 10, "BERT is a bidirectional encoder.")
    path = tmp_path / "sample.pdf"
    pdf.output(str(path))
    return path
```

- [ ] **Step 2: Escrever testes que falham**

```python
# tests/test_pdf_extractor.py
import pytest

from rag.errors import ExtractionError
from rag.ingestion.pdf_extractor import extract_pages


def test_extracts_pages_with_numbers(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "attention" in pages[0][1].lower()
    assert "bidirectional" in pages[1][1].lower()


def test_invalid_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(ExtractionError):
        extract_pages(bad)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `pytest tests/test_pdf_extractor.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 4: Implementar `src/rag/ingestion/pdf_extractor.py`**

```python
from pathlib import Path

from pypdf import PdfReader

from rag.errors import ExtractionError


def extract_pages(path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((number, text))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Não consegui ler o PDF '{path.name}': {exc}") from exc
    if not pages:
        raise ExtractionError(f"Nenhum texto extraível em '{path.name}'.")
    return pages
```

- [ ] **Step 5: Rodar e ver passar**

Run: `pytest tests/test_pdf_extractor.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add src/rag/ingestion/pdf_extractor.py tests/conftest.py tests/test_pdf_extractor.py
git commit -m "feat: page-aware PDF text extraction with clear errors"
```

---

### Task 4: Índice vetorial FAISS

**Files:**
- Create: `src/rag/retrieval/__init__.py`, `src/rag/retrieval/vector_index.py`
- Test: `tests/test_vector_index.py`

**Interfaces:**
- Produces: classe `VectorIndex` em `rag.retrieval.vector_index`: `__init__(dim: int)`, `add(vectors: np.ndarray) -> None` (float32, já normalizados; ids posicionais na ordem de inserção), `search(query: np.ndarray, k: int) -> list[tuple[int, float]]` (posição, score cosseno, decrescente), `save(path: Path)`, `classmethod load(path: Path) -> VectorIndex`, propriedade `size: int`.

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_vector_index.py
import numpy as np

from rag.retrieval.vector_index import VectorIndex


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_search_returns_nearest_first():
    idx = VectorIndex(dim=4)
    idx.add(np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([1, 1, 0, 0])]))
    hits = idx.search(_unit([1, 0.1, 0, 0]), k=2)
    assert hits[0][0] == 0
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]


def test_search_empty_index():
    assert VectorIndex(dim=4).search(_unit([1, 0, 0, 0]), k=5) == []


def test_save_and_load(tmp_path):
    idx = VectorIndex(dim=4)
    idx.add(np.stack([_unit([0, 0, 1, 0])]))
    idx.save(tmp_path / "v.faiss")
    loaded = VectorIndex.load(tmp_path / "v.faiss")
    assert loaded.size == 1
    assert loaded.search(_unit([0, 0, 1, 0]), k=1)[0][0] == 0
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_vector_index.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/vector_index.py`** (criar também `src/rag/retrieval/__init__.py` vazio)

```python
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.asarray(vectors, dtype="float32"))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.size)
        if k == 0:
            return []
        scores, ids = self.index.search(np.asarray([query], dtype="float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        index = faiss.read_index(str(path))
        obj = cls(index.d)
        obj.index = index
        return obj
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_vector_index.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval tests/test_vector_index.py
git commit -m "feat: FAISS flat inner-product vector index with persistence"
```

---

### Task 5: Índice BM25

**Files:**
- Create: `src/rag/retrieval/bm25_index.py`
- Test: `tests/test_bm25_index.py`

**Interfaces:**
- Produces: classe `BM25Index` em `rag.retrieval.bm25_index`: `__init__(texts: list[str])`, `add(texts: list[str]) -> None` (reconstrói o índice — rank-bm25 não é incremental), `search(query: str, k: int) -> list[tuple[int, float]]` (posição, score > 0, decrescente).

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_bm25_index.py
from rag.retrieval.bm25_index import BM25Index


def test_lexical_match_ranks_first():
    idx = BM25Index([
        "the transformer architecture uses attention",
        "convolutional networks process images",
        "reinforcement learning maximizes reward",
    ])
    hits = idx.search("transformer attention", k=2)
    assert hits[0][0] == 0


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything", k=5) == []


def test_add_rebuilds():
    idx = BM25Index(["first document about cats"])
    idx.add(["second document about faiss indexes"])
    hits = idx.search("faiss", k=1)
    assert hits[0][0] == 1
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_bm25_index.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/bm25_index.py`**

```python
import re

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self._rebuild()

    def _rebuild(self) -> None:
        corpus = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, texts: list[str]) -> None:
        self._texts.extend(texts)
        self._rebuild()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_bm25_index.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval/bm25_index.py tests/test_bm25_index.py
git commit -m "feat: BM25 lexical index with rebuild-on-add"
```

---

### Task 6: Fusão RRF

**Files:**
- Create: `src/rag/retrieval/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `rag.config.RRF_K`
- Produces: `reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]` em `rag.retrieval.fusion` — recebe listas de ids ordenadas por relevância, retorna (id, score RRF) decrescente; empates desempatados por id crescente.

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_fusion.py
from rag.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_wins():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]], k=60)
    assert fused[0][0] in (1, 2)
    # 2 aparece em pos 1 e 0; 1 em pos 0 e 2 → score(2) = 1/61+1/62 > score(1) = 1/61+1/63
    assert fused[0][0] == 2


def test_scores_formula():
    fused = reciprocal_rank_fusion([[7]], k=60)
    assert fused == [(7, 1 / 61)]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/fusion.py`**

```python
from rag.config import RRF_K


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_fusion.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval/fusion.py tests/test_fusion.py
git commit -m "feat: reciprocal rank fusion"
```

---

### Task 7: Embedder

**Files:**
- Create: `src/rag/retrieval/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Consumes: `rag.config.EMBEDDING_MODEL`
- Produces: classe `Embedder` em `rag.retrieval.embedder`: `__init__(model_name: str = EMBEDDING_MODEL, model=None)` (injeção para testes; sem `model`, carrega `SentenceTransformer(model_name)`); `embed_texts(texts: list[str]) -> np.ndarray` shape (n, dim) float32 L2-normalizado; `embed_query(text: str) -> np.ndarray` shape (dim,).

- [ ] **Step 1: Escrever testes que falham** (com modelo fake — sem download)

```python
# tests/test_embedder.py
import numpy as np

from rag.retrieval.embedder import Embedder


class FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        # vetores determinísticos pelo tamanho do texto
        out = np.stack([[len(t), 1.0, 0.0] for t in texts]).astype("float32")
        if normalize_embeddings:
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


def test_embed_texts_shape_and_norm():
    emb = Embedder(model=FakeModel())
    vecs = emb.embed_texts(["abc", "de"])
    assert vecs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embed_query_is_1d():
    emb = Embedder(model=FakeModel())
    assert emb.embed_query("hello").shape == (3,)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/embedder.py`**

```python
import numpy as np

from rag.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer  # import tardio: pesado
            model = SentenceTransformer(model_name)
        self._model = model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_embedder.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval/embedder.py tests/test_embedder.py
git commit -m "feat: sentence-transformers embedder with injectable model"
```

---

### Task 8: Re-ranker (cross-encoder)

**Files:**
- Create: `src/rag/retrieval/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `rag.models.Chunk`, `rag.config.RERANKER_MODEL`, `rag.config.TOP_K`
- Produces: classe `Reranker` em `rag.retrieval.reranker`: `__init__(model_name: str = RERANKER_MODEL, model=None)`; `rerank(query: str, chunks: list[Chunk], top_k: int = TOP_K) -> list[tuple[Chunk, float]]` (decrescente por score do cross-encoder).

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_reranker.py
from rag.models import Chunk
from rag.retrieval.reranker import Reranker


def _chunk(i, text):
    return Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title="D", page=1, position=i, text=text)


class FakeCrossEncoder:
    def predict(self, pairs):
        # pontua mais alto quando o chunk contém a query
        return [1.0 if p[0] in p[1] else 0.0 for p in pairs]


def test_rerank_orders_by_score():
    chunks = [_chunk(0, "nothing here"), _chunk(1, "the query appears: attention")]
    out = Reranker(model=FakeCrossEncoder()).rerank("attention", chunks, top_k=2)
    assert out[0][0].position == 1
    assert out[0][1] > out[1][1]


def test_rerank_truncates_to_top_k():
    chunks = [_chunk(i, "attention text") for i in range(10)]
    out = Reranker(model=FakeCrossEncoder()).rerank("attention", chunks, top_k=5)
    assert len(out) == 5


def test_rerank_empty():
    assert Reranker(model=FakeCrossEncoder()).rerank("q", [], top_k=5) == []
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_reranker.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/reranker.py`**

```python
import numpy as np

from rag.config import RERANKER_MODEL, TOP_K
from rag.models import Chunk


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder  # import tardio: pesado
            model = CrossEncoder(model_name)
        self._model = model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = np.asarray(self._model.predict([(query, c.text) for c in chunks]), dtype="float32")
        order = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in order]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_reranker.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: cross-encoder reranker with injectable model"
```

---

### Task 9: IndexStore (chunks + índices + persistência)

**Files:**
- Create: `src/rag/retrieval/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Chunk`, `VectorIndex`, `BM25Index`, `rag.errors.IndexNotFoundError`, `rag.config.EMBEDDING_DIM`
- Produces: classe `IndexStore` em `rag.retrieval.store`: atributos `chunks: list[Chunk]`, `vectors: VectorIndex`, `bm25: BM25Index`; `__init__(dim: int = EMBEDDING_DIM)`; `add(chunks: list[Chunk], vectors: np.ndarray) -> None`; `doc_ids() -> set[str]`; `save(dir: Path) -> None` (grava `index.faiss` + `chunks.json`); `classmethod load(dir: Path) -> IndexStore` (levanta `IndexNotFoundError` se ausente; BM25 é reconstruído dos textos).

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_store.py
import numpy as np
import pytest

from rag.errors import IndexNotFoundError
from rag.models import Chunk
from rag.retrieval.store import IndexStore


def _chunk(doc, i, text):
    return Chunk(chunk_id=f"{doc}:{i}", doc_id=doc, doc_title=doc.title(), page=1, position=i, text=text)


def _vecs(n):
    out = np.eye(4, dtype="float32")[:n]
    return out


def test_add_keeps_positions_aligned():
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "cats"), _chunk("b", 0, "faiss index")], _vecs(2))
    assert store.vectors.size == 2
    assert store.bm25.search("faiss", k=1)[0][0] == 1
    assert store.doc_ids() == {"a", "b"}


def test_save_load_roundtrip(tmp_path):
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "hello world")], _vecs(1))
    store.save(tmp_path)
    loaded = IndexStore.load(tmp_path)
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0].text == "hello world"
    assert loaded.vectors.size == 1
    assert loaded.bm25.search("hello", k=1)[0][0] == 0


def test_load_missing_raises(tmp_path):
    with pytest.raises(IndexNotFoundError):
        IndexStore.load(tmp_path / "nope")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_store.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/store.py`**

```python
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag.config import EMBEDDING_DIM
from rag.errors import IndexNotFoundError
from rag.models import Chunk
from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.vector_index import VectorIndex


class IndexStore:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.chunks: list[Chunk] = []
        self.vectors = VectorIndex(dim)
        self.bm25 = BM25Index([])

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors.add(vectors)
        self.bm25.add([c.text for c in chunks])

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.chunks}

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        self.vectors.save(dir / "index.faiss")
        (dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir: Path) -> "IndexStore":
        chunks_path = dir / "chunks.json"
        faiss_path = dir / "index.faiss"
        if not chunks_path.exists() or not faiss_path.exists():
            raise IndexNotFoundError(
                f"Índice não encontrado em '{dir}'. Rode: python scripts/build_index.py"
            )
        store = cls()
        store.chunks = [Chunk(**d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
        store.vectors = VectorIndex.load(faiss_path)
        store.bm25 = BM25Index([c.text for c in store.chunks])
        return store
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_store.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval/store.py tests/test_store.py
git commit -m "feat: index store with faiss+chunks persistence and bm25 rebuild"
```

---

### Task 10: HybridRetriever

**Files:**
- Create: `src/rag/retrieval/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `IndexStore`, `Embedder`, `Reranker`, `reciprocal_rank_fusion`, `rag.config` (`CANDIDATES_PER_INDEX`, `RERANK_CANDIDATES`, `TOP_K`)
- Produces: `@dataclass RetrievedChunk(chunk: Chunk, score: float)` e classe `HybridRetriever` em `rag.retrieval.retriever`: `__init__(store, embedder, reranker, candidates_per_index: int = CANDIDATES_PER_INDEX, rerank_candidates: int = RERANK_CANDIDATES, top_k: int = TOP_K)`; `retrieve(query: str) -> list[RetrievedChunk]`.

- [ ] **Step 1: Escrever testes que falham** (fakes para embedder/reranker; store real pequeno)

```python
# tests/test_retriever.py
import numpy as np

from rag.models import Chunk
from rag.retrieval.retriever import HybridRetriever, RetrievedChunk
from rag.retrieval.store import IndexStore


class PassthroughReranker:
    def rerank(self, query, chunks, top_k):
        return [(c, float(len(chunks) - i)) for i, c in enumerate(chunks[:top_k])]


class OneHotEmbedder:
    """Query 'dim0'..'dim3' vira o eixo correspondente."""

    def embed_query(self, text):
        v = np.zeros(4, dtype="float32")
        v[int(text[-1])] = 1.0
        return v


def _store():
    texts = ["faiss vector search", "bm25 lexical search", "transformers attention", "cats and dogs"]
    chunks = [Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title="D", page=1, position=i, text=t)
              for i, t in enumerate(texts)]
    store = IndexStore(dim=4)
    store.add(chunks, np.eye(4, dtype="float32"))
    return store


def test_hybrid_combines_vector_and_lexical():
    retriever = HybridRetriever(_store(), OneHotEmbedder(), PassthroughReranker(), top_k=3)
    # vetorial aponta para posição 0 ('dim0'); lexical acha 'bm25' na posição 1
    results = retriever.retrieve("bm25 dim0")
    positions = [r.chunk.position for r in results]
    assert 0 in positions and 1 in positions
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_respects_top_k():
    retriever = HybridRetriever(_store(), OneHotEmbedder(), PassthroughReranker(), top_k=2)
    assert len(retriever.retrieve("search dim0")) == 2
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/retrieval/retriever.py`**

```python
from dataclasses import dataclass

from rag.config import CANDIDATES_PER_INDEX, RERANK_CANDIDATES, TOP_K
from rag.models import Chunk
from rag.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, store, embedder, reranker,
                 candidates_per_index: int = CANDIDATES_PER_INDEX,
                 rerank_candidates: int = RERANK_CANDIDATES,
                 top_k: int = TOP_K):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates_per_index = candidates_per_index
        self.rerank_candidates = rerank_candidates
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vector_hits = self.store.vectors.search(
            self.embedder.embed_query(query), self.candidates_per_index)
        lexical_hits = self.store.bm25.search(query, self.candidates_per_index)
        fused = reciprocal_rank_fusion(
            [[i for i, _ in vector_hits], [i for i, _ in lexical_hits]])
        candidates = [self.store.chunks[i] for i, _ in fused[:self.rerank_candidates]]
        ranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
        return [RetrievedChunk(chunk=c, score=s) for c, s in ranked]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_retriever.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: hybrid retriever with RRF fusion and reranking"
```

---

### Task 11: Pipeline de ingestão

**Files:**
- Create: `src/rag/ingestion/pipeline.py`
- Test: `tests/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `extract_pages`, `chunk_pages`, `IndexStore`, `Embedder`, `rag.errors.DuplicateDocumentError`
- Produces: `ingest_pdf(path: Path, store: IndexStore, embedder: Embedder) -> int` em `rag.ingestion.pipeline` — `doc_id` = stem do arquivo; `doc_title` = stem com `_`/`-` → espaço, title-case; levanta `DuplicateDocumentError` se `doc_id` já indexado; retorna nº de chunks adicionados.

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_ingestion_pipeline.py
import numpy as np
import pytest

from rag.errors import DuplicateDocumentError
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.store import IndexStore


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


def test_ingest_adds_chunks(sample_pdf):
    store = IndexStore(dim=4)
    added = ingest_pdf(sample_pdf, store, FakeEmbedder())
    assert added == len(store.chunks) > 0
    assert store.chunks[0].doc_id == "sample"
    assert store.chunks[0].doc_title == "Sample"
    assert store.vectors.size == added


def test_duplicate_raises(sample_pdf):
    store = IndexStore(dim=4)
    ingest_pdf(sample_pdf, store, FakeEmbedder())
    with pytest.raises(DuplicateDocumentError):
        ingest_pdf(sample_pdf, store, FakeEmbedder())
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_ingestion_pipeline.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/ingestion/pipeline.py`**

```python
import re
from pathlib import Path

from rag.errors import DuplicateDocumentError
from rag.ingestion.chunker import chunk_pages
from rag.ingestion.pdf_extractor import extract_pages


def _title_from_stem(stem: str) -> str:
    return re.sub(r"[_-]+", " ", stem).strip().title()


def ingest_pdf(path: Path, store, embedder) -> int:
    doc_id = path.stem
    if doc_id in store.doc_ids():
        raise DuplicateDocumentError(f"Documento '{doc_id}' já está indexado.")
    pages = extract_pages(path)
    chunks = chunk_pages(pages, doc_id=doc_id, doc_title=_title_from_stem(doc_id))
    if not chunks:
        return 0
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.add(chunks, vectors)
    return len(chunks)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_ingestion_pipeline.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingestion/pipeline.py tests/test_ingestion_pipeline.py
git commit -m "feat: pdf ingestion pipeline with duplicate detection"
```

---

### Task 12: Cliente Groq com retry

**Files:**
- Create: `src/rag/generation/__init__.py`, `src/rag/generation/groq_chat.py`
- Create: `tests/fakes.py`
- Create: `.env.example`
- Test: `tests/test_groq_chat.py`

**Interfaces:**
- Consumes: `rag.errors.GenerationError`
- Produces: classe `GroqChat` em `rag.generation.groq_chat`: `__init__(api_key: str | None = None, client=None)` (sem `client`, cria `groq.Groq(api_key=api_key or env GROQ_API_KEY)`); `complete(model: str, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.2) -> str` — 3 tentativas com backoff exponencial (1s, 2s), depois `GenerationError`. Fakes reutilizáveis em `tests.fakes`: `FakeGroq(replies)` onde cada item é `str` (resposta) ou `Exception` (erro a levantar).

- [ ] **Step 1: Criar `tests/fakes.py`**

```python
from types import SimpleNamespace


class _FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


class FakeGroq:
    def __init__(self, replies):
        self.chat = SimpleNamespace(completions=_FakeCompletions(replies))

    @property
    def calls(self):
        return self.chat.completions.calls
```

- [ ] **Step 2: Escrever testes que falham**

```python
# tests/test_groq_chat.py
import pytest

from rag.errors import GenerationError
from rag.generation import groq_chat
from rag.generation.groq_chat import GroqChat
from tests.fakes import FakeGroq


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(groq_chat.time, "sleep", lambda s: None)


def test_returns_content():
    chat = GroqChat(client=FakeGroq(["hello"]))
    assert chat.complete("model-x", [{"role": "user", "content": "hi"}]) == "hello"


def test_retries_then_succeeds():
    fake = FakeGroq([RuntimeError("boom"), RuntimeError("boom"), "ok"])
    chat = GroqChat(client=fake)
    assert chat.complete("m", []) == "ok"
    assert len(fake.calls) == 3


def test_exhausted_raises_generation_error():
    fake = FakeGroq([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    with pytest.raises(GenerationError):
        GroqChat(client=fake).complete("m", [])
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `pytest tests/test_groq_chat.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 4: Implementar `src/rag/generation/groq_chat.py`** (criar `src/rag/generation/__init__.py` vazio)

```python
import os
import time

from rag.errors import GenerationError

_MAX_ATTEMPTS = 3


class GroqChat:
    def __init__(self, api_key: str | None = None, client=None):
        if client is None:
            from groq import Groq  # import tardio
            client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self._client = client

    def complete(self, model: str, messages: list[dict],
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        delay = 1.0
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise GenerationError(f"LLM indisponível: {exc}") from exc
                time.sleep(delay)
                delay *= 2
        raise GenerationError("unreachable")
```

- [ ] **Step 5: Criar `.env.example`**

```
GROQ_API_KEY=coloque_sua_chave_aqui
```

- [ ] **Step 6: Rodar e ver passar**

Run: `pytest tests/test_groq_chat.py -v`
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add src/rag/generation tests/fakes.py tests/test_groq_chat.py .env.example
git commit -m "feat: groq chat client with exponential backoff retry"
```

---

### Task 13: Reescrita de query

**Files:**
- Create: `src/rag/generation/rewriter.py`
- Test: `tests/test_rewriter.py`

**Interfaces:**
- Consumes: `GroqChat`, `rag.config.REWRITE_MODEL`, `rag.errors.GenerationError`, `tests.fakes.FakeGroq`
- Produces: `rewrite_query(chat: GroqChat, question: str, history: list[dict]) -> str` em `rag.generation.rewriter` — `history` no formato `[{"role": "user"|"assistant", "content": str}]` (últimos 6 turnos usados); em `GenerationError` ou saída vazia, retorna `question` (degradação graciosa).

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_rewriter.py
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat
from rag.generation.rewriter import rewrite_query
from tests.fakes import FakeGroq


def test_returns_rewritten_query():
    chat = GroqChat(client=FakeGroq(['"limitações da arquitetura Transformer"']))
    out = rewrite_query(chat, "e as limitações disso?",
                        [{"role": "user", "content": "o que é o Transformer?"}])
    assert out == "limitações da arquitetura Transformer"


def test_falls_back_to_original_on_error(monkeypatch):
    chat = GroqChat(client=FakeGroq([]))
    monkeypatch.setattr(chat, "complete",
                        lambda *a, **k: (_ for _ in ()).throw(GenerationError("down")))
    assert rewrite_query(chat, "pergunta original", []) == "pergunta original"


def test_falls_back_on_empty_output():
    chat = GroqChat(client=FakeGroq(["   "]))
    assert rewrite_query(chat, "pergunta original", []) == "pergunta original"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_rewriter.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/generation/rewriter.py`**

```python
from rag.config import REWRITE_MODEL
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat

_SYSTEM = (
    "Você reescreve a última pergunta do usuário como uma consulta de busca "
    "autocontida, resolvendo referências ao histórico da conversa e expandindo "
    "siglas quando útil. Responda APENAS com a consulta reescrita, sem aspas "
    "e sem explicações."
)


def rewrite_query(chat: GroqChat, question: str, history: list[dict]) -> str:
    messages = (
        [{"role": "system", "content": _SYSTEM}]
        + history[-6:]
        + [{"role": "user", "content": f"Pergunta: {question}\nConsulta de busca:"}]
    )
    try:
        out = chat.complete(REWRITE_MODEL, messages, max_tokens=100)
    except GenerationError:
        return question
    out = out.strip().strip('"').strip()
    return out or question
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_rewriter.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/generation/rewriter.py tests/test_rewriter.py
git commit -m "feat: history-aware query rewriting with graceful fallback"
```

---

### Task 14: Prompt de resposta com citações + gerador

**Files:**
- Create: `src/rag/generation/prompts.py`, `src/rag/generation/generator.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: `Chunk`, `GroqChat`, `rag.config.GENERATION_MODEL`
- Produces: em `rag.generation.prompts`: `build_context(chunks: list[Chunk]) -> str` (blocos `[n] (título, p. X)` separados por `---`) e `build_answer_messages(question: str, chunks: list[Chunk]) -> list[dict]`; em `rag.generation.generator`: `generate_answer(chat: GroqChat, question: str, chunks: list[Chunk]) -> str`. Constante `NO_ANSWER = "Não encontrei essa informação nos documentos."` em `prompts`.

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_generation.py
from rag.generation.generator import generate_answer
from rag.generation.groq_chat import GroqChat
from rag.generation.prompts import build_answer_messages, build_context
from rag.models import Chunk
from tests.fakes import FakeGroq


def _chunk(i, title, page, text):
    return Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title=title, page=page, position=i, text=text)


def test_build_context_numbers_and_cites_sources():
    ctx = build_context([
        _chunk(0, "Attention Paper", 3, "Self-attention relates positions."),
        _chunk(1, "BERT Paper", 7, "Masked language modeling."),
    ])
    assert "[1] (Attention Paper, p. 3)" in ctx
    assert "[2] (BERT Paper, p. 7)" in ctx
    assert "Self-attention relates positions." in ctx


def test_messages_contain_rules_and_question():
    messages = build_answer_messages("O que é atenção?", [_chunk(0, "T", 1, "txt")])
    assert messages[0]["role"] == "system"
    assert "Não encontrei essa informação" in messages[0]["content"]
    assert "O que é atenção?" in messages[1]["content"]


def test_generate_answer_calls_model():
    fake = FakeGroq(["A atenção é... [1]"])
    out = generate_answer(GroqChat(client=fake), "O que é atenção?", [_chunk(0, "T", 1, "txt")])
    assert out == "A atenção é... [1]"
    assert fake.calls[0]["model"]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_generation.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/generation/prompts.py`**

```python
from rag.models import Chunk

NO_ANSWER = "Não encontrei essa informação nos documentos."

_SYSTEM = f"""Você é um assistente de pesquisa que responde com base EXCLUSIVAMENTE \
no contexto fornecido (trechos de documentos numerados).

Regras:
1. Use apenas informações presentes no contexto. Não use conhecimento externo.
2. Cite a fonte de cada afirmação com o número entre colchetes, ex.: [1], [2].
3. Se o contexto não contém a resposta, diga exatamente: "{NO_ANSWER}"
4. Responda no idioma da pergunta."""


def build_context(chunks: list[Chunk]) -> str:
    blocks = [
        f"[{i}] ({c.doc_title}, p. {c.page})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def build_answer_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    user = f"Contexto:\n\n{build_context(chunks)}\n\nPergunta: {question}"
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
```

- [ ] **Step 4: Implementar `src/rag/generation/generator.py`**

```python
from rag.config import GENERATION_MODEL
from rag.generation.groq_chat import GroqChat
from rag.generation.prompts import build_answer_messages
from rag.models import Chunk


def generate_answer(chat: GroqChat, question: str, chunks: list[Chunk]) -> str:
    return chat.complete(GENERATION_MODEL, build_answer_messages(question, chunks))
```

- [ ] **Step 5: Rodar e ver passar**

Run: `pytest tests/test_generation.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/rag/generation/prompts.py src/rag/generation/generator.py tests/test_generation.py
git commit -m "feat: grounded answer prompt with numbered citations"
```

---

### Task 15: FeedbackDB (SQLite)

**Files:**
- Create: `src/rag/feedback/__init__.py`, `src/rag/feedback/db.py`
- Test: `tests/test_feedback_db.py`

**Interfaces:**
- Produces: classe `FeedbackDB` em `rag.feedback.db`: `__init__(path: Path | str)` (cria schema; aceita `":memory:"`); `log_interaction(query: str, rewritten_query: str, answer: str, sources: list[dict], model: str, latency_ms: int) -> int` (id); `add_feedback(interaction_id: int, rating: int, comment: str | None = None) -> None` (rating ∈ {-1, 1}); `metrics() -> dict` com chaves `total_questions: int`, `feedback_count: int`, `approval_rate: float | None`, `approval_rate_7d: float | None`, `avg_latency_ms: float | None`, `negatives: list[dict]` (interaction_id, query, answer, sources, created_at), `top_documents: list[dict]` (doc_title, citations); `close()`.

- [ ] **Step 1: Escrever testes que falham**

```python
# tests/test_feedback_db.py
import pytest

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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_feedback_db.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/feedback/db.py`** (criar `src/rag/feedback/__init__.py` vazio)

```python
import json
import sqlite3
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

    def close(self) -> None:
        self._conn.close()

    def log_interaction(self, query: str, rewritten_query: str, answer: str,
                        sources: list[dict], model: str, latency_ms: int) -> int:
        cur = self._conn.execute(
            "INSERT INTO interactions (query, rewritten_query, answer, sources, model, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (query, rewritten_query, answer, json.dumps(sources, ensure_ascii=False),
             model, latency_ms),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_feedback(self, interaction_id: int, rating: int, comment: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO feedback (interaction_id, rating, comment) VALUES (?, ?, ?)",
            (interaction_id, rating, comment),
        )
        self._conn.commit()

    def metrics(self) -> dict:
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_feedback_db.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/feedback tests/test_feedback_db.py
git commit -m "feat: sqlite feedback store with metrics aggregation"
```

---

### Task 16: RAGService (fachada do núcleo)

**Files:**
- Create: `src/rag/service.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `HybridRetriever`, `RetrievedChunk`, `rewrite_query`, `generate_answer`, `ingest_pdf`, `FeedbackDB`, `IndexStore`, `Embedder`, `Reranker`, `GroqChat`, `rag.config.GENERATION_MODEL`
- Produces: em `rag.service`: `@dataclass Source(doc_title: str, page: int, text: str, score: float)`; `@dataclass AskResult(interaction_id: int, answer: str, rewritten_query: str, sources: list[Source])`; classe `RAGService`: `__init__(store: IndexStore, embedder: Embedder, reranker: Reranker, chat: GroqChat, db: FeedbackDB, index_dir: Path, documents_dir: Path)`; `ask(question: str, history: list[dict] | None = None) -> AskResult`; `add_document(pdf_bytes: bytes, filename: str) -> int` (salva o PDF em `documents_dir`, ingere incrementalmente, persiste índice em `index_dir`); `feedback(interaction_id: int, rating: int, comment: str | None = None) -> None`; `metrics() -> dict`; `documents() -> list[dict]` (doc_id, doc_title, chunks).

- [ ] **Step 1: Escrever testes que falham** (tudo com fakes — sem modelos, sem rede)

```python
# tests/test_service.py
import numpy as np
import pytest

from rag.feedback.db import FeedbackDB
from rag.models import Chunk
from rag.retrieval.store import IndexStore
from rag.service import AskResult, RAGService
from rag.generation.groq_chat import GroqChat
from tests.fakes import FakeGroq


class FakeEmbedder:
    def embed_query(self, text):
        return np.eye(4, dtype="float32")[0]

    def embed_texts(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


class FakeReranker:
    def rerank(self, query, chunks, top_k):
        return [(c, 1.0) for c in chunks[:top_k]]


@pytest.fixture
def service(tmp_path):
    store = IndexStore(dim=4)
    chunks = [Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title="Doc D", page=i + 1,
                    position=i, text=f"chunk text {i}") for i in range(3)]
    store.add(chunks, np.eye(4, dtype="float32")[:3])
    chat = GroqChat(client=FakeGroq(["query reescrita", "resposta final [1]"]))
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=chat, db=db, index_dir=tmp_path / "index",
                     documents_dir=tmp_path / "docs")
    yield svc
    db.close()


def test_ask_returns_answer_with_sources_and_logs(service):
    result = service.ask("qual é o chunk?")
    assert isinstance(result, AskResult)
    assert result.answer == "resposta final [1]"
    assert result.rewritten_query == "query reescrita"
    assert result.sources and result.sources[0].doc_title == "Doc D"
    assert service.metrics()["total_questions"] == 1


def test_feedback_links_to_interaction(service):
    result = service.ask("pergunta")
    service.feedback(result.interaction_id, 1)
    assert service.metrics()["approval_rate"] == 1.0


def test_documents_lists_indexed(service):
    docs = service.documents()
    assert docs == [{"doc_id": "d", "doc_title": "Doc D", "chunks": 3}]


def test_add_document_persists_index(service, sample_pdf, tmp_path):
    added = service.add_document(sample_pdf.read_bytes(), "novo_paper.pdf")
    assert added > 0
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "docs" / "novo_paper.pdf").exists()
    assert any(d["doc_id"] == "novo_paper" for d in service.documents())
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_service.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/rag/service.py`**

```python
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.config import GENERATION_MODEL
from rag.feedback.db import FeedbackDB
from rag.generation.generator import generate_answer
from rag.generation.groq_chat import GroqChat
from rag.generation.rewriter import rewrite_query
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.embedder import Embedder
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.store import IndexStore


@dataclass
class Source:
    doc_title: str
    page: int
    text: str
    score: float


@dataclass
class AskResult:
    interaction_id: int
    answer: str
    rewritten_query: str
    sources: list[Source]


class RAGService:
    def __init__(self, store: IndexStore, embedder: Embedder, reranker: Reranker,
                 chat: GroqChat, db: FeedbackDB, index_dir: Path, documents_dir: Path):
        self.store = store
        self.embedder = embedder
        self.chat = chat
        self.db = db
        self.index_dir = index_dir
        self.documents_dir = documents_dir
        self.retriever = HybridRetriever(store, embedder, reranker)

    def ask(self, question: str, history: list[dict] | None = None) -> AskResult:
        start = time.perf_counter()
        rewritten = rewrite_query(self.chat, question, history or [])
        retrieved = self.retriever.retrieve(rewritten)
        answer = generate_answer(self.chat, question, [r.chunk for r in retrieved])
        sources = [Source(doc_title=r.chunk.doc_title, page=r.chunk.page,
                          text=r.chunk.text, score=r.score) for r in retrieved]
        latency_ms = int((time.perf_counter() - start) * 1000)
        interaction_id = self.db.log_interaction(
            query=question, rewritten_query=rewritten, answer=answer,
            sources=[asdict(s) for s in sources], model=GENERATION_MODEL,
            latency_ms=latency_ms,
        )
        return AskResult(interaction_id=interaction_id, answer=answer,
                         rewritten_query=rewritten, sources=sources)

    def add_document(self, pdf_bytes: bytes, filename: str) -> int:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        path = self.documents_dir / filename
        path.write_bytes(pdf_bytes)
        added = ingest_pdf(path, self.store, self.embedder)
        self.store.save(self.index_dir)
        return added

    def feedback(self, interaction_id: int, rating: int, comment: str | None = None) -> None:
        self.db.add_feedback(interaction_id, rating, comment)

    def metrics(self) -> dict:
        return self.db.metrics()

    def documents(self) -> list[dict]:
        counts = Counter((c.doc_id, c.doc_title) for c in self.store.chunks)
        return [{"doc_id": doc_id, "doc_title": title, "chunks": n}
                for (doc_id, title), n in sorted(counts.items())]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `pytest tests/test_service.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag/service.py tests/test_service.py
git commit -m "feat: RAG service facade wiring rewrite, retrieval, generation and logging"
```

---

### Task 17: API FastAPI

**Files:**
- Create: `src/api/schemas.py`, `src/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `RAGService`, `AskResult`, `Source`, exceções de `rag.errors`
- Produces: `create_app(service: RAGService | None = None) -> FastAPI` em `api.main` (factory; sem `service`, monta o real a partir do disco — modelos carregados aí). Rodar com `uvicorn "api.main:create_app" --factory`. Endpoints: `POST /ask` (`{question, history?}` → `{interaction_id, answer, rewritten_query, sources[]}`), `POST /upload` (multipart `file` → `{doc_id, chunks_added}`; 409 duplicado; 422 sem texto), `POST /feedback` (`{interaction_id, rating(1|-1), comment?}` → `{ok: true}`), `GET /metrics`, `GET /documents`, `GET /health` (`{status, documents, chunks}`). `GenerationError` → 503 com `{"detail": ...}`.

- [ ] **Step 1: Escrever testes que falham** (TestClient + service fake)

```python
# tests/test_api.py
import io

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from rag.errors import DuplicateDocumentError, ExtractionError, GenerationError
from rag.service import AskResult, Source


class FakeService:
    def __init__(self):
        self.fail_generation = False

    def ask(self, question, history=None):
        if self.fail_generation:
            raise GenerationError("down")
        return AskResult(interaction_id=1, answer="resp [1]", rewritten_query="rw",
                         sources=[Source(doc_title="T", page=2, text="x", score=0.5)])

    def add_document(self, pdf_bytes, filename):
        if filename == "dup.pdf":
            raise DuplicateDocumentError("dup")
        if filename == "bad.pdf":
            raise ExtractionError("sem texto")
        return 7

    def feedback(self, interaction_id, rating, comment=None):
        self.last_feedback = (interaction_id, rating, comment)

    def metrics(self):
        return {"total_questions": 0, "feedback_count": 0, "approval_rate": None,
                "approval_rate_7d": None, "avg_latency_ms": None,
                "negatives": [], "top_documents": []}

    def documents(self):
        return [{"doc_id": "d", "doc_title": "D", "chunks": 3}]


@pytest.fixture
def client():
    service = FakeService()
    app = create_app(service=service)
    return TestClient(app), service


def test_ask(client):
    c, _ = client
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "resp [1]"
    assert body["sources"][0]["doc_title"] == "T"


def test_ask_generation_down_returns_503(client):
    c, service = client
    service.fail_generation = True
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 503


def test_feedback_validates_rating(client):
    c, service = client
    ok = c.post("/feedback", json={"interaction_id": 1, "rating": 1})
    assert ok.status_code == 200
    assert service.last_feedback == (1, 1, None)
    bad = c.post("/feedback", json={"interaction_id": 1, "rating": 0})
    assert bad.status_code == 422


def test_upload_paths(client):
    c, _ = client
    pdf = ("file", ("ok.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[pdf]).status_code == 200
    dup = ("file", ("dup.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[dup]).status_code == 409
    bad = ("file", ("bad.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[bad]).status_code == 422


def test_health_documents_metrics(client):
    c, _ = client
    assert c.get("/health").json()["documents"] == 1
    assert c.get("/documents").json()[0]["doc_id"] == "d"
    assert "total_questions" in c.get("/metrics").json()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `pytest tests/test_api.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implementar `src/api/schemas.py`**

```python
from typing import Literal

from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


class SourceOut(BaseModel):
    doc_title: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    interaction_id: int
    answer: str
    rewritten_query: str
    sources: list[SourceOut]


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: Literal[1, -1]
    comment: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    chunks_added: int
```

- [ ] **Step 4: Implementar `src/api/main.py`**

```python
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile

from api.schemas import AskRequest, AskResponse, FeedbackRequest, UploadResponse
from rag.errors import DuplicateDocumentError, ExtractionError, GenerationError


def _build_real_service():
    from rag.config import DB_PATH, DOCUMENTS_DIR, INDEX_DIR
    from rag.feedback.db import FeedbackDB
    from rag.generation.groq_chat import GroqChat
    from rag.retrieval.embedder import Embedder
    from rag.retrieval.reranker import Reranker
    from rag.retrieval.store import IndexStore
    from rag.service import RAGService

    load_dotenv()
    return RAGService(
        store=IndexStore.load(INDEX_DIR),
        embedder=Embedder(),
        reranker=Reranker(),
        chat=GroqChat(),
        db=FeedbackDB(DB_PATH),
        index_dir=INDEX_DIR,
        documents_dir=DOCUMENTS_DIR,
    )


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="AI Research Assistant", version="0.1.0")
    app.state.service = service or _build_real_service()

    def svc():
        return app.state.service

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest):
        try:
            result = svc().ask(body.question, [m.model_dump() for m in body.history])
        except GenerationError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return result

    @app.post("/upload", response_model=UploadResponse)
    async def upload(file: UploadFile):
        data = await file.read()
        try:
            added = svc().add_document(data, file.filename)
        except DuplicateDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return UploadResponse(doc_id=Path(file.filename).stem, chunks_added=added)

    @app.post("/feedback")
    def feedback(body: FeedbackRequest):
        svc().feedback(body.interaction_id, body.rating, body.comment)
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        return svc().metrics()

    @app.get("/documents")
    def documents():
        return svc().documents()

    @app.get("/health")
    def health():
        docs = svc().documents()
        return {"status": "ok", "documents": len(docs),
                "chunks": sum(d["chunks"] for d in docs)}

    return app
```

- [ ] **Step 5: Rodar e ver passar**

Run: `pytest tests/test_api.py -v`
Expected: 5 PASS

- [ ] **Step 6: Commit**

```bash
git add src/api tests/test_api.py
git commit -m "feat: FastAPI endpoints over RAG service with error mapping"
```

---

### Task 18: Scripts de setup (download + build do índice)

**Files:**
- Create: `scripts/download_papers.py`, `scripts/build_index.py`
- Create: `data/documents/.gitkeep`

**Interfaces:**
- Consumes: `ingest_pdf`, `IndexStore`, `Embedder`, `rag.config` (`DOCUMENTS_DIR`, `INDEX_DIR`)
- Produces: `python scripts/download_papers.py` baixa os PDFs padrão para `data/documents/`; `python scripts/build_index.py` constrói e salva o índice em `data/index/`. Sem testes automatizados (rede/modelos reais) — verificação manual.

- [ ] **Step 1: Criar `scripts/download_papers.py`**

```python
"""Baixa os papers clássicos de IA (arXiv) usados como coleção padrão."""
from pathlib import Path

import requests

PAPERS = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762",
    "retrieval_augmented_generation": "https://arxiv.org/pdf/2005.11401",
    "bert": "https://arxiv.org/pdf/1810.04805",
    "gpt3_language_models_are_few_shot_learners": "https://arxiv.org/pdf/2005.14165",
    "dense_passage_retrieval": "https://arxiv.org/pdf/2004.04906",
}

DEST = Path("data/documents")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS.items():
        path = DEST / f"{name}.pdf"
        if path.exists():
            print(f"[skip] {name} (já existe)")
            continue
        print(f"[baixando] {name} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    print("Concluído.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Criar `scripts/build_index.py`**

```python
"""Constrói o índice (FAISS + chunks.json) a partir de data/documents/*.pdf."""
import sys
from pathlib import Path

from rag.config import DOCUMENTS_DIR, INDEX_DIR
from rag.errors import ExtractionError
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.embedder import Embedder
from rag.retrieval.store import IndexStore


def main() -> None:
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF em '{DOCUMENTS_DIR}'. Rode antes: python scripts/download_papers.py")

    print("Carregando modelo de embeddings...")
    embedder = Embedder()
    store = IndexStore()

    for pdf in pdfs:
        try:
            added = ingest_pdf(pdf, store, embedder)
            print(f"[ok] {pdf.name}: {added} chunks")
        except ExtractionError as exc:
            print(f"[erro] {pdf.name}: {exc}")

    store.save(INDEX_DIR)
    print(f"Índice salvo em '{INDEX_DIR}' ({len(store.chunks)} chunks).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Criar `data/documents/.gitkeep`** (arquivo vazio, para versionar a pasta)

- [ ] **Step 4: Verificação manual**

Run: `python scripts/download_papers.py` e depois `python scripts/build_index.py`
Expected: 5 PDFs baixados; índice salvo com centenas de chunks; sem erros.

- [ ] **Step 5: Fumaça da API real**

Run: `uvicorn "api.main:create_app" --factory` e `curl http://localhost:8000/health`
Expected: `{"status":"ok","documents":5,...}` (requer `.env` com GROQ_API_KEY para o `/ask`, mas `/health` funciona sem).

- [ ] **Step 6: Commit**

```bash
git add scripts data/documents/.gitkeep
git commit -m "feat: setup scripts to download default papers and build index"
```

---

### Task 19: UI Streamlit (chat, documentos, métricas)

**Files:**
- Create: `src/app/Home.py`, `src/app/api_client.py`
- Create: `src/app/pages/1_Documentos.py`, `src/app/pages/2_Metricas.py`

**Interfaces:**
- Consumes: API HTTP (`/ask`, `/upload`, `/feedback`, `/metrics`, `/documents`), env `API_URL` (default `http://localhost:8000`)
- Produces: app Streamlit multipage rodável com `streamlit run src/app/Home.py`. Sem testes automatizados (UI) — verificação manual guiada.

- [ ] **Step 1: Criar `src/app/api_client.py`**

```python
import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
_TIMEOUT = 120


class ApiError(Exception):
    pass


def _handle(response: requests.Response) -> dict | list:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise ApiError(detail)
    return response.json()


def ask(question: str, history: list[dict]) -> dict:
    return _handle(requests.post(f"{API_URL}/ask", timeout=_TIMEOUT,
                                 json={"question": question, "history": history}))


def upload(filename: str, data: bytes) -> dict:
    return _handle(requests.post(f"{API_URL}/upload", timeout=_TIMEOUT,
                                 files={"file": (filename, data, "application/pdf")}))


def send_feedback(interaction_id: int, rating: int) -> dict:
    return _handle(requests.post(f"{API_URL}/feedback", timeout=_TIMEOUT,
                                 json={"interaction_id": interaction_id, "rating": rating}))


def metrics() -> dict:
    return _handle(requests.get(f"{API_URL}/metrics", timeout=_TIMEOUT))


def documents() -> list:
    return _handle(requests.get(f"{API_URL}/documents", timeout=_TIMEOUT))
```

- [ ] **Step 2: Criar `src/app/Home.py`** (chat com fontes e feedback)

```python
import streamlit as st

from app.api_client import ApiError, ask, send_feedback

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Research Assistant")
st.caption("Pergunte sobre os papers indexados — respostas com citações [n].")

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources"?, "interaction_id"?}
if "voted" not in st.session_state:
    st.session_state.voted = set()


def _render_sources(sources):
    with st.expander(f"📄 Fontes ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['doc_title']}** — p. {s['page']} (score {s['score']:.2f})")
            st.text(s["text"][:500])


def _render_feedback(interaction_id):
    if interaction_id in st.session_state.voted:
        st.caption("Obrigado pelo feedback!")
        return
    col_up, col_down, _ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"up-{interaction_id}"):
        send_feedback(interaction_id, 1)
        st.session_state.voted.add(interaction_id)
        st.rerun()
    if col_down.button("👎", key=f"down-{interaction_id}"):
        send_feedback(interaction_id, -1)
        st.session_state.voted.add(interaction_id)
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])
        if message.get("interaction_id"):
            _render_feedback(message["interaction_id"])

if question := st.chat_input("Faça uma pergunta sobre os documentos..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]][-6:]
        try:
            with st.spinner("Buscando nos documentos..."):
                result = ask(question, history)
        except ApiError as exc:
            st.error(f"Não consegui responder agora: {exc}")
        else:
            st.markdown(result["answer"])
            _render_sources(result["sources"])
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"],
                "sources": result["sources"],
                "interaction_id": result["interaction_id"],
            })
            _render_feedback(result["interaction_id"])
```

- [ ] **Step 3: Criar `src/app/pages/1_Documentos.py`**

```python
import streamlit as st

from app.api_client import ApiError, documents, upload

st.title("📄 Documentos")

uploaded = st.file_uploader("Adicionar PDF à coleção", type=["pdf"])
if uploaded is not None and st.button("Indexar documento"):
    try:
        with st.spinner("Extraindo, chunkeando e indexando..."):
            result = upload(uploaded.name, uploaded.getvalue())
        st.success(f"'{result['doc_id']}' indexado: {result['chunks_added']} chunks.")
    except ApiError as exc:
        st.error(str(exc))

st.divider()
st.subheader("Coleção atual")
try:
    for doc in documents():
        st.markdown(f"- **{doc['doc_title']}** — {doc['chunks']} chunks")
except ApiError as exc:
    st.error(str(exc))
```

- [ ] **Step 4: Criar `src/app/pages/2_Metricas.py`**

```python
import streamlit as st

from app.api_client import ApiError, metrics

st.title("📊 Métricas")

try:
    data = metrics()
except ApiError as exc:
    st.error(str(exc))
    st.stop()


def _pct(value):
    return f"{value * 100:.0f}%" if value is not None else "—"


col1, col2, col3, col4 = st.columns(4)
col1.metric("Perguntas", data["total_questions"])
col2.metric("Aprovação", _pct(data["approval_rate"]))
col3.metric("Aprovação (7d)", _pct(data["approval_rate_7d"]))
col4.metric("Latência média",
            f"{data['avg_latency_ms']:.0f} ms" if data["avg_latency_ms"] else "—")

st.subheader("Perguntas com 👎 (fila de investigação)")
if not data["negatives"]:
    st.caption("Nenhum feedback negativo. 🎉")
for item in data["negatives"]:
    with st.expander(f"{item['created_at']} — {item['query']}"):
        st.markdown(item["answer"])
        st.json(item["sources"])

st.subheader("Documentos mais citados")
for doc in data["top_documents"]:
    st.markdown(f"- **{doc['doc_title']}** — {doc['citations']} citações")
```

- [ ] **Step 5: Verificação manual (com API rodando e índice construído)**

Run: `streamlit run src/app/Home.py`
Checklist manual:
- Pergunta "Quem propôs a arquitetura Transformer?" → resposta com [n] e painel de fontes com página.
- Pergunta fora do escopo ("qual a capital da Mongólia?") → "Não encontrei essa informação nos documentos."
- 👍 em uma resposta → página Métricas mostra aprovação.
- Upload de um PDF novo em Documentos → aparece na coleção; pergunta sobre ele responde.

- [ ] **Step 6: Commit**

```bash
git add src/app
git commit -m "feat: streamlit chat with citations, feedback and metrics dashboard"
```

---

### Task 20: Teste de integração E2E + README

**Files:**
- Create: `tests/test_e2e_retrieval.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `IndexStore`, `Embedder`, `Reranker`, `HybridRetriever`, `Chunk`, `rag.config.EMBEDDING_DIM`
- Produces: teste marcado `integration` validando retrieval semântico real (sem LLM); README com setup completo.

- [ ] **Step 1: Escrever `tests/test_e2e_retrieval.py`**

```python
import pytest

from rag.config import EMBEDDING_DIM
from rag.models import Chunk
from rag.retrieval.embedder import Embedder
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.store import IndexStore


@pytest.mark.integration
def test_semantic_retrieval_end_to_end():
    docs = {
        "transformers": "The Transformer architecture relies entirely on self-attention "
                        "mechanisms to model relationships between tokens in a sequence.",
        "cnn": "Convolutional neural networks apply learned filters over images to "
               "detect visual patterns such as edges and textures.",
        "rl": "Reinforcement learning agents interact with an environment and learn "
              "policies that maximize cumulative reward.",
    }
    chunks = [Chunk(chunk_id=f"{k}:0", doc_id=k, doc_title=k.title(), page=1,
                    position=0, text=v) for k, v in docs.items()]
    embedder = Embedder()
    store = IndexStore(dim=EMBEDDING_DIM)
    store.add(chunks, embedder.embed_texts([c.text for c in chunks]))

    retriever = HybridRetriever(store, embedder, Reranker(), top_k=2)
    results = retriever.retrieve("Which architecture is based on self-attention?")
    assert results[0].chunk.doc_id == "transformers"
```

- [ ] **Step 2: Rodar o teste de integração**

Run: `pytest tests/test_e2e_retrieval.py -v -m integration`
Expected: 1 PASS (primeira execução baixa os modelos MiniLM — precisa de rede)

- [ ] **Step 3: Rodar a suíte offline completa**

Run: `pytest -m "not integration" -v`
Expected: todos os testes das Tasks 1–17 PASS, sem rede.

- [ ] **Step 4: Escrever `README.md`**

````markdown
# 📚 AI Research Assistant — Advanced RAG

Assistente de pesquisa sobre papers clássicos de IA, construído **from scratch**
(sem LangChain) para demonstrar técnicas avançadas de RAG:

- **Busca híbrida** — FAISS (semântica) + BM25 (lexical) com **Reciprocal Rank Fusion**
- **Re-ranking** com cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **Reescrita de query** consciente do histórico do chat (Groq, Llama 3.1 8B)
- **Respostas com citações** `[n]` fundamentadas nos documentos (Groq, Llama 3.3 70B)
- **Feedback 👍/👎** persistido em SQLite + dashboard de métricas
- Upload incremental de PDFs

## Arquitetura

Streamlit (UI) → FastAPI (API) → núcleo RAG (biblioteca Python pura, testável isoladamente).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
copy .env.example .env                            # colocar sua GROQ_API_KEY

python scripts/download_papers.py                 # baixa 5 papers clássicos do arXiv
python scripts/build_index.py                     # constrói FAISS + BM25
```

## Rodando

```bash
uvicorn "api.main:create_app" --factory           # API em http://localhost:8000 (docs em /docs)
streamlit run src/app/Home.py                     # UI em http://localhost:8501
```

## Testes

```bash
pytest -m "not integration"    # suíte offline (LLM sempre mockado)
pytest -m integration          # retrieval real (baixa modelos na 1ª vez)
```

## Como funciona uma pergunta

1. A pergunta + histórico são reescritos em uma query de busca autocontida.
2. A query roda em FAISS (top-20) e BM25 (top-20); RRF funde os rankings.
3. O cross-encoder re-ranqueia os candidatos; sobram os top-5 chunks.
4. O LLM responde **apenas** com base nos chunks, citando `[n]` (doc + página).
5. A interação é registrada; o feedback do usuário alimenta o dashboard.
````

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_retrieval.py README.md
git commit -m "test: e2e semantic retrieval; docs: README with setup and architecture"
```

---

## Cobertura do spec → tasks

| Requisito do spec | Task |
|---|---|
| Extração PDF com página | 3 |
| Chunking ~800 tokens/overlap | 2 |
| Embeddings MiniLM | 7 |
| FAISS IndexFlatIP | 4 |
| BM25 | 5 |
| RRF | 6 |
| Re-ranking cross-encoder | 8 |
| Persistência índice + chunks.json | 9 |
| Ingestão incremental / upload | 11, 16, 17 |
| Reescrita de query + fallback | 13 |
| Geração com citações + anti-alucinação | 14 |
| Retry/backoff Groq | 12 |
| SQLite interactions/feedback + métricas | 15 |
| Endpoints API + erros HTTP | 17 |
| Scripts download/build | 18 |
| UI chat/upload/dashboard | 19 |
| Testes offline + e2e | 1–17, 20 |
| README/setup | 20 |
