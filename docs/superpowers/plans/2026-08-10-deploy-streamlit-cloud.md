# Deploy: Streamlit Community Cloud — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Task 3 is interactive** — the controller executes it with the user.

**Goal:** Demo pública gratuita no Streamlit Community Cloud, com a UI rodando o núcleo RAG em processo (modo embutido) e o comportamento local (HTTP) intacto.

**Architecture:** Novo `src/app/backend.py` chaveado por `BACKEND_MODE` (env ou st.secrets): default reexporta o `api_client` HTTP; `embedded` monta o `RAGService` via `st.cache_resource` com rate limit global e mapeamento de exceções para `ApiError`. Índice commitado no repo; `requirements.txt` com torch CPU.

**Tech Stack:** Streamlit Community Cloud, código existente. Nenhuma dependência nova.

**Spec:** `docs/superpowers/specs/2026-08-10-deploy-streamlit-cloud-design.md`

## Global Constraints

- `BACKEND_MODE`: valores `http` (default) e `embedded`; lido de `os.environ` E de `st.secrets` (secrets vencem se ambos).
- Modo embutido espelha a API: dicts com o MESMO shape do JSON; rate limit global 10/min (ask+upload somados) com a mensagem exata `"Rate limit exceeded — try again in a minute."`; pergunta >500 chars → `ApiError("Question too long (max 500 characters).")`.
- Exceções do núcleo (`GenerationError`, `DuplicateDocumentError`, `ExtractionError`, `ValueError`) viram `ApiError(str(exc))` no modo embutido.
- Modo http: comportamento byte-idêntico ao atual (reexport puro).
- `requirements.txt` exato: 3 linhas (`--extra-index-url https://download.pytorch.org/whl/cpu`, `torch==2.13.0+cpu`, `-e .`).
- Índice versionado: `data/index/index.faiss` + `data/index/chunks.json` (só esses; `feedback.db` e PDFs continuam ignorados).
- Suíte offline verde. Commits `feat:`/`chore:`/`docs:` em inglês.

---

### Task 1: Backend embutido + troca de imports nas páginas

**Files:**
- Create: `src/app/backend.py`
- Modify: `src/app/Home.py`, `src/app/pages/1_Documents.py`, `src/app/pages/2_Metrics.py` (só a linha de import)
- Test: `tests/test_backend_embedded.py`

**Interfaces:**
- Consumes: `app.api_client` (ApiError, ApiConnectionError e funções HTTP), `rag.service.RAGService`, `rag.errors.*`, `rag.config.*`
- Produces: `app.backend` com `ApiError`, `ApiConnectionError`, `ask(question, history, language="en") -> dict`, `upload(filename, data) -> dict`, `send_feedback(interaction_id, rating) -> dict`, `metrics() -> dict`, `documents() -> list`. As páginas importam APENAS de `app.backend`.

- [ ] **Step 1: Escrever testes que falham** — `tests/test_backend_embedded.py`:

```python
import importlib

import pytest


@pytest.fixture
def embedded_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "embedded")
    import app.backend as backend
    importlib.reload(backend)
    yield backend
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    importlib.reload(backend)


class FakeService:
    def __init__(self):
        self.fail = False

    def ask(self, question, history=None, language="en"):
        from rag.errors import GenerationError
        if self.fail:
            raise GenerationError("down")
        from rag.service import AskResult, Source
        return AskResult(interaction_id=7, answer="resp [1]", rewritten_query="rw",
                         sources=[Source(doc_title="T", page=2, text="x", score=0.5)])

    def add_document(self, data, filename):
        return 3

    def feedback(self, interaction_id, rating, comment=None):
        self.last = (interaction_id, rating)

    def metrics(self):
        return {"total_questions": 1}

    def documents(self):
        return [{"doc_id": "d", "doc_title": "D", "chunks": 3}]


@pytest.fixture
def fake_service(embedded_backend, monkeypatch):
    service = FakeService()
    monkeypatch.setattr(embedded_backend, "_build_service", lambda: service)
    return service


def test_http_is_default(monkeypatch):
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    import app.backend as backend
    importlib.reload(backend)
    import app.api_client as api_client
    assert backend.ask is api_client.ask


def test_embedded_ask_returns_api_shaped_dict(embedded_backend, fake_service):
    out = embedded_backend.ask("q?", [], "en")
    assert out["interaction_id"] == 7
    assert out["answer"] == "resp [1]"
    assert out["sources"][0]["doc_title"] == "T"


def test_embedded_generation_error_becomes_api_error(embedded_backend, fake_service):
    fake_service.fail = True
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.ask("q?", [], "en")


def test_embedded_long_question_rejected(embedded_backend, fake_service):
    with pytest.raises(embedded_backend.ApiError, match="too long"):
        embedded_backend.ask("x" * 501, [], "en")


def test_embedded_rate_limit(embedded_backend, fake_service, monkeypatch):
    monkeypatch.setattr(embedded_backend, "_RATE_LIMIT", 2)
    embedded_backend._hits.clear()
    embedded_backend.ask("q?", [], "en")
    embedded_backend.upload("a.pdf", b"%PDF")
    with pytest.raises(embedded_backend.ApiError, match="Rate limit"):
        embedded_backend.ask("q?", [], "en")


def test_embedded_upload_and_misc(embedded_backend, fake_service):
    embedded_backend._hits.clear()
    out = embedded_backend.upload("novo_doc.pdf", b"%PDF")
    assert out == {"doc_id": "novo_doc", "chunks_added": 3}
    assert embedded_backend.send_feedback(7, 1) == {"ok": True}
    assert embedded_backend.metrics()["total_questions"] == 1
    assert embedded_backend.documents()[0]["doc_id"] == "d"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv\Scripts\pytest tests/test_backend_embedded.py -v`
Expected: FAIL (ModuleNotFoundError: app.backend)

- [ ] **Step 3: Implementar `src/app/backend.py`**

```python
"""Backend switch for the UI: HTTP (default) or embedded RAG core.

Set BACKEND_MODE=embedded (env var or st.secrets) to run the RAG service
in-process — used by the hosted demo on Streamlit Community Cloud, where a
separate FastAPI process is not available.
"""
import os


def _mode() -> str:
    mode = os.environ.get("BACKEND_MODE", "http")
    try:
        import streamlit as st
        mode = st.secrets.get("BACKEND_MODE", mode)
    except Exception:
        pass
    return mode


if _mode() != "embedded":
    from app.api_client import (ApiConnectionError, ApiError, ask, documents,
                                metrics, send_feedback, upload)
else:
    import threading
    import time
    from dataclasses import asdict
    from pathlib import Path

    from app.api_client import ApiConnectionError, ApiError
    from rag.errors import DuplicateDocumentError, ExtractionError, GenerationError

    _RATE_LIMIT = 10
    _RATE_WINDOW_S = 60
    _hits: list[float] = []
    _hits_lock = threading.Lock()

    def _check_rate() -> None:
        now = time.monotonic()
        with _hits_lock:
            _hits[:] = [t for t in _hits if now - t < _RATE_WINDOW_S]
            if len(_hits) >= _RATE_LIMIT:
                raise ApiError("Rate limit exceeded — try again in a minute.")
            _hits.append(now)

    def _bridge_secrets() -> None:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
        except Exception:
            pass

    def _build_service():
        import streamlit as st

        @st.cache_resource(show_spinner="Loading models and index (first visit only)...")
        def _service():
            _bridge_secrets()
            from rag.config import DB_PATH, DOCUMENTS_DIR, INDEX_DIR
            from rag.feedback.db import FeedbackDB
            from rag.generation.groq_chat import GroqChat
            from rag.retrieval.embedder import Embedder
            from rag.retrieval.reranker import Reranker
            from rag.retrieval.store import IndexStore
            from rag.service import RAGService

            return RAGService(store=IndexStore.load(INDEX_DIR), embedder=Embedder(),
                              reranker=Reranker(), chat=GroqChat(), db=FeedbackDB(DB_PATH),
                              index_dir=INDEX_DIR, documents_dir=DOCUMENTS_DIR)

        return _service()

    def ask(question: str, history: list[dict], language: str = "en") -> dict:
        _check_rate()
        if len(question) > 500:
            raise ApiError("Question too long (max 500 characters).")
        try:
            result = _build_service().ask(question, history, language)
        except GenerationError as exc:
            raise ApiError(str(exc)) from exc
        return {"interaction_id": result.interaction_id, "answer": result.answer,
                "rewritten_query": result.rewritten_query,
                "sources": [asdict(s) for s in result.sources]}

    def upload(filename: str, data: bytes) -> dict:
        _check_rate()
        try:
            added = _build_service().add_document(data, filename)
        except (DuplicateDocumentError, ExtractionError, ValueError) as exc:
            raise ApiError(str(exc)) from exc
        return {"doc_id": Path(filename).stem, "chunks_added": added}

    def send_feedback(interaction_id: int, rating: int) -> dict:
        _build_service().feedback(interaction_id, rating)
        return {"ok": True}

    def metrics() -> dict:
        return _build_service().metrics()

    def documents() -> list:
        return _build_service().documents()
```

- [ ] **Step 4: Trocar imports nas 3 páginas** — em `src/app/Home.py`: `from app.backend import ApiConnectionError, ApiError, ask, send_feedback`; em `src/app/pages/1_Documents.py`: `from app.backend import ApiConnectionError, ApiError, documents, upload`; em `src/app/pages/2_Metrics.py`: `from app.backend import ApiConnectionError, ApiError, metrics`. Nada mais muda nas páginas.

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv\Scripts\pytest tests/test_backend_embedded.py -v` e depois `.venv\Scripts\pytest -m "not integration" -q`
Expected: 6 novos PASS; suíte inteira verde. Também: `py_compile` nas 3 páginas + `backend.py`.

- [ ] **Step 6: Commit**

```bash
git add src/app tests/test_backend_embedded.py
git commit -m "feat: embedded backend mode for hosted demo (Streamlit Community Cloud)"
```

---

### Task 2: requirements.txt + índice versionado + README + remoção do workflow HF

**Files:**
- Create: `requirements.txt`
- Modify: `.gitignore` (remover linha `data/index/`)
- Add: `data/index/index.faiss`, `data/index/chunks.json` (artefatos existentes localmente)
- Modify: `README.md`
- Delete: `.github/workflows/deploy-to-hf.yml`

- [ ] **Step 1: Criar `requirements.txt`** (exatamente 3 linhas)

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.13.0+cpu
-e .
```

- [ ] **Step 2: Versionar o índice** — remover a linha `data/index/` do `.gitignore`; `git add data/index/index.faiss data/index/chunks.json`. Confirmar que `data/feedback.db` e `data/documents/*.pdf` continuam ignorados (`git status` não deve listá-los).

- [ ] **Step 3: Atualizar `README.md`**
- Remover o bloco de front-matter YAML do topo (as 9 linhas `---` ... `---` do HF Space)
- Trocar a linha Live Demo por: `**🔴 Live demo:** <https://ai-research-assistant-turazzi.streamlit.app> — no setup needed, just open and ask. *(First visit after idle may take ~1-2 min while the app wakes up.)*` (URL será confirmada/ajustada na Task 3)
- Na nota do demo hospedado, trocar a menção ao Space por: `> **Note on the hosted demo:** it runs the RAG core in-process (embedded mode) on Streamlit Community Cloud; the FastAPI layer runs locally and in Docker. Uploaded PDFs and feedback are cleared whenever the app restarts. Ask up to 10 questions per minute (public rate limit).`

- [ ] **Step 4: Deletar `.github/workflows/deploy-to-hf.yml`** (`git rm`)

- [ ] **Step 5: Verificação** — `.venv\Scripts\pytest -m "not integration" -q` → verde; `grep -n "huggingface" README.md || echo CLEAN` → CLEAN.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore data/index README.md
git commit -m "chore: Streamlit Community Cloud packaging — requirements, versioned index, README"
```

---

### Task 3: Publicação (INTERATIVA — controller + usuário)

- [ ] **Step 1:** Push da `master` para o GitHub (`git push`)
- [ ] **Step 2:** Usuário: login em <https://share.streamlit.io> com GitHub; **Create app** → repo `filippoturazzi/ai-research-assistant`, branch `master`, main file `src/app/Home.py`; em **Advanced settings → Secrets**, colar:

```toml
GROQ_API_KEY = "<chave do usuário>"
BACKEND_MODE = "embedded"
```

- [ ] **Step 3:** Aguardar o build (instala requirements + baixa modelos ~200MB; 5-10 min na primeira vez). Acompanhar logs no painel do app.
- [ ] **Step 4:** Testar: pergunta EN com citações; seletor PT; pergunta fora do escopo; 👍/👎 + página Metrics; 11 perguntas/min → rate limit amigável.
- [ ] **Step 5:** Confirmar a URL real do app; se diferente do README, atualizar o link, commit (`docs: point live demo to the deployed app URL`) e push (redeploy automático).
- [ ] **Step 6:** Validar os critérios de sucesso do spec (1-5).

---

## Cobertura do spec → tasks

| Requisito do spec | Task |
|---|---|
| backend.py chaveado (http default / embedded) + shape da API + rate limit + exceções | 1 |
| Páginas importando de app.backend | 1 |
| requirements.txt (torch CPU) | 2 |
| Índice versionado | 2 |
| README (demo URL, embedded note, sem HF) | 2, 3 (URL real) |
| Remoção do workflow HF | 2 |
| Publicação + secrets + testes ao vivo | 3 |
