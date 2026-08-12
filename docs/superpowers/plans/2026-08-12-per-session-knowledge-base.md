# Base de conhecimento por sessão — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cada visitante do demo hospedado ganha sua própria base de conhecimento, de modo que subir, remover ou limpar documentos numa aba não afete nenhuma outra.

**Architecture:** `RAGService` ganha um flag `persist` — com `persist=False` nenhuma escrita toca o disco, o que transforma `data/index` num molde somente leitura. O ramo embutido do `backend.py` deixa de guardar um `RAGService` único no `st.cache_resource` e passa a guardar só os recursos caros e sem estado (embedder, reranker, chat, banco de feedback); o `IndexStore` e o `RAGService` mudam para `st.session_state`, um por sessão, versionados contra a armadilha de módulo velho do Streamlit Cloud.

**Tech Stack:** Python 3.11+, pytest, Streamlit, FAISS, FastAPI (modo local, inalterado).

## Global Constraints

- Spec de referência: `docs/superpowers/specs/2026-08-12-per-session-knowledge-base-design.md`
- Rodar os testes com `.venv/Scripts/python.exe -m pytest -q -m "not integration"` (Windows; venv na raiz do repositório). 139 passam hoje.
- A suíte roda offline: nenhum teste pode chamar o Groq, a rede, nem carregar modelos reais (`Embedder`/`Reranker`)
- Uma advertência pré-existente, `StarletteDeprecationWarning` vinda de `fastapi/testclient.py`, é conhecida e não conta como falha; qualquer advertência nova conta
- `persist=True` é o default: o modo HTTP local e toda a suíte existente têm que continuar idênticos
- O rate limit global (`_RATE_LIMIT = 10` por minuto) continua global de propósito — ele protege a cota única do Groq, não o visitante
- O banco de feedback continua compartilhado entre sessões
- Comentários e nomes de código em inglês, seguindo o resto do repositório

---

### Task 1: Flag `persist` no RAGService

**Files:**
- Modify: `src/rag/service.py` — `__init__` (linha 39), `add_document` (linha 80), `remove_document` (linha 106), `reset_documents` (linha 116)
- Test: `tests/test_service.py` (acrescentar helper e testes ao final)

**Interfaces:**
- Consumes: nada de tarefas anteriores
- Produces: `RAGService(..., persist: bool = True)`. Com `persist=False`, nenhum método escreve em `index_dir`, e `add_document` apaga o PDF de `documents_dir` assim que termina de indexar. Com `persist=True` o comportamento é idêntico ao de hoje.

- [ ] **Step 1: Write the failing tests**

Acrescentar ao final de `tests/test_service.py`. Os imports `numpy as np`, `pytest`, `Chunk`, `IndexStore`, `RAGService`, `GroqChat`, `FeedbackDB`, `FakeGroq` e as classes `FakeEmbedder`/`FakeReranker` já existem no topo do arquivo; a fixture `sample_pdf` vem de `tests/conftest.py`.

```python
def _persist_service(tmp_path, persist):
    store = IndexStore(dim=4)
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=GroqChat(client=FakeGroq([])), db=db,
                     index_dir=tmp_path / "index",
                     documents_dir=tmp_path / "docs", persist=persist)
    return svc, db


def test_ephemeral_add_document_never_writes_the_index(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=False)
    assert svc.add_document(sample_pdf.read_bytes(), "a.pdf") > 0
    assert not (tmp_path / "index").exists()
    db.close()


def test_ephemeral_add_document_deletes_the_pdf_after_indexing(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=False)
    svc.add_document(sample_pdf.read_bytes(), "a.pdf")
    assert list((tmp_path / "docs").glob("*.pdf")) == []
    db.close()


def test_ephemeral_add_document_deletes_the_pdf_even_when_ingestion_fails(tmp_path):
    svc, db = _persist_service(tmp_path, persist=False)
    with pytest.raises(ExtractionError):
        svc.add_document(b"not a pdf at all", "bad.pdf")
    assert list((tmp_path / "docs").glob("*.pdf")) == []
    db.close()


def test_ephemeral_remove_and_reset_never_write_the_index(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=False)
    svc.add_document(sample_pdf.read_bytes(), "a.pdf")
    svc.add_document(sample_pdf.read_bytes(), "b.pdf")
    svc.remove_document("a")
    svc.reset_documents()
    assert not (tmp_path / "index").exists()
    db.close()


def test_persisting_service_still_writes_the_index_and_keeps_the_pdf(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=True)
    svc.add_document(sample_pdf.read_bytes(), "a.pdf")
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "docs" / "a.pdf").exists()
    db.close()
```

`ExtractionError` ainda não está importado nesse arquivo. Trocar a linha de import existente por:

```python
from rag.errors import (DocumentNotFoundError, DuplicateDocumentError,
                        EmptyIndexError, ExtractionError, GenerationError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_service.py -k "ephemeral or persisting" -v`
Expected: FAIL — `TypeError: RAGService.__init__() got an unexpected keyword argument 'persist'`

- [ ] **Step 3: Add the flag and the save seam**

Em `src/rag/service.py`, na assinatura do `__init__`:

```python
    def __init__(self, store: IndexStore, embedder: Embedder, reranker: Reranker,
                 chat: GroqChat, db: FeedbackDB, index_dir: Path, documents_dir: Path,
                 persist: bool = True):
```

e no corpo, junto das outras atribuições (depois de `self.documents_dir = documents_dir`):

```python
        # An ephemeral service (persist=False) is scoped to one visitor's
        # session: it must never write to the shared index on disk, which is
        # the read-only template every session is loaded from.
        self.persist = persist
```

Acrescentar o seam logo antes de `add_document`:

```python
    def _save(self) -> None:
        if self.persist:
            self.store.save(self.index_dir)
```

- [ ] **Step 4: Route every write through the seam**

Em `add_document`, trocar o bloco da indexação em diante por:

```python
        path = self.documents_dir / safe_name
        path.write_bytes(pdf_bytes)
        try:
            added = ingest_pdf(path, self.store, self.embedder)
        finally:
            if not self.persist:
                # ingest_pdf needs a file on disk, but once the chunks are in
                # memory the PDF has no further use in an ephemeral session.
                path.unlink(missing_ok=True)
        self._save()
        return added
```

Em `remove_document`, trocar `self.store.save(self.index_dir)` por `self._save()`.

Em `reset_documents`, trocar `self.store.save(self.index_dir)` por `self._save()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_service.py -v`
Expected: PASS — os 5 testes novos e todos os que já existiam

- [ ] **Step 6: Run the whole offline suite**

Run: `.venv/Scripts/python.exe -m pytest -q -m "not integration"`
Expected: PASS — 144 passando, sem advertência nova

- [ ] **Step 7: Commit**

```bash
git add src/rag/service.py tests/test_service.py
git commit -m "feat: persist flag so an ephemeral service never writes to disk"
```

---

### Task 2: Serviço por sessão no backend embutido

**Files:**
- Modify: `src/app/backend.py` — imports do ramo embutido (linha 25-33), `_cached_service`/`_build_service` (linhas 87-103), `restore_defaults` (linha 145)
- Test: `tests/test_backend_embedded.py` (acrescentar fixtures e testes ao final)

**Interfaces:**
- Consumes: `RAGService(..., persist: bool = True)` da Task 1
- Produces: `_session_cache() -> dict` (indireção sobre `st.session_state`, trocável nos testes); `_shared_resources(code_version: str) -> dict` com as chaves `embedder`, `reranker`, `chat`, `db`; `_new_session_service(version: str) -> RAGService`; `_build_service()` continua com o mesmo nome e agora devolve o serviço da sessão corrente

- [ ] **Step 1: Write the failing tests**

Acrescentar ao final de `tests/test_backend_embedded.py` (o módulo já importa `importlib` e `pytest` no topo):

```python
class _Nothing:
    """Stand-in for the shared, stateless resources — never called in these tests."""


@pytest.fixture
def session_backend(embedded_backend, monkeypatch, tmp_path):
    import numpy as np

    from rag.models import Chunk
    from rag.retrieval.store import IndexStore

    index_dir = tmp_path / "index"
    store = IndexStore(dim=4)
    store.add([Chunk(chunk_id="d:0", doc_id="d", doc_title="Doc D", page=1,
                     position=0, text="texto")], np.eye(4, dtype="float32")[:1])
    store.save(index_dir)
    monkeypatch.setattr("rag.config.INDEX_DIR", index_dir)
    monkeypatch.setattr(embedded_backend, "_shared_resources",
                        lambda code_version: {"embedder": _Nothing(),
                                              "reranker": _Nothing(),
                                              "chat": _Nothing(), "db": _Nothing()})
    return embedded_backend


def test_each_session_gets_its_own_store(session_backend):
    first = session_backend._new_session_service("v1")
    second = session_backend._new_session_service("v1")
    assert first.store is not second.store
    assert first.documents_dir != second.documents_dir
    assert [c.doc_id for c in first.store.chunks] == ["d"]
    assert [c.doc_id for c in second.store.chunks] == ["d"]


def test_the_session_service_is_ephemeral(session_backend):
    assert session_backend._new_session_service("v1").persist is False


def test_two_sessions_do_not_see_each_other(session_backend, monkeypatch):
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    tab_a, tab_b = {}, {}

    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab_a)
    service_a = session_backend._build_service()
    service_a.store.clear()

    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab_b)
    service_b = session_backend._build_service()

    assert service_a is not service_b
    assert service_a.store.chunks == []
    assert [c.doc_id for c in service_b.store.chunks] == ["d"]


def test_the_same_session_reuses_its_service(session_backend, monkeypatch):
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    tab = {}
    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab)
    assert session_backend._build_service() is session_backend._build_service()


def test_a_new_code_version_rebuilds_the_session_service(session_backend, monkeypatch):
    tab = {}
    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab)
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    first = session_backend._build_service()
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v2")
    assert session_backend._build_service() is not first


def test_restore_defaults_rebuilds_from_disk_without_downloading(session_backend,
                                                                monkeypatch):
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    tab = {}
    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab)
    session_backend._hits.clear()
    session_backend._build_service().store.clear()

    def _no_download(*args, **kwargs):
        raise AssertionError("restore_defaults must not hit the network")

    monkeypatch.setattr("rag.ingestion.default_papers.fetch_default_papers",
                        _no_download)
    assert session_backend.restore_defaults() == {"documents_added": 1,
                                                  "chunks_added": 1}
    assert [c.doc_id for c in session_backend._build_service().store.chunks] == ["d"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend_embedded.py -k "session or restore_defaults_rebuilds" -v`
Expected: FAIL — `AttributeError: module 'app.backend' has no attribute '_shared_resources'`

- [ ] **Step 3: Add the imports the session service needs**

Em `src/app/backend.py`, no bloco de imports do ramo embutido (o que começa com `import streamlit as st`), acrescentar `tempfile` mantendo a ordem alfabética existente:

```python
    import streamlit as st
    import hashlib
    import sys
    import tempfile
    import threading
    import time
    from dataclasses import asdict
    from pathlib import Path
```

- [ ] **Step 4: Split the cached service into shared resources plus a session service**

Trocar o bloco `_cached_service` / `_build_service` (do comentário `# code_version (not "version")` até o fim de `_build_service`) por:

```python
    def _session_cache() -> dict:
        # Indirection over st.session_state: outside a Streamlit runtime it
        # still works but is process-global, so the tests swap this for a
        # plain dict — one dict per simulated tab is what makes the isolation
        # testable at all.
        return st.session_state

    # code_version (not "version"): the rename shifts this function's cache
    # hash, orphaning entries a pre-purge deploy built from stale modules
    @st.cache_resource(show_spinner="Loading models (first visit only)...")
    def _shared_resources(code_version: str) -> dict:
        # Only the expensive, stateless pieces are shared across visitors.
        # The index is not: it is what each visitor changes.
        _bridge_secrets()
        from rag.config import DB_PATH
        from rag.feedback.db import FeedbackDB
        from rag.generation.groq_chat import GroqChat
        from rag.retrieval.embedder import Embedder
        from rag.retrieval.reranker import Reranker

        return {"embedder": Embedder(), "reranker": Reranker(),
                "chat": GroqChat(), "db": FeedbackDB(DB_PATH)}

    def _new_session_service(version: str):
        from rag.config import INDEX_DIR
        from rag.retrieval.store import IndexStore
        from rag.service import RAGService

        shared = _shared_resources(version)
        # A per-session temp dir keeps two visitors uploading the same file
        # name from colliding; add_document deletes the PDF right after
        # indexing, so the directory stays empty.
        return RAGService(store=IndexStore.load(INDEX_DIR),
                          embedder=shared["embedder"], reranker=shared["reranker"],
                          chat=shared["chat"], db=shared["db"],
                          index_dir=INDEX_DIR,
                          documents_dir=Path(tempfile.mkdtemp(prefix="rag-session-")),
                          persist=False)

    def _build_service():
        version = _ensure_fresh_rag()
        cache = _session_cache()
        # (version, service): a session service survives cache_resource.clear(),
        # so without the version check it would keep classes from a generation
        # of the rag package that _ensure_fresh_rag already purged.
        entry = cache.get(_SESSION_KEY)
        if entry is None or entry[0] != version:
            entry = (version, _new_session_service(version))
            cache[_SESSION_KEY] = entry
        return entry[1]
```

E declarar a chave junto das outras constantes do módulo, logo acima de `_RATE_LIMIT`:

```python
    _SESSION_KEY = "rag_session_service"
```

- [ ] **Step 5: Make restore_defaults rebuild the session**

Trocar a função `restore_defaults` inteira por:

```python
    def restore_defaults() -> dict:
        _check_rate()
        _ensure_fresh_rag()
        # An ephemeral session restores by starting over from the read-only
        # index on disk: instant, and no arXiv round trip.
        cache = _session_cache()
        if _SESSION_KEY in cache:
            del cache[_SESSION_KEY]
        docs = _service_or_api_error().documents()
        return {"documents_added": len(docs),
                "chunks_added": sum(d["chunks"] for d in docs)}
```

As importações de `DownloadError`, `DuplicateDocumentError` e `ExtractionError` que estavam dentro dessa função saem junto — nenhuma delas pode mais ocorrer aqui.

- [ ] **Step 6: Update the one existing test this changes**

`test_embedded_document_management` (em `tests/test_backend_embedded.py`) afirma hoje que `restore_defaults()` devolve `{"documents_added": 5, "chunks_added": 42}`, vindo de `FakeService.restore_default_documents`. A partir daqui `restore_defaults` não delega mais ao serviço: ela descarta a sessão e conta o que voltou do disco. Com a `FakeService`, `documents()` devolve um documento de 3 chunks.

Além disso, `restore_defaults` passa a chamar `_session_cache()`. Fora de um runtime do Streamlit o `st.session_state` funciona, mas é **global do processo** — sem trocar por um dicionário, este teste vazaria estado para os outros. Trocar o teste inteiro por:

```python
def test_embedded_document_management(embedded_backend, fake_service, monkeypatch):
    embedded_backend._hits.clear()
    monkeypatch.setattr(embedded_backend, "_session_cache", lambda: {})
    assert embedded_backend.remove_document("d") == {"doc_id": "d", "chunks_removed": 2}
    assert embedded_backend.reset_documents() == {"chunks_removed": 5}
    assert embedded_backend.restore_defaults() == {"documents_added": 1,
                                                   "chunks_added": 3}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend_embedded.py -v`
Expected: PASS — os 6 testes novos, o teste alterado e todos os demais

- [ ] **Step 8: Run the whole offline suite**

Run: `.venv/Scripts/python.exe -m pytest -q -m "not integration"`
Expected: PASS — 150 passando, sem advertência nova

- [ ] **Step 9: Commit**

```bash
git add src/app/backend.py tests/test_backend_embedded.py
git commit -m "feat: give each visitor session its own knowledge base"
```

---

### Task 3: Aviso na página Documentos

**Files:**
- Modify: `src/app/views/documents.py:10` (logo abaixo do hero)
- Modify: `src/app/translations.py` (nova chave junto das outras da página Documentos)

**Interfaces:**
- Consumes: nada — é só copy
- Produces: nada

- [ ] **Step 1: Add the translation key**

Em `src/app/translations.py`, acrescentar junto das chaves da página Documentos (perto de `upload_label`):

```python
    "session_scope_note": {
        "en": "This knowledge base belongs to this browser tab and lasts while "
              "it stays open. Other visitors have their own; reloading starts "
              "over from the default collection.",
        "pt": "Esta base é desta aba do navegador e vale enquanto ela estiver "
              "aberta. Outros visitantes têm a sua; recarregar recomeça da "
              "coleção padrão.",
    },
```

- [ ] **Step 2: Render it**

Em `src/app/views/documents.py`, logo após a linha do `hero(...)`:

```python
st.caption(":material/info: " + t("session_scope_note", lang))
```

- [ ] **Step 3: Verify it compiles and the suite is still green**

Run: `.venv/Scripts/python.exe -m py_compile src/app/views/documents.py src/app/translations.py`
Expected: sem saída

Run: `.venv/Scripts/python.exe -m pytest -q -m "not integration"`
Expected: PASS — 150 passando

- [ ] **Step 4: Commit**

```bash
git add src/app/views/documents.py src/app/translations.py
git commit -m "feat: tell visitors the knowledge base is scoped to their tab"
```

---

## Verificação manual (o Filip roda, precisa da GROQ_API_KEY)

Estes são os critérios de sucesso que nenhum teste offline cobre. Subir a UI em modo embutido:

```bash
BACKEND_MODE=embedded .venv/Scripts/python.exe -m streamlit run src/app/Home.py
```

1. Abrir o app em duas abas. Subir um PDF na aba A; a lista de documentos da aba B não muda e as respostas dela não citam o PDF novo.
2. Clicar em "Limpar tudo" na aba A; a aba B continua com os 5 papers.
3. Recarregar a aba A: volta a coleção padrão, e o aviso na página Documentos explica isso.
4. `git status` em `data/index` depois de tudo: sem modificação.
5. "Restaurar coleção padrão" responde na hora, sem barra de download.

## Notas de implantação

Esta mudança acrescenta nomes novos a `app.backend` (`_session_cache`, `_shared_resources`, `_new_session_service`) e remove `_cached_service`. Um deploy do Streamlit Cloud que reaproveite o processo pode manter o módulo velho em `sys.modules` — o mesmo `ImportError` de 2026-08-12. Depois do push, use **Manage app → ⋮ → Reboot app**, ou salve os Secrets (o que também reinicia).
