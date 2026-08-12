# Sugestões de perguntas derivadas da base — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** As pills de "experimente perguntar" no chat passam a ser geradas a partir dos documentos indexados, mudando sozinhas sempre que a base muda.

**Architecture:** Um módulo novo no núcleo (`rag/generation/suggestions.py`) pede 3 perguntas ao Groq a partir de trechos dos documentos e cai num fallback determinístico por título quando o LLM falha. `RAGService.suggested_questions(language)` memoiza o resultado com chave = impressão digital da base, de modo que as quatro operações de escrita (upload, remoção, reset, restore) invalidam o cache sem código explícito. A UI consome isso por um endpoint novo `GET /suggestions` no modo HTTP e pelo serviço direto no modo embedded.

**Tech Stack:** Python 3.11+, pytest, FastAPI + Pydantic v2, Streamlit, Groq (`llama-3.1-8b-instant`).

## Global Constraints

- Spec de referência: `docs/superpowers/specs/2026-08-12-dynamic-question-suggestions-design.md`
- Rodar os testes com `.venv/Scripts/python.exe -m pytest` (Windows; o repo tem venv local)
- Toda a suíte roda offline: nenhum teste novo pode chamar o Groq de verdade — use `tests.fakes.FakeGroq`
- Idiomas suportados: exatamente `"en"` e `"pt"`
- Número de sugestões: 3 (parâmetro `n`, default 3)
- Modelo das sugestões: `llama-3.1-8b-instant`, via a constante `SUGGESTION_MODEL` de `rag/config.py`
- A UI nunca importa `rag` nem chama o LLM: fala só com `app.backend`
- Comentários e nomes de código em inglês, seguindo o resto do repositório

---

### Task 1: Módulo de sugestões no núcleo

**Files:**
- Create: `src/rag/generation/suggestions.py`
- Modify: `src/rag/config.py:12` (acrescentar `SUGGESTION_MODEL` depois de `REWRITE_MODEL`)
- Test: `tests/test_suggestions.py`

**Interfaces:**
- Consumes: `rag.generation.groq_chat.GroqChat.complete(model, messages, max_tokens=, temperature=) -> str` (levanta `rag.errors.GenerationError`); `rag.models.Chunk(chunk_id, doc_id, doc_title, page, position, text)`
- Produces: `suggest_questions(chat: GroqChat, chunks: list[Chunk], language: str = "en", n: int = 3) -> list[str]` — devolve exatamente `n` perguntas, ou `[]` quando `chunks` está vazio. Nunca levanta.

- [ ] **Step 1: Write the failing test**

Criar `tests/test_suggestions.py`:

```python
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat
from rag.generation.suggestions import suggest_questions
from rag.models import Chunk
from tests.fakes import FakeGroq


def _chunks(*docs):
    """Two chunks per document, positions 0 and 1."""
    out = []
    for doc_id, title in docs:
        for position in range(2):
            out.append(Chunk(chunk_id=f"{doc_id}:{position}", doc_id=doc_id,
                             doc_title=title, page=position + 1,
                             position=position, text=f"{title} texto {position}"))
    return out


def _broken_chat(monkeypatch):
    chat = GroqChat(client=FakeGroq([]))
    monkeypatch.setattr(chat, "complete",
                        lambda *a, **k: (_ for _ in ()).throw(GenerationError("down")))
    return chat


def test_returns_questions_from_the_llm():
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?"]))
    assert suggest_questions(chat, _chunks(("d", "Doc D"))) == ["Q1?", "Q2?", "Q3?"]


def test_strips_numbering_bullets_and_quotes():
    chat = GroqChat(client=FakeGroq(['1. "Q1?"\n- Q2?\n3) Q3?\n\n']))
    assert suggest_questions(chat, _chunks(("d", "Doc D"))) == ["Q1?", "Q2?", "Q3?"]


def test_caps_at_n():
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?\nQ4?"]))
    assert suggest_questions(chat, _chunks(("d", "Doc D"))) == ["Q1?", "Q2?", "Q3?"]


def test_empty_base_returns_empty_without_calling_the_llm():
    fake = FakeGroq([])
    assert suggest_questions(GroqChat(client=fake), []) == []
    assert fake.calls == []


def test_generation_error_falls_back_to_titles(monkeypatch):
    out = suggest_questions(_broken_chat(monkeypatch),
                            _chunks(("a", "Paper A"), ("b", "Paper B")))
    assert len(out) == 3
    assert all("Paper A" in q or "Paper B" in q for q in out)


def test_short_output_falls_back():
    chat = GroqChat(client=FakeGroq(["Q1?\n\n"]))
    out = suggest_questions(chat, _chunks(("a", "Paper A")))
    assert len(out) == 3
    assert all("Paper A" in q for q in out)


def test_fallback_covers_n_with_a_single_document(monkeypatch):
    out = suggest_questions(_broken_chat(monkeypatch), _chunks(("a", "Paper A")))
    assert len(out) == len(set(out)) == 3


def test_prompt_samples_first_chunk_of_at_most_five_documents():
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?"]))
    docs = [(f"d{i}", f"Doc {i}") for i in range(7)]
    suggest_questions(chat, _chunks(*docs))
    user = chat._client.calls[0]["messages"][1]["content"]
    assert user.count("---") == 4      # 5 trechos -> 4 separadores
    assert "Doc 5" not in user         # sexto documento fora da amostra
    assert "texto 1" not in user       # só a posição 0 de cada documento


def test_uses_the_suggestion_model():
    from rag.config import SUGGESTION_MODEL
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?"]))
    suggest_questions(chat, _chunks(("d", "Doc D")))
    assert chat._client.calls[0]["model"] == SUGGESTION_MODEL


def test_portuguese_fallback_uses_portuguese_templates(monkeypatch):
    out = suggest_questions(_broken_chat(monkeypatch), _chunks(("a", "Paper A")),
                            language="pt")
    assert out[0] == "O que o Paper A propõe?"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_suggestions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.generation.suggestions'`

- [ ] **Step 3: Add the model constant**

Em `src/rag/config.py`, logo depois da linha `REWRITE_MODEL = "llama-3.1-8b-instant"`:

```python
SUGGESTION_MODEL = "llama-3.1-8b-instant"
```

- [ ] **Step 4: Write the implementation**

Criar `src/rag/generation/suggestions.py`:

```python
import re

from rag.config import SUGGESTION_MODEL
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat
from rag.models import Chunk

_MAX_DOCS = 5
_EXCERPT_CHARS = 600

_SYSTEMS = {
    "en": (
        "You write example questions for a research assistant's knowledge base. "
        "From the excerpts of the indexed documents, write exactly {n} short "
        "questions those documents can answer. Each question must be "
        "self-contained, under 80 characters, and about the content of the "
        "documents. Reply with one question per line and nothing else: no "
        "numbering, no bullets, no quotes, no commentary."
    ),
    "pt": (
        "Você escreve perguntas de exemplo para a base de conhecimento de um "
        "assistente de pesquisa. A partir dos trechos dos documentos indexados, "
        "escreva exatamente {n} perguntas curtas que esses documentos consigam "
        "responder. Cada pergunta deve ser autocontida, ter menos de 80 "
        "caracteres e falar do conteúdo dos documentos. Responda com uma "
        "pergunta por linha e nada mais: sem numeração, sem marcadores, sem "
        "aspas e sem comentários."
    ),
}

_FALLBACKS = {
    "en": ["What does {title} propose?",
           "What are the main findings in {title}?",
           "How does {title} evaluate its approach?"],
    "pt": ["O que o {title} propõe?",
           "Quais são os principais resultados de {title}?",
           "Como {title} avalia a abordagem?"],
}


def _sample(chunks: list[Chunk]) -> list[Chunk]:
    # One excerpt per document — the lowest position is the start of the paper,
    # where the abstract and the introduction live. Sorting by doc_id keeps the
    # prompt identical across calls on an unchanged knowledge base.
    first: dict[str, Chunk] = {}
    for chunk in chunks:
        current = first.get(chunk.doc_id)
        if current is None or chunk.position < current.position:
            first[chunk.doc_id] = chunk
    return [first[doc_id] for doc_id in sorted(first)][:_MAX_DOCS]


def _clean(line: str) -> str:
    line = re.sub(r"^\s*[-*•]\s*", "", line.strip())
    line = re.sub(r"^\d+[.)]\s*", "", line)
    return line.strip().strip('"').strip("'").strip()


def _parse(raw: str, n: int) -> list[str]:
    questions = [_clean(line) for line in raw.splitlines()]
    return [q for q in questions if q][:n]


def _fallback(sample: list[Chunk], language: str, n: int) -> list[str]:
    titles = [c.doc_title for c in sample]
    templates = _FALLBACKS[language]
    return [templates[i % len(templates)].format(title=titles[i % len(titles)])
            for i in range(n)]


def suggest_questions(chat: GroqChat, chunks: list[Chunk],
                      language: str = "en", n: int = 3) -> list[str]:
    sample = _sample(chunks)
    if not sample:
        return []
    excerpts = "\n\n---\n\n".join(
        f"({c.doc_title})\n{c.text[:_EXCERPT_CHARS]}" for c in sample)
    messages = [
        {"role": "system", "content": _SYSTEMS[language].format(n=n)},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}"},
    ]
    try:
        raw = chat.complete(SUGGESTION_MODEL, messages, max_tokens=200,
                            temperature=0.5)
    except GenerationError:
        return _fallback(sample, language, n)
    questions = _parse(raw, n)
    return questions if len(questions) == n else _fallback(sample, language, n)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_suggestions.py -v`
Expected: PASS — 10 testes

- [ ] **Step 6: Commit**

```bash
git add src/rag/generation/suggestions.py src/rag/config.py tests/test_suggestions.py
git commit -m "feat: generate example questions from the indexed documents"
```

---

### Task 2: Cache por versão da base no RAGService

**Files:**
- Modify: `src/rag/service.py` (imports no topo, `__init__` linha 37-45, novo método depois de `documents()` na linha 132)
- Test: `tests/test_service.py` (acrescentar fixture e testes ao final)

**Interfaces:**
- Consumes: `suggest_questions(chat, chunks, language="en", n=3) -> list[str]` da Task 1
- Produces: `RAGService.suggested_questions(language: str = "en") -> list[str]` — memoizado; a chave é a impressão digital da base, então `add_document`, `remove_document`, `reset_documents` e `restore_default_documents` invalidam sozinhos

- [ ] **Step 1: Write the failing test**

Acrescentar ao final de `tests/test_service.py` (os imports `numpy as np`, `pytest`, `Chunk`, `IndexStore`, `RAGService`, `GroqChat`, `FeedbackDB`, `FakeGroq` e as classes `FakeEmbedder`/`FakeReranker` já existem no topo do arquivo):

```python
@pytest.fixture
def suggest_service(tmp_path):
    store = IndexStore(dim=4)
    chunks = [Chunk(chunk_id=f"{doc}:0", doc_id=doc, doc_title=f"Paper {doc.upper()}",
                    page=1, position=0, text=f"texto do {doc}")
              for doc in ("a", "b")]
    store.add(chunks, np.eye(4, dtype="float32")[:2])
    fake = FakeGroq(["Q1?\nQ2?\nQ3?", "R1?\nR2?\nR3?"])
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=GroqChat(client=fake), db=db,
                     index_dir=tmp_path / "index", documents_dir=tmp_path / "docs")
    yield svc, fake
    db.close()


def test_suggested_questions_are_cached_for_an_unchanged_base(suggest_service):
    svc, fake = suggest_service
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]
    assert len(fake.calls) == 1


def test_suggested_questions_regenerate_after_the_base_changes(suggest_service):
    svc, fake = suggest_service
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]
    svc.remove_document("b")
    assert svc.suggested_questions("en") == ["R1?", "R2?", "R3?"]
    assert len(fake.calls) == 2


def test_suggested_questions_are_cached_per_language(suggest_service):
    svc, fake = suggest_service
    svc.suggested_questions("en")
    svc.suggested_questions("pt")
    svc.suggested_questions("en")
    assert len(fake.calls) == 2


def test_suggested_questions_on_an_empty_base(tmp_path):
    fake = FakeGroq([])
    db = FeedbackDB(":memory:")
    svc = RAGService(store=IndexStore(dim=4), embedder=FakeEmbedder(),
                     reranker=FakeReranker(), chat=GroqChat(client=fake), db=db,
                     index_dir=tmp_path / "index", documents_dir=tmp_path / "docs")
    assert svc.suggested_questions("en") == []
    assert fake.calls == []
    db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_service.py -k suggested -v`
Expected: FAIL — `AttributeError: 'RAGService' object has no attribute 'suggested_questions'`

- [ ] **Step 3: Write the implementation**

Em `src/rag/service.py`, acrescentar `import hashlib` no topo (antes de `import time`) e a importação do módulo novo junto das outras de `rag.generation`:

```python
from rag.generation.suggestions import suggest_questions
```

No final do `__init__`, depois de `self.retriever = HybridRetriever(store, embedder, reranker)`:

```python
        # (base fingerprint, {language: questions}) — a new fingerprint drops the
        # whole language map, so the cache only ever holds the current base.
        self._suggestions_cache: tuple[str, dict[str, list[str]]] = ("", {})
```

Depois do método `documents()`, no final da classe:

```python
    def _base_fingerprint(self) -> str:
        counts = Counter(c.doc_id for c in self.store.chunks)
        return hashlib.md5(repr(sorted(counts.items())).encode()).hexdigest()

    def suggested_questions(self, language: str = "en") -> list[str]:
        fingerprint = self._base_fingerprint()
        cached_fingerprint, by_language = self._suggestions_cache
        if cached_fingerprint != fingerprint:
            by_language = {}
            self._suggestions_cache = (fingerprint, by_language)
        if language not in by_language:
            by_language[language] = suggest_questions(self.chat, self.store.chunks,
                                                      language)
        return by_language[language]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_service.py -v`
Expected: PASS — os 4 testes novos e todos os que já existiam

- [ ] **Step 5: Commit**

```bash
git add src/rag/service.py tests/test_service.py
git commit -m "feat: cache suggested questions per knowledge base version"
```

---

### Task 3: Endpoint GET /suggestions

**Files:**
- Modify: `src/api/schemas.py` (nova classe no final)
- Modify: `src/api/main.py:1-8` (imports) e depois da rota `documents()` na linha 107-109
- Test: `tests/test_api.py` (método na `FakeService` e teste novo)

**Interfaces:**
- Consumes: `RAGService.suggested_questions(language: str = "en") -> list[str]` da Task 2
- Produces: `GET /suggestions?language=en` → `{"questions": ["...", "...", "..."]}`; `language` fora de `en|pt` responde 422

- [ ] **Step 1: Write the failing test**

Em `tests/test_api.py`, acrescentar o método à classe `FakeService`, logo depois de `documents()`:

```python
    def suggested_questions(self, language="en"):
        self.last_suggestions_language = language
        return ["Q1?", "Q2?", "Q3?"]
```

E acrescentar os testes ao final do arquivo:

```python
def test_suggestions(client):
    c, service = client
    r = c.get("/suggestions", params={"language": "pt"})
    assert r.status_code == 200
    assert r.json() == {"questions": ["Q1?", "Q2?", "Q3?"]}
    assert service.last_suggestions_language == "pt"


def test_suggestions_defaults_to_english(client):
    c, service = client
    assert c.get("/suggestions").status_code == 200
    assert service.last_suggestions_language == "en"


def test_suggestions_rejects_unknown_language(client):
    c, _ = client
    assert c.get("/suggestions", params={"language": "fr"}).status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -k suggestions -v`
Expected: FAIL — 404, a rota não existe

- [ ] **Step 3: Add the response schema**

No final de `src/api/schemas.py`:

```python
class SuggestionsResponse(BaseModel):
    questions: list[str]
```

- [ ] **Step 4: Add the route**

Em `src/api/main.py`, trocar a linha de import dos schemas por:

```python
from api.schemas import (AskRequest, AskResponse, FeedbackRequest,
                         SuggestionsResponse, UploadResponse)
```

e acrescentar `from typing import Literal` no topo do arquivo (antes de `import threading`).

Depois da rota `documents()`, antes de `health()`:

```python
    # Read-only and served from the service cache, so it stays out of the rate
    # limit — a visitor should not spend an /ask slot on the chat's example pills.
    @app.get("/suggestions", response_model=SuggestionsResponse)
    def suggestions(language: Literal["en", "pt"] = "en"):
        return SuggestionsResponse(questions=svc().suggested_questions(language))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: PASS — os 3 testes novos e todos os que já existiam

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py src/api/schemas.py tests/test_api.py
git commit -m "feat: expose suggested questions over the API"
```

---

### Task 4: Cliente HTTP e backend embedded

**Files:**
- Modify: `src/app/api_client.py` (função nova no final)
- Modify: `src/app/backend.py:21-23` (lista de imports do ramo HTTP) e final do ramo embedded
- Test: `tests/test_backend_embedded.py` (método na `FakeService` e testes novos)

**Interfaces:**
- Consumes: `GET /suggestions` da Task 3; `RAGService.suggested_questions(language)` da Task 2
- Produces: `app.backend.suggestions(language: str = "en") -> list[str]`, disponível nos dois modos, levantando `ApiError`/`ApiConnectionError` como as demais funções do módulo

- [ ] **Step 1: Write the failing test**

Em `tests/test_backend_embedded.py`, acrescentar o método à classe `FakeService`, depois de `documents()`:

```python
    def suggested_questions(self, language="en"):
        self.last_suggestions_language = language
        return ["Q1?", "Q2?", "Q3?"]
```

E os testes ao final do arquivo:

```python
def test_embedded_suggestions(embedded_backend, fake_service):
    assert embedded_backend.suggestions("pt") == ["Q1?", "Q2?", "Q3?"]
    assert fake_service.last_suggestions_language == "pt"


def test_embedded_suggestions_ignore_the_rate_limit(embedded_backend, fake_service,
                                                    monkeypatch):
    monkeypatch.setattr(embedded_backend, "_RATE_LIMIT", 1)
    embedded_backend._hits.clear()
    for _ in range(3):
        assert embedded_backend.suggestions("en") == ["Q1?", "Q2?", "Q3?"]


def test_http_mode_exports_suggestions(monkeypatch):
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    import app.backend as backend
    importlib.reload(backend)
    import app.api_client as api_client
    assert backend.suggestions is api_client.suggestions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend_embedded.py -k suggestions -v`
Expected: FAIL — `AttributeError: module 'app.backend' has no attribute 'suggestions'`

- [ ] **Step 3: Add the HTTP client function**

No final de `src/app/api_client.py`:

```python
def suggestions(language: str = "en") -> list:
    return _request("GET", f"{API_URL}/suggestions",
                    params={"language": language})["questions"]
```

- [ ] **Step 4: Wire both backend branches**

Em `src/app/backend.py`, trocar a lista de imports do ramo HTTP por:

```python
    from app.api_client import (ApiConnectionError, ApiError, ask, documents,
                                metrics, remove_document, reset_documents,
                                restore_defaults, send_feedback, suggestions,
                                upload)
```

E acrescentar, no final do ramo embedded (depois de `documents()`):

```python
    def suggestions(language: str = "en") -> list:
        # Not rate limited: read-only and answered from the service cache.
        return _service_or_api_error().suggested_questions(language)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend_embedded.py -v`
Expected: PASS — os 3 testes novos e todos os que já existiam

- [ ] **Step 6: Commit**

```bash
git add src/app/api_client.py src/app/backend.py tests/test_backend_embedded.py
git commit -m "feat: reach suggested questions from both backend modes"
```

---

### Task 5: Pills dinâmicas no chat

**Files:**
- Modify: `src/app/views/chat.py:3` (import) e `src/app/views/chat.py:70-73` (bloco das pills)
- Modify: `src/app/translations.py:31-36` (remover `example_q1`, `example_q2`, `example_q3`)

**Interfaces:**
- Consumes: `app.backend.suggestions(language) -> list[str]` da Task 4
- Produces: nada — é a ponta da cadeia

- [ ] **Step 1: Update the import**

Em `src/app/views/chat.py`, trocar a linha 3 por:

```python
from app.backend import (ApiConnectionError, ApiError, ask, documents,
                         send_feedback, suggestions)
```

- [ ] **Step 2: Replace the pills block**

Trocar o bloco `else:` das linhas 70-73 por:

```python
    else:
        try:
            examples = suggestions(lang)
        except ApiError:
            examples = []  # decoração: sem sugestões, a página segue normal
        if examples:
            st.pills(t("try_asking", lang), examples,
                     key="example_pills", on_change=_pick_example)
```

(`ApiConnectionError` herda de `ApiError`, então o `except` cobre os dois casos.)

- [ ] **Step 3: Remove the hardcoded strings**

Em `src/app/translations.py`, apagar as seis linhas das chaves `example_q1`, `example_q2` e `example_q3` (en e pt). A chave `try_asking`, logo acima, permanece.

- [ ] **Step 4: Verify no reference is left behind**

Run: `git grep -n "example_q" -- src tests`
Expected: nenhuma saída

- [ ] **Step 5: Run the whole offline suite**

Run: `.venv/Scripts/python.exe -m pytest -q -m "not integration"`
Expected: PASS — suíte inteira verde

- [ ] **Step 6: Manual check in the app**

Com `GROQ_API_KEY` no `.env`, subir a API e a UI:

```bash
.venv/Scripts/python.exe -m uvicorn api.main:create_app --factory --port 8000
.venv/Scripts/python.exe -m streamlit run src/app/Home.py
```

Conferir, na página Chat:
1. As pills refletem os documentos indexados no momento
2. Remover um documento na página Documents e voltar ao Chat muda as pills
3. Alternar EN/PT devolve sugestões no idioma escolhido
4. Com a base vazia, aparece o card "Comece aqui" e nenhuma pill

- [ ] **Step 7: Commit**

```bash
git add src/app/views/chat.py src/app/translations.py
git commit -m "feat: chat pills follow the knowledge base"
```

---

## Notas de implantação

O deploy no Streamlit Community Cloud pode manter o processo Python vivo entre deploys, o que deixa módulos antigos em `sys.modules` — foi a causa do `ImportError` investigado em 2026-08-12. Esta mudança acrescenta nomes novos a módulos existentes (`app.backend.suggestions`, `app.api_client.suggestions`), exatamente o padrão que dispara o problema. Depois do push, use **Manage app → ⋮ → Reboot app** para forçar um processo novo.
