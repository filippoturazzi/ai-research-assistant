# i18n: English Default + EN/PT Selector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tornar inglês o idioma padrão de todo o sistema (README, prompts, mensagens, scripts, UI) e adicionar um seletor EN/PT no Streamlit que troca interface + prompts.

**Architecture:** i18n leve com dicionários Python. Um parâmetro `language` ("en"|"pt", default "en") flui UI → API → `RAGService.ask` → prompts. Strings da UI vêm de `src/app/translations.py`; prompts por idioma em `rag.generation`; mensagens de backend ficam fixas em inglês.

**Tech Stack:** o existente (FastAPI, Streamlit, pytest) — nenhuma dependência nova.

**Spec:** `docs/superpowers/specs/2026-08-09-i18n-english-default-design.md`

## Global Constraints

- Códigos de idioma: `"en"` e `"pt"`; default SEMPRE `"en"` em toda assinatura nova (`language: str = "en"`).
- Resposta do LLM segue o idioma selecionado (system prompt instrui "Answer in English." / "Responda em português.").
- Query reescrita para busca é SEMPRE em inglês (ambos os templates instruem isso — os papers são em inglês).
- Mensagens de backend (`rag.*`, details HTTP, scripts) sempre em inglês; NÃO variam com o seletor.
- `NO_ANSWER` exato por idioma: en `"I could not find this information in the documents."`, pt `"Não encontrei essa informação nos documentos."`.
- Nada muda no banco (sem coluna de idioma).
- Suíte offline continua sem rede; testes rodam com `.venv\Scripts\pytest`.
- Commits em inglês, convenção `feat:`/`fix:`/`test:`/`docs:`/`chore:`.

---

### Task 1: Prompts por idioma (núcleo de geração)

**Files:**
- Modify: `src/rag/generation/prompts.py` (reescrever)
- Modify: `src/rag/generation/generator.py` (reescrever)
- Modify: `src/rag/generation/rewriter.py` (reescrever)
- Test: `tests/test_generation.py` (reescrever), `tests/test_rewriter.py` (ajustar)

**Interfaces:**
- Consumes: `Chunk`, `GroqChat.complete`, `GENERATION_MODEL`, `REWRITE_MODEL`, `GenerationError`
- Produces: `NO_ANSWER: dict[str, str]`; `build_answer_messages(question, chunks, language: str = "en") -> list[dict]`; `generate_answer(chat, question, chunks, language: str = "en") -> str`; `rewrite_query(chat, question, history, language: str = "en") -> str`. Task 3 (service) consome exatamente essas assinaturas.

- [ ] **Step 1: Reescrever os testes (failing)** — `tests/test_generation.py` completo:

```python
from rag.generation.generator import generate_answer
from rag.generation.groq_chat import GroqChat
from rag.generation.prompts import NO_ANSWER, build_answer_messages, build_context
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


def test_messages_default_english():
    messages = build_answer_messages("What is attention?", [_chunk(0, "T", 1, "txt")])
    assert messages[0]["role"] == "system"
    assert "Answer in English." in messages[0]["content"]
    assert NO_ANSWER["en"] in messages[0]["content"]
    assert "What is attention?" in messages[1]["content"]


def test_messages_portuguese():
    messages = build_answer_messages("O que é atenção?", [_chunk(0, "T", 1, "txt")], language="pt")
    assert "Responda em português." in messages[0]["content"]
    assert NO_ANSWER["pt"] in messages[0]["content"]


def test_generate_answer_passes_language():
    fake = FakeGroq(["A atenção é... [1]"])
    out = generate_answer(GroqChat(client=fake), "O que é atenção?",
                          [_chunk(0, "T", 1, "txt")], language="pt")
    assert out == "A atenção é... [1]"
    assert "Responda em português." in fake.calls[0]["messages"][0]["content"]
```

E em `tests/test_rewriter.py`, adicionar ao final (mantendo os 3 testes existentes intactos):

```python
def test_rewriter_portuguese_template_still_targets_english_query():
    from rag.generation import rewriter
    assert "search query in English" in rewriter._SYSTEMS["en"]
    assert "consulta de busca em inglês" in rewriter._SYSTEMS["pt"]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv\Scripts\pytest tests/test_generation.py tests/test_rewriter.py -v`
Expected: FAIL (ImportError: `NO_ANSWER` agora é dict / `_SYSTEMS` não existe)

- [ ] **Step 3: Reescrever `src/rag/generation/prompts.py`**

```python
from rag.models import Chunk

NO_ANSWER = {
    "en": "I could not find this information in the documents.",
    "pt": "Não encontrei essa informação nos documentos.",
}

_SYSTEMS = {
    "en": f"""You are a research assistant that answers based EXCLUSIVELY on the \
provided context (numbered document excerpts).

Rules:
1. Use only information present in the context. Do not use outside knowledge.
2. Cite the source of every claim with its number in brackets, e.g. [1], [2].
3. If the context does not contain the answer, say exactly: "{NO_ANSWER['en']}"
4. Answer in English.""",
    "pt": f"""Você é um assistente de pesquisa que responde com base EXCLUSIVAMENTE \
no contexto fornecido (trechos de documentos numerados).

Regras:
1. Use apenas informações presentes no contexto. Não use conhecimento externo.
2. Cite a fonte de cada afirmação com o número entre colchetes, ex.: [1], [2].
3. Se o contexto não contém a resposta, diga exatamente: "{NO_ANSWER['pt']}"
4. Responda em português.""",
}


def build_context(chunks: list[Chunk]) -> str:
    blocks = [
        f"[{i}] ({c.doc_title}, p. {c.page})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def build_answer_messages(question: str, chunks: list[Chunk],
                          language: str = "en") -> list[dict]:
    user = f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _SYSTEMS[language]},
        {"role": "user", "content": user},
    ]
```

- [ ] **Step 4: Reescrever `src/rag/generation/generator.py`**

```python
from rag.config import GENERATION_MODEL
from rag.generation.groq_chat import GroqChat
from rag.generation.prompts import build_answer_messages
from rag.models import Chunk


def generate_answer(chat: GroqChat, question: str, chunks: list[Chunk],
                    language: str = "en") -> str:
    return chat.complete(GENERATION_MODEL, build_answer_messages(question, chunks, language))
```

- [ ] **Step 5: Reescrever `src/rag/generation/rewriter.py`**

```python
from rag.config import REWRITE_MODEL
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat

_SYSTEMS = {
    "en": (
        "You rewrite the user's last question as a self-contained search query, "
        "resolving references to the conversation history and expanding acronyms "
        "when useful. The documents are in English, so write the search query in "
        "English. Reply ONLY with the rewritten query, without quotes or explanations."
    ),
    "pt": (
        "Você reescreve a última pergunta do usuário como uma consulta de busca "
        "autocontida, resolvendo referências ao histórico da conversa e expandindo "
        "siglas quando útil. Os documentos são em inglês, então escreva a consulta "
        "de busca em inglês. Responda APENAS com a consulta reescrita, sem aspas "
        "e sem explicações."
    ),
}


def rewrite_query(chat: GroqChat, question: str, history: list[dict],
                  language: str = "en") -> str:
    messages = (
        [{"role": "system", "content": _SYSTEMS[language]}]
        + history[-6:]
        + [{"role": "user", "content": f"Question: {question}\nSearch query:"}]
    )
    try:
        out = chat.complete(REWRITE_MODEL, messages, max_tokens=100)
    except GenerationError:
        return question
    out = out.strip().strip('"').strip()
    return out or question
```

- [ ] **Step 6: Rodar e ver passar** — `.venv\Scripts\pytest tests/test_generation.py tests/test_rewriter.py -v` → 5 + 4 PASS. Depois a suíte: `.venv\Scripts\pytest -m "not integration"` — **atenção:** `tests/test_service.py` pode quebrar se referenciar as assinaturas antigas; ele NÃO deve quebrar (chama via service, que ainda usa default "en" implícito — service só muda na Task 3). Se algo quebrar por import de `NO_ANSWER` string, ajustar o teste afetado para `NO_ANSWER["en"]`.

- [ ] **Step 7: Commit**

```bash
git add src/rag/generation tests/test_generation.py tests/test_rewriter.py
git commit -m "feat: per-language prompts with English default"
```

---

### Task 2: Mensagens de backend em inglês

**Files:**
- Modify: `src/rag/errors.py`, `src/rag/ingestion/pdf_extractor.py`, `src/rag/ingestion/pipeline.py`, `src/rag/retrieval/store.py`, `src/rag/generation/groq_chat.py`, `src/rag/service.py`
- Test: suíte existente (asserts que casem mensagens, se houver)

**Interfaces:**
- Produces: mensagens novas exatas (Task 5 e o README dependem delas serem estáveis):
  - pdf_extractor: `f"Could not read the PDF '{path.name}': {exc}"` e `f"No extractable text in '{path.name}'."`
  - pipeline: `f"Document '{doc_id}' is already indexed."`
  - store (load ausente): `f"Index not found in '{dir}'. Run: python scripts/build_index.py"`
  - store (add mismatch): manter `f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length"` (já em inglês)
  - groq_chat (retry esgotado): `f"LLM unavailable: {exc}"`
  - groq_chat (sem key): `"GROQ_API_KEY is not set — copy .env.example to .env and add your key."`
  - service (nome inválido): `"Invalid file name."`
  - service (duplicado): `f"Document '{stem}' is already indexed."` (mesmo texto do pipeline)

- [ ] **Step 1: Traduzir as mensagens** nos 6 arquivos acima para os textos exatos listados. Traduzir também os docstrings de `errors.py` para inglês (ex.: `"""PDF without extractable text, or corrupted."""`, `"""LLM call failed after exhausting retries."""`, `"""Index missing on disk — run scripts/build_index.py."""`, `"""A document with the same doc_id is already indexed."""`).

- [ ] **Step 2: Varredura de sobras** — garantir que não restou string PT no núcleo/API:

Run: `grep -rnP "[ãáâàéêíóôõúç]" src/rag src/api || echo CLEAN`
Expected: as únicas ocorrências permitidas são em `src/rag/generation/prompts.py` e `src/rag/generation/rewriter.py` (templates `"pt"`). Qualquer outra é sobra — traduzir.

- [ ] **Step 3: Rodar a suíte** — `.venv\Scripts\pytest -m "not integration" -v`. Se algum teste casar mensagem antiga em PT (ex.: `match=` em algum `pytest.raises`), atualizar o assert para a mensagem nova em inglês.
Expected: tudo PASS.

- [ ] **Step 4: Commit**

```bash
git add src/rag src/api tests
git commit -m "chore: translate backend messages and docstrings to English"
```

---

### Task 3: `language` no service e na API

**Files:**
- Modify: `src/rag/service.py` (método `ask`)
- Modify: `src/api/schemas.py` (AskRequest), `src/api/main.py` (endpoint /ask)
- Test: `tests/test_service.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `rewrite_query(..., language)` e `generate_answer(..., language)` da Task 1
- Produces: `RAGService.ask(question: str, history: list[dict] | None = None, language: str = "en") -> AskResult`; `AskRequest.language: Literal["en", "pt"] = "en"`. Task 4 (UI) envia `language` no JSON de `/ask`.

- [ ] **Step 1: Testes que falham** — adicionar a `tests/test_service.py`:

```python
def test_ask_portuguese_uses_pt_prompt(service):
    result = service.ask("qual é o chunk?", language="pt")
    assert result.answer == "resposta final [1]"
    # segunda chamada ao Groq é a geração; system prompt deve ser o PT
    generation_call = service.chat._client.calls[1]
    assert "Responda em português." in generation_call["messages"][0]["content"]
```

(Obs.: o fixture `service` guarda `chat=GroqChat(client=FakeGroq([...]))`; exponha o fake via `service.chat._client` — já acessível, `FakeGroq.calls` existe.)

E a `tests/test_api.py` (no FakeService, mudar `ask` para registrar o idioma):

```python
# no FakeService:
    def ask(self, question, history=None, language="en"):
        self.last_language = language
        if self.fail_generation:
            raise GenerationError("down")
        return AskResult(interaction_id=1, answer="resp [1]", rewritten_query="rw",
                         sources=[Source(doc_title="T", page=2, text="x", score=0.5)])

# novos testes:
def test_ask_passes_language(client):
    c, service = client
    c.post("/ask", json={"question": "q?", "language": "pt"})
    assert service.last_language == "pt"


def test_ask_defaults_to_english(client):
    c, service = client
    c.post("/ask", json={"question": "q?"})
    assert service.last_language == "en"


def test_ask_rejects_unknown_language(client):
    c, _ = client
    assert c.post("/ask", json={"question": "q?", "language": "fr"}).status_code == 422
```

- [ ] **Step 2: Rodar e ver falhar** — `.venv\Scripts\pytest tests/test_service.py tests/test_api.py -v`
Expected: FAIL (TypeError: ask() got an unexpected keyword argument 'language')

- [ ] **Step 3: Implementar** — em `src/rag/service.py`, assinatura e repasse:

```python
    def ask(self, question: str, history: list[dict] | None = None,
            language: str = "en") -> AskResult:
        start = time.perf_counter()
        rewritten = rewrite_query(self.chat, question, history or [], language)
        retrieved = self.retriever.retrieve(rewritten)
        answer = generate_answer(self.chat, question,
                                 [r.chunk for r in retrieved], language)
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
```

Em `src/api/schemas.py`, no `AskRequest`:

```python
class AskRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []
    language: Literal["en", "pt"] = "en"
```

Em `src/api/main.py`, no endpoint `/ask`:

```python
            result = svc().ask(body.question,
                               [m.model_dump() for m in body.history],
                               language=body.language)
```

- [ ] **Step 4: Rodar e ver passar** — `.venv\Scripts\pytest tests/test_service.py tests/test_api.py -v`, depois a suíte offline inteira.
Expected: tudo PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag/service.py src/api tests/test_service.py tests/test_api.py
git commit -m "feat: language parameter flows from API to prompts"
```

---

### Task 4: UI — seletor de idioma e traduções

**Files:**
- Create: `src/app/translations.py`
- Modify: `src/app/api_client.py`, `src/app/Home.py`, `src/app/pages/1_Documentos.py`, `src/app/pages/2_Metricas.py`

**Interfaces:**
- Consumes: `/ask` com `language` (Task 3)
- Produces: `t(key: str, language: str) -> str`; `language_selector() -> str` (renderiza selectbox no sidebar e retorna "en"|"pt"); `ApiConnectionError(ApiError)` em `api_client`; `ask(question, history, language)` no api_client. Sem testes automatizados (UI): verificação = py_compile + smoke HTTP 200.

- [ ] **Step 1: Criar `src/app/translations.py`**

```python
import streamlit as st

TRANSLATIONS = {
    "language_label": {"en": "Language", "pt": "Idioma"},
    "page_title": {"en": "AI Research Assistant", "pt": "AI Research Assistant"},
    "tagline": {
        "en": "Ask about the indexed papers — answers with [n] citations.",
        "pt": "Pergunte sobre os papers indexados — respostas com citações [n].",
    },
    "chat_placeholder": {
        "en": "Ask a question about the documents...",
        "pt": "Faça uma pergunta sobre os documentos...",
    },
    "searching": {"en": "Searching the documents...", "pt": "Buscando nos documentos..."},
    "sources": {"en": "📄 Sources ({n})", "pt": "📄 Fontes ({n})"},
    "page_abbrev": {"en": "p.", "pt": "p."},
    "score": {"en": "score", "pt": "score"},
    "feedback_thanks": {"en": "Thanks for the feedback!", "pt": "Obrigado pelo feedback!"},
    "ask_failed": {
        "en": "I couldn't answer right now:",
        "pt": "Não consegui responder agora:",
    },
    "api_unreachable": {
        "en": "Could not reach the API — is it running?",
        "pt": "Não consegui falar com a API — ela está rodando?",
    },
    "docs_title": {"en": "📄 Documents", "pt": "📄 Documentos"},
    "upload_label": {"en": "Add a PDF to the collection", "pt": "Adicionar PDF à coleção"},
    "index_button": {"en": "Index document", "pt": "Indexar documento"},
    "indexing": {
        "en": "Extracting, chunking and indexing...",
        "pt": "Extraindo, chunkeando e indexando...",
    },
    "indexed_ok": {
        "en": "'{doc}' indexed: {n} chunks.",
        "pt": "'{doc}' indexado: {n} chunks.",
    },
    "collection": {"en": "Current collection", "pt": "Coleção atual"},
    "chunks": {"en": "chunks", "pt": "chunks"},
    "metrics_title": {"en": "📊 Metrics", "pt": "📊 Métricas"},
    "m_questions": {"en": "Questions", "pt": "Perguntas"},
    "m_approval": {"en": "Approval", "pt": "Aprovação"},
    "m_approval_7d": {"en": "Approval (7d)", "pt": "Aprovação (7d)"},
    "m_latency": {"en": "Avg latency", "pt": "Latência média"},
    "negatives_title": {
        "en": "👎 questions (investigation queue)",
        "pt": "Perguntas com 👎 (fila de investigação)",
    },
    "no_negatives": {"en": "No negative feedback. 🎉", "pt": "Nenhum feedback negativo. 🎉"},
    "top_docs": {"en": "Most cited documents", "pt": "Documentos mais citados"},
    "citations": {"en": "citations", "pt": "citações"},
}

_LANGUAGE_NAMES = {"en": "English", "pt": "Português"}


def t(key: str, language: str) -> str:
    return TRANSLATIONS[key][language]


def language_selector() -> str:
    if "language" not in st.session_state:
        st.session_state.language = "en"
    st.sidebar.selectbox(
        t("language_label", st.session_state.language),
        options=["en", "pt"],
        format_func=lambda code: _LANGUAGE_NAMES[code],
        key="language",
    )
    return st.session_state.language
```

- [ ] **Step 2: `src/app/api_client.py`** — três mudanças: (a) nova exceção; (b) `_request` levanta `ApiConnectionError` (mensagem em inglês) em `RequestException`; (c) `ask` ganha `language`:

```python
class ApiError(Exception):
    pass


class ApiConnectionError(ApiError):
    pass
```

No `_request`, trocar o `raise ApiError(...)` do bloco `except requests.exceptions.RequestException` por:

```python
        raise ApiConnectionError(
            f"Could not reach the API — is it running? ({exc.__class__.__name__})"
        ) from exc
```

E:

```python
def ask(question: str, history: list[dict], language: str = "en") -> dict:
    return _handle(_request("post", f"{API_URL}/ask",
                            json={"question": question, "history": history,
                                  "language": language}))
```

- [ ] **Step 3: Reescrever `src/app/Home.py`**

```python
import streamlit as st

from app.api_client import ApiConnectionError, ApiError, ask, send_feedback
from app.translations import language_selector, t

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
lang = language_selector()
st.title("📚 " + t("page_title", lang))
st.caption(t("tagline", lang))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "voted" not in st.session_state:
    st.session_state.voted = set()


def _render_sources(sources):
    with st.expander(t("sources", lang).format(n=len(sources))):
        for i, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {s['doc_title']}** — {t('page_abbrev', lang)} {s['page']} "
                f"({t('score', lang)} {s['score']:.2f})"
            )
            st.text(s["text"][:500])


def _render_feedback(interaction_id):
    if interaction_id in st.session_state.voted:
        st.caption(t("feedback_thanks", lang))
        return
    col_up, col_down, _ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"up-{interaction_id}"):
        try:
            send_feedback(interaction_id, 1)
        except ApiConnectionError:
            st.error(t("api_unreachable", lang))
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state.voted.add(interaction_id)
            st.rerun()
    if col_down.button("👎", key=f"down-{interaction_id}"):
        try:
            send_feedback(interaction_id, -1)
        except ApiConnectionError:
            st.error(t("api_unreachable", lang))
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state.voted.add(interaction_id)
            st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])
        if message.get("interaction_id"):
            _render_feedback(message["interaction_id"])

if question := st.chat_input(t("chat_placeholder", lang)):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]][-6:]
        try:
            with st.spinner(t("searching", lang)):
                result = ask(question, history, lang)
        except ApiConnectionError:
            st.error(t("api_unreachable", lang))
        except ApiError as exc:
            st.error(f"{t('ask_failed', lang)} {exc}")
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

- [ ] **Step 4: Reescrever `src/app/pages/1_Documentos.py`**

```python
import streamlit as st

from app.api_client import ApiConnectionError, ApiError, documents, upload
from app.translations import language_selector, t

lang = language_selector()
st.title(t("docs_title", lang))

uploaded = st.file_uploader(t("upload_label", lang), type=["pdf"])
if uploaded is not None and st.button(t("index_button", lang)):
    try:
        with st.spinner(t("indexing", lang)):
            result = upload(uploaded.name, uploaded.getvalue())
        st.success(t("indexed_ok", lang).format(doc=result["doc_id"],
                                                n=result["chunks_added"]))
    except ApiConnectionError:
        st.error(t("api_unreachable", lang))
    except ApiError as exc:
        st.error(str(exc))

st.divider()
st.subheader(t("collection", lang))
try:
    for doc in documents():
        st.markdown(f"- **{doc['doc_title']}** — {doc['chunks']} {t('chunks', lang)}")
except ApiConnectionError:
    st.error(t("api_unreachable", lang))
except ApiError as exc:
    st.error(str(exc))
```

- [ ] **Step 5: Reescrever `src/app/pages/2_Metricas.py`**

```python
import streamlit as st

from app.api_client import ApiConnectionError, ApiError, metrics
from app.translations import language_selector, t

lang = language_selector()
st.title(t("metrics_title", lang))

try:
    data = metrics()
except ApiConnectionError:
    st.error(t("api_unreachable", lang))
    st.stop()
except ApiError as exc:
    st.error(str(exc))
    st.stop()


def _pct(value):
    return f"{value * 100:.0f}%" if value is not None else "—"


col1, col2, col3, col4 = st.columns(4)
col1.metric(t("m_questions", lang), data["total_questions"])
col2.metric(t("m_approval", lang), _pct(data["approval_rate"]))
col3.metric(t("m_approval_7d", lang), _pct(data["approval_rate_7d"]))
col4.metric(t("m_latency", lang),
            f"{data['avg_latency_ms']:.0f} ms" if data["avg_latency_ms"] else "—")

st.subheader(t("negatives_title", lang))
if not data["negatives"]:
    st.caption(t("no_negatives", lang))
for item in data["negatives"]:
    with st.expander(f"{item['created_at']} — {item['query']}"):
        st.markdown(item["answer"])
        st.json(item["sources"])

st.subheader(t("top_docs", lang))
for doc in data["top_documents"]:
    st.markdown(f"- **{doc['doc_title']}** — {doc['citations']} {t('citations', lang)}")
```

- [ ] **Step 6: Verificação** — `py_compile` nos 5 arquivos; smoke: `streamlit run src/app/Home.py --server.headless true --server.port 8501` em background → GET http://localhost:8501 retorna 200 → matar o processo. Checklist interativo (trocar idioma e ver UI/resposta mudarem) fica pendente para o usuário.

- [ ] **Step 7: Commit**

```bash
git add src/app
git commit -m "feat: EN/PT language selector with translated UI strings"
```

---

### Task 5: Scripts e README em inglês + verificação final

**Files:**
- Modify: `scripts/download_papers.py` (docstring + prints), `scripts/build_index.py` (docstring + prints)
- Modify: `README.md` (reescrever em inglês)

**Interfaces:**
- Consumes: mensagens/fluxos das tasks anteriores (o README documenta o comportamento final, incluindo o seletor de idioma)

- [ ] **Step 1: Traduzir prints/docstrings dos scripts** — `download_papers.py`: docstring `"""Downloads the classic AI papers (arXiv) used as the default collection."""`; prints `f"[skip] {name} (already exists)"`, `f"[downloading] {name} ..."`, `"Done."`. `build_index.py`: docstring `"""Builds the index (FAISS + chunks.json) from data/documents/*.pdf."""`; mensagens `f"No PDFs in '{DOCUMENTS_DIR}'. Run first: python scripts/download_papers.py"`, `"Loading embedding model..."`, `f"[ok] {pdf.name}: {added} chunks"`, `f"[error] {pdf.name}: {exc}"`, `f"Index saved to '{INDEX_DIR}' ({len(store.chunks)} chunks)."`. Remover o import morto `from pathlib import Path` de `build_index.py` (sobra apontada em review anterior).

- [ ] **Step 2: Reescrever `README.md`**

````markdown
# 📚 AI Research Assistant — Advanced RAG

A research assistant over classic AI papers, built **from scratch** (no LangChain)
to demonstrate advanced RAG techniques:

- **Hybrid search** — FAISS (semantic) + BM25 (lexical) merged with **Reciprocal Rank Fusion**
- **Re-ranking** with a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **History-aware query rewriting** (Groq, Llama 3.1 8B)
- **Grounded answers with `[n]` citations** (Groq, Llama 3.3 70B)
- **👍/👎 feedback** persisted in SQLite + a metrics dashboard
- Incremental PDF upload
- **Bilingual UI** — English by default, with a Português option that switches
  the interface and the assistant's answers

## Architecture

Streamlit (UI) → FastAPI (API) → RAG core (pure Python library, independently testable).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
copy .env.example .env                            # add your GROQ_API_KEY

python scripts/download_papers.py                 # downloads 5 classic arXiv papers
python scripts/build_index.py                     # builds FAISS + BM25
```

## Running

```bash
uvicorn "api.main:create_app" --factory           # API at http://localhost:8000 (docs at /docs)
streamlit run src/app/Home.py                     # UI at http://localhost:8501
```

## Tests

```bash
pytest -m "not integration"    # offline suite (LLM always mocked)
pytest -m integration          # real retrieval (downloads models on first run)
```

## How a question flows

1. The question + chat history are rewritten into a self-contained search query
   (always in English — the corpus is English).
2. The query runs through FAISS (top-20) and BM25 (top-20); RRF fuses the rankings.
3. A cross-encoder re-ranks the candidates; the top-5 chunks survive.
4. The LLM answers **only** from those chunks, citing `[n]` (document + page),
   in the language selected in the UI.
5. The interaction is logged; user feedback feeds the dashboard.

## Notes

- BM25 uses the **BM25L** variant: classic BM25Okapi zeroes the IDF of terms that
  appear in half or more of a small corpus, silently dropping valid lexical
  matches. BM25L smooths the IDF while keeping non-matching documents at score 0.
````

- [ ] **Step 3: Varredura final de PT fora dos lugares permitidos**

Run: `grep -rnP "[ãáâàéêíóôõúç]" src scripts README.md || echo CLEAN`
Expected: ocorrências apenas em `src/rag/generation/prompts.py`, `src/rag/generation/rewriter.py` (templates "pt") e `src/app/translations.py` (valores "pt"). Nada em README, scripts, api, ou outros módulos.

- [ ] **Step 4: Suíte completa** — `.venv\Scripts\pytest -m "not integration" -v`
Expected: tudo PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts README.md
git commit -m "docs: English README and script messages; note language selector"
```

---

## Cobertura do spec → tasks

| Requisito do spec | Task |
|---|---|
| Prompts por idioma + NO_ANSWER dict + resposta no idioma selecionado | 1 |
| Query reescrita sempre em inglês | 1 |
| Mensagens de backend em inglês (fixas) | 2 |
| `language` em AskRequest/service.ask (default "en") | 3 |
| Seletor EN/PT + translations.py + t() | 4 |
| ApiConnectionError traduzível na UI | 4 |
| Scripts em inglês | 5 |
| README em inglês (+ seletor documentado) | 5 |
| Testes PT/EN + suíte offline verde | 1, 3, 5 |
