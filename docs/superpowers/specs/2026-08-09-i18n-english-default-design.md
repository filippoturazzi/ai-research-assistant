# i18n: English Default + EN/PT Selector — Design

**Data:** 2026-08-09
**Status:** Aprovado em conversa; aguardando revisão final do spec
**Contexto:** O AI Research Assistant (spec `2026-08-09-ai-research-assistant-design.md`) foi construído com strings em português. Este spec muda o padrão do sistema para inglês e adiciona um seletor de idioma EN/PT na UI.

## Requisitos

1. **Inglês é o idioma padrão de todo o sistema:** README, prompts, mensagens de erro do backend, prints dos scripts, strings da UI.
2. **Seletor de idioma no Streamlit** (sidebar, default **English**): opções English / Português. Trocar o idioma afeta **somente**:
   - As strings da interface (títulos, botões, placeholders, mensagens amigáveis)
   - Os prompts enviados ao LLM (system prompts de reescrita e de resposta, e a string `NO_ANSWER`)
3. **A resposta segue o idioma selecionado** — não o idioma da pergunta. PT selecionado → responde em português; EN → inglês.
4. **Mensagens de backend ficam sempre em inglês** (erros de `rag.*`, details HTTP da API, logs de scripts) — não mudam com o seletor.
5. Nada muda no banco (interactions/feedback não registram idioma — YAGNI).

## Abordagem

i18n leve com dicionários Python — sem gettext/Babel (2 idiomas, ~30 strings).

## Mudanças por componente

### Núcleo (`src/rag/`)
- `generation/prompts.py`:
  - `NO_ANSWER: dict[str, str]` — `{"en": "I could not find this information in the documents.", "pt": "Não encontrei essa informação nos documentos."}`
  - `SYSTEM_PROMPTS: dict[str, str]` — versões EN e PT do system prompt de resposta. Cada versão instrui: responder APENAS com base no contexto; citar `[n]`; usar exatamente o `NO_ANSWER` do idioma; **responder no idioma selecionado** (en → "Answer in English.", pt → "Responda em português.").
  - `build_answer_messages(question, chunks, language: str = "en")`.
  - `build_context` inalterado (formato `[n] (título, p. X)` é neutro).
- `generation/generator.py`: `generate_answer(chat, question, chunks, language: str = "en")`.
- `generation/rewriter.py`: system prompt de reescrita por idioma (`_SYSTEMS: dict`), `rewrite_query(chat, question, history, language: str = "en")`. A query reescrita serve para busca — instruir a reescrever no idioma da pergunta/documentos (os papers são em inglês; a reescrita em inglês melhora o retrieval lexical, então ambos os templates instruem: "write the search query in English").
- Mensagens de erro → inglês: `errors.py` (docstrings), `pdf_extractor.py`, `pipeline.py` ("Document 'x' is already indexed."), `store.py` ("Index not found... Run: python scripts/build_index.py"), `groq_chat.py` ("LLM unavailable...", "GROQ_API_KEY is not set — copy .env.example to .env and add your key."), `service.py` ("Invalid file name.").

### API (`src/api/`)
- `schemas.py`: `AskRequest.language: Literal["en", "pt"] = "en"`.
- `main.py`: repassa `language` para `service.ask(...)`.
- `service.py`: `ask(question, history=None, language="en")` → repassa a `rewrite_query` e `generate_answer`.

### UI (`src/app/`)
- Novo `src/app/translations.py`: `TRANSLATIONS: dict[str, dict[str, str]]` (chaves de string → {"en": ..., "pt": ...}) e `t(key: str, language: str) -> str`.
- Todas as páginas: `st.sidebar.selectbox` de idioma (English default), valor em `st.session_state.language` (inicializado "en"); todas as strings visíveis via `t()`.
- `Home.py`: envia `language` no payload de `/ask` (`api_client.ask(question, history, language)`).
- `api_client.py`:
  - `ask(question, history, language)` inclui `language` no JSON.
  - Nova exceção `ApiConnectionError(ApiError)` para falhas de transporte (conexão/timeout), com mensagem em inglês.
  - Nas páginas: `except ApiConnectionError` → exibir `t("api_unreachable", lang)` (traduzida); `except ApiError` → exibir `str(exc)` como veio (detail da API, inglês).

### Scripts e docs
- `scripts/download_papers.py`, `scripts/build_index.py`: prints em inglês.
- `README.md`: reescrito em inglês (mesmo conteúdo/estrutura, + menção ao seletor EN/PT).

### Testes
- Atualizar asserts existentes para strings em inglês (prompts, NO_ANSWER, erros).
- Novos: `build_answer_messages` com `language="pt"` (system prompt PT + NO_ANSWER PT), `rewrite_query` com PT, API `/ask` repassa `language` ao service (fake registra), default "en" quando omitido.
- Suíte continua offline; teste de integração inalterado.

## Fora de escopo
- Persistir idioma no banco; detecção automática de idioma; outros idiomas além de EN/PT; tradução dos PDFs.

## Critérios de sucesso
1. Repo clonado → tudo que o usuário vê por padrão está em inglês (README, UI, respostas).
2. Selecionar "Português" na UI → interface em PT e respostas em PT (incluindo "Não encontrei..." para fora de escopo).
3. Voltar para English → tudo em inglês de novo.
4. Suíte offline verde sem rede.
