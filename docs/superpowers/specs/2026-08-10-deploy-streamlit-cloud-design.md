# Deploy: Streamlit Community Cloud (modo embutido) — Design

**Data:** 2026-08-10
**Status:** Aprovado em conversa
**Contexto:** O plano original (HF Spaces Docker) foi bloqueado por mudança de política do Hugging Face (Docker/Gradio agora exigem PRO; 402 na criação). Pivô aprovado: hospedar a demo no **Streamlit Community Cloud** (gratuito, oficial), com a UI rodando o núcleo RAG **em processo** ("modo embutido"). O repo GitHub público já existe (`filippoturazzi/ai-research-assistant`).

## Decisões

| Decisão | Escolha |
|---|---|
| Hospedagem | Streamlit Community Cloud (share.streamlit.io), deploy do repo GitHub, entrypoint `src/app/Home.py` |
| Modo da demo | Embutido: UI chama `RAGService` direto (sem processo FastAPI). Localmente, nada muda (HTTP continua o default) |
| Chaveamento | Env var `BACKEND_MODE=embedded` (setada só no Community Cloud); default `http` |
| Índice | Commitado no repo (`data/index/index.faiss` + `chunks.json`, ~1MB) — startup rápido, sem rebuild no boot |
| Torch | `torch==2.13.0+cpu` via extra-index no `requirements.txt` (CUDA estouraria o ambiente) |
| Segredo | `GROQ_API_KEY` nos Secrets do app (st.secrets), com ponte para `os.environ` no modo embutido |
| FastAPI | Continua no repo (vitrine + uso local + Docker); demo hospedada documentada como "embedded mode" no README |
| HF | Workflow `deploy-to-hf.yml` removido; front-matter YAML do Space removido do README; Dockerfile PERMANECE (uso local/futuro) |
| Custo | R$ 0 |

## Componentes

### `src/app/backend.py` (novo)
Módulo com a mesma interface que as páginas consomem hoje do `api_client`: `ask(question, history, language)`, `upload(filename, data)`, `send_feedback(interaction_id, rating)`, `metrics()`, `documents()`, e as exceções `ApiError`/`ApiConnectionError` (reexportadas).

- `BACKEND_MODE != "embedded"` (default): reexporta tudo de `app.api_client` — comportamento atual intacto.
- `BACKEND_MODE == "embedded"`:
  - Constrói o `RAGService` real uma única vez via `st.cache_resource` (carrega índice do `data/index`, modelos, SQLite em `data/feedback.db`)
  - Ponte de segredo: se `GROQ_API_KEY` estiver em `st.secrets`, copia para `os.environ` antes de construir o serviço
  - Rate limit global em memória: 10 chamadas/min somando `ask`+`upload` (mesma semântica do limite da API, que não roda neste modo); excedeu → `ApiError("Rate limit exceeded — try again in a minute.")`
  - Pergunta >500 chars → `ApiError` (espelha o limite da API; a UI já tem `max_chars=500`)
  - Mapeamento de exceções do núcleo → `ApiError(str(exc))` (`GenerationError`, `DuplicateDocumentError`, `ExtractionError`, `ValueError`) — a UI já exibe `ApiError` amigavelmente
  - Retornos em dicts com o MESMO shape do JSON da API (`interaction_id`, `answer`, `rewritten_query`, `sources[]`; `doc_id`, `chunks_added`; etc.)

### Páginas
`Home.py`, `pages/1_Documents.py`, `pages/2_Metrics.py`: trocar `from app.api_client import ...` por `from app.backend import ...`. Nenhuma outra mudança.

### `requirements.txt` (novo, raiz)
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.13.0+cpu
-e .
```

### `.gitignore`
Remover a linha `data/index/` (índice passa a ser versionado). `data/documents/*.pdf` e `data/feedback.db` continuam ignorados.

### README
- Live Demo aponta para a URL do Community Cloud (placeholder até a Task interativa)
- Remover o front-matter YAML do HF Space
- Nota: demo hospedada roda em embedded mode (a FastAPI roda local/Docker); app pode "dormir" após inatividade e demora ~1-2 min no primeiro acesso

### Testes
- `tests/test_backend_embedded.py`: com `BACKEND_MODE=embedded` (monkeypatch + reload do módulo) e `_build_service` mockado: shape dos dicts de `ask`; rate limit dispara `ApiError` na N+1ª chamada; `GenerationError` vira `ApiError`; pergunta longa vira `ApiError`.
- Suíte existente intacta (modo http é o default).

## Passos manuais do Filip (guiados)
1. Login em share.streamlit.io com a conta GitHub; autorizar acesso ao repo
2. Create app → repo `filippoturazzi/ai-research-assistant`, branch `master`, main file `src/app/Home.py`
3. Advanced settings → Secrets: `GROQ_API_KEY = "..."`; env var `BACKEND_MODE = "embedded"` (via secrets TOML também vira env? Não — definir no próprio secrets como `BACKEND_MODE = "embedded"` e a ponte do backend lê de `st.secrets` além do env)
4. Aguardar build; testar; atualizar a URL real no README

**Nota técnica do item 3:** o Community Cloud não tem campo separado de env vars; o `backend.py` deve checar `BACKEND_MODE` em `os.environ` E em `st.secrets` (secrets têm precedência de configuração da plataforma).

## Fora de escopo
- Persistência entre restarts (feedback/uploads são efêmeros — como no design anterior)
- Autenticação, domínios custom

## Critérios de sucesso
1. URL pública `*.streamlit.app` abre a UI; pergunta responde com citações; seletor PT funciona
2. Zero setup para o visitante
3. Push na `master` redeploya automaticamente (comportamento nativo do Community Cloud)
4. 11ª chamada no mesmo minuto → mensagem amigável de rate limit
5. Suíte offline local verde; comportamento local (HTTP) inalterado
