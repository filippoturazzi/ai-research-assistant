# Deploy: Hugging Face Spaces + GitHub — Design

**Data:** 2026-08-09
**Status:** SUBSTITUÍDO em 2026-08-10 — na publicação, o Hugging Face retornou 402: Spaces Docker/Gradio agora exigem assinatura PRO (só Spaces estáticos são gratuitos), e o SDK Streamlit foi descontinuado (400). As Tasks 1-3 do plano (proteções de API, Docker, CI) permanecem válidas e mergeadas; a hospedagem pivotou para o Streamlit Community Cloud — ver `2026-08-10-deploy-streamlit-cloud-design.md`.
**Contexto:** O AI Research Assistant roda local (FastAPI + Streamlit + modelos de embedding em CPU). Este spec o coloca online para demo pública de portfólio, sem custo, testável por URL.

## Decisões

| Decisão | Escolha |
|---|---|
| Hospedagem | Hugging Face Spaces, SDK **Docker** (free tier: 2 vCPU, 16GB RAM, disco efêmero) |
| Fonte da verdade | Repo público no GitHub + GitHub Action espelhando `master` → Space |
| Vercel | Descartada — serverless não roda PyTorch (~limite 250MB) nem processos persistentes/Streamlit |
| Segredo | `GROQ_API_KEY` como Secret do Space (runtime); `HF_TOKEN` como Secret do repo GitHub (deploy) |
| Custo | R$ 0 |

## Arquitetura do deploy

Um único container Docker roda os dois processos:

- **API FastAPI** interna em `localhost:8000` (não exposta publicamente)
- **Streamlit** exposto na porta **7860** (a única porta que o Space publica)
- A UI fala com a API via HTTP (`API_URL=http://localhost:8000`) — arquitetura de 3 camadas preservada no deploy

**Build-time (Dockerfile):**
1. Instala dependências de produção (sem `[dev]`)
2. Roda `scripts/download_papers.py` (papers do arXiv entram na imagem)
3. Roda `scripts/build_index.py` (índice FAISS + chunks.json prontos na imagem)
4. Pré-baixa os modelos (`all-MiniLM-L6-v2` + cross-encoder) para o cache HF da imagem — o passo 3 já baixa o de embeddings; um passo explícito garante o cross-encoder

Resultado: startup do Space em segundos (só carga de modelo em RAM), sem rede no boot além do Groq.

**Runtime (start.sh):**
1. `uvicorn "api.main:create_app" --factory --host 127.0.0.1 --port 8000 &`
2. Aguarda `/health` responder (loop com timeout ~60s)
3. `streamlit run src/app/Home.py --server.port 7860 --server.address 0.0.0.0 --server.headless true`

**Usuário não-root** (requisito do HF Spaces): Dockerfile cria user `user` (uid 1000), define `HOME=/home/user`, tudo roda como ele. Caches HF (`HF_HOME`) apontam para diretório gravável.

**Persistência:** disco efêmero. Uploads de PDF e feedback (SQLite) duram até o restart do Space — comportamento aceito para demo (poluição some sozinha). Nada disso é bug: documentado no README.

## Proteções de demo pública

A `GROQ_API_KEY` do Filip atende anônimos; o free tier da Groq é o teto natural. Mitigações simples, sem dependência nova:

1. **Tamanho da pergunta:** `AskRequest.question: str = Field(max_length=500)` (Pydantic já retorna 422)
2. **Rate limit em memória** no `POST /ask`: máx. **10 requisições/minuto por IP** (janela deslizante simples com `dict[ip, list[timestamps]]` protegido por lock, no `create_app`). Excedeu → HTTP 429 com detail `"Rate limit exceeded — try again in a minute."`. UI mostra a mensagem via fluxo `ApiError` existente.
3. Upload permanece habilitado (efêmero); `/upload` reaproveita o mesmo rate limit.

Fora de escopo: autenticação, captcha, persistência de feedback entre restarts, HTTPS custom (o Space já dá TLS).

## GitHub + sync automático

1. Repo público `ai-research-assistant` no GitHub do Filip (`gh repo create` + push da `master`)
2. Space `ai-research-assistant` na conta HF do Filip (SDK Docker)
3. GitHub Action `.github/workflows/deploy-to-hf.yml`: em push na `master`, faz `git push` forçado do conteúdo para o remote do Space usando `HF_TOKEN` (secret do repo GitHub). Padrão oficial documentado pelo HF.
4. O Space exige um bloco de metadados no topo do `README.md` (front-matter YAML: `title`, `emoji`, `sdk: docker`, `app_port: 7860`, `pinned`). **Decisão: um único README com o front-matter no topo** — o GitHub renderiza front-matter YAML como uma tabela discreta no início da página (padrão comum na comunidade HF), e manter um README só evita duplicação.

## Mudanças no repositório

| Arquivo | Mudança |
|---|---|
| `Dockerfile` | Novo — build conforme acima |
| `start.sh` | Novo — orquestra API + UI |
| `.dockerignore` | Novo — exclui `.venv`, `data/index`, `data/documents/*.pdf`, `.git`, testes, docs |
| `src/api/schemas.py` | `question` com `max_length=500` |
| `src/api/main.py` | Rate limiter em memória (10/min/IP) em `/ask` e `/upload` → 429 |
| `.github/workflows/deploy-to-hf.yml` | Novo — sync GitHub → HF Space |
| `README.md` | Front-matter YAML do Space + seção **Live Demo** no topo (link do Space) + nota sobre disco efêmero |
| `tests/test_api.py` | Testes: 422 para pergunta >500 chars; 429 após exceder o limite (limiter injetável/configurável para teste) |

Detalhe do rate limiter testável: `create_app(service=None, rate_limit: int = 10, rate_window_s: int = 60)` — testes passam `rate_limit=2` e disparam 3 requests.

## Passos manuais do Filip (guiados na execução)

1. Criar conta em huggingface.co (se não tiver) + token de escrita (Settings → Access Tokens)
2. Autenticar `gh` CLI (`gh auth login`) ou criar o repo público manualmente
3. Colar `HF_TOKEN` nos secrets do repo GitHub
4. Criar o Space (Docker) e colar `GROQ_API_KEY` nos Secrets do Space

## Critérios de sucesso

1. URL pública do Space abre a UI; pergunta em inglês responde com citações; seletor PT funciona.
2. Recrutador não precisa de conta, chave, clone ou terminal — só a URL.
3. Push na `master` do GitHub atualiza o Space automaticamente.
4. 11ª pergunta no mesmo minuto → mensagem amigável de rate limit (não stacktrace).
5. Suíte offline local continua verde.
