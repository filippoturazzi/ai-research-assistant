# Deploy: Hugging Face Spaces + GitHub — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Task 4 is interactive** (requires the user's accounts/tokens) — the controller executes it directly with the user, not via subagent.

**Goal:** Colocar o AI Research Assistant online no Hugging Face Spaces (Docker, gratuito), com GitHub público como fonte da verdade e deploy automático a cada push.

**Architecture:** Um container Docker roda API (interna, 8000) + Streamlit (exposto, 7860); índice e modelos são construídos/baixados no build da imagem. GitHub Action espelha `master` → branch `main` do Space. Proteções de demo pública: pergunta ≤500 chars e rate limit em memória 10/min/IP.

**Tech Stack:** Docker (python:3.12-slim), Hugging Face Spaces (SDK docker), GitHub Actions, FastAPI/Streamlit existentes.

**Spec:** `docs/superpowers/specs/2026-08-09-deploy-hf-spaces-design.md`

## Global Constraints

- Porta pública do Space: **7860** (Streamlit); API só em `127.0.0.1:8000` dentro do container.
- Container roda como usuário não-root `user` (uid 1000), `HOME=/home/user`, `HF_HOME` gravável.
- Build da imagem: instala produção (sem `[dev]`), baixa papers, constrói índice, pré-baixa cross-encoder — startup sem rede além do Groq.
- Rate limit: 10 req/min/IP em `/ask` e `/upload`, HTTP 429 com detail `"Rate limit exceeded — try again in a minute."`; parâmetros injetáveis em `create_app` para teste.
- `question` com `max_length=500` (422 automático do Pydantic).
- Segredos: `GROQ_API_KEY` só como Secret do Space; `HF_TOKEN`/`HF_USERNAME` só como Secrets do repo GitHub. Nunca em arquivo commitado.
- README único com front-matter YAML do Space no topo (`sdk: docker`, `app_port: 7860`).
- Suíte offline continua verde. Commits `feat:`/`chore:`/`docs:`/`ci:` em inglês.

---

### Task 1: Proteções de demo pública na API

**Files:**
- Modify: `src/api/schemas.py` (campo `question`)
- Modify: `src/api/main.py` (assinatura de `create_app` + rate limiter + endpoints `/ask` e `/upload`)
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `create_app(service=None, rate_limit: int = 10, rate_window_s: int = 60) -> FastAPI`. Comportamento: cada IP pode fazer até `rate_limit` requisições a `/ask`+`/upload` (contador compartilhado entre os dois) por janela de `rate_window_s` segundos; excedeu → 429.

- [ ] **Step 1: Escrever testes que falham** — adicionar a `tests/test_api.py`:

```python
def test_question_too_long_rejected(client):
    c, _ = client
    assert c.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_question_at_limit_accepted(client):
    c, _ = client
    assert c.post("/ask", json={"question": "x" * 500}).status_code == 200


def test_rate_limit_returns_429():
    service = FakeService()
    app = create_app(service=service, rate_limit=2, rate_window_s=60)
    c = TestClient(app)
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 429
    assert "Rate limit" in r.json()["detail"]


def test_rate_limit_shared_with_upload():
    import io
    service = FakeService()
    app = create_app(service=service, rate_limit=2, rate_window_s=60)
    c = TestClient(app)
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    pdf = ("file", ("ok.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[pdf]).status_code == 200
    assert c.post("/ask", json={"question": "q?"}).status_code == 429
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv\Scripts\pytest tests/test_api.py -v`
Expected: FAIL (422 não ocorre; TypeError em `create_app(rate_limit=...)`)

- [ ] **Step 3: Implementar** — em `src/api/schemas.py`:

```python
from pydantic import BaseModel, Field
# ...
class AskRequest(BaseModel):
    question: str = Field(max_length=500)
    history: list[HistoryMessage] = []
    language: Literal["en", "pt"] = "en"
```

Em `src/api/main.py` — novos imports `import threading`, `import time`, `from fastapi import FastAPI, HTTPException, Request, UploadFile`; assinatura e limiter:

```python
def create_app(service=None, rate_limit: int = 10, rate_window_s: int = 60) -> FastAPI:
    app = FastAPI(title="AI Research Assistant", version="0.1.0")
    app.state.service = service or _build_real_service()

    hits: dict[str, list[float]] = {}
    hits_lock = threading.Lock()

    def _check_rate(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with hits_lock:
            window = [t for t in hits.get(ip, []) if now - t < rate_window_s]
            if len(window) >= rate_limit:
                hits[ip] = window
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded — try again in a minute.",
                )
            window.append(now)
            hits[ip] = window
```

E nos dois endpoints, adicionar o parâmetro `request: Request` e a checagem como primeira linha:

```python
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        _check_rate(request)
        # ... corpo existente inalterado

    @app.post("/upload", response_model=UploadResponse)
    async def upload(file: UploadFile, request: Request):
        _check_rate(request)
        # ... corpo existente inalterado
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv\Scripts\pytest tests/test_api.py -v` e depois `.venv\Scripts\pytest -m "not integration" -q`
Expected: tudo PASS (os testes existentes usam < 10 requisições por app criada — sem colisão com o limite default).

- [ ] **Step 5: Commit**

```bash
git add src/api tests/test_api.py
git commit -m "feat: question length cap and in-memory IP rate limit for public demo"
```

---

### Task 2: Dockerfile + start.sh + .dockerignore

**Files:**
- Create: `Dockerfile`, `start.sh`, `.dockerignore`

**Interfaces:**
- Consumes: `scripts/download_papers.py`, `scripts/build_index.py`, `uvicorn "api.main:create_app" --factory`, `streamlit run src/app/Home.py`, `GET /health`
- Produces: imagem que o HF Spaces builda e roda (porta 7860). Task 3 depende do `app_port: 7860` no README.

- [ ] **Step 1: Criar `.dockerignore`**

```
.venv/
.git/
.github/
.superpowers/
.claude/
data/index/
data/documents/*.pdf
data/feedback.db
docs/
tests/
.env
.env.example
__pycache__/
*.egg-info/
.pytest_cache/
README.md
```

- [ ] **Step 2: Criar `Dockerfile`**

```dockerfile
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app
RUN chown user /app
USER user

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user scripts ./scripts

RUN pip install --no-cache-dir --user .

# Bake the default collection and index into the image
# (downloads the papers and the embedding model at build time)
RUN python scripts/download_papers.py && python scripts/build_index.py

# Pre-download the cross-encoder so startup needs no network
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user start.sh ./
EXPOSE 7860
CMD ["bash", "start.sh"]
```

- [ ] **Step 3: Criar `start.sh`** (line endings LF — importante no Windows; usar `git add --renormalize` se necessário e conferir que o arquivo não tem CRLF)

```bash
#!/usr/bin/env bash
set -e

uvicorn "api.main:create_app" --factory --host 127.0.0.1 --port 8000 &

python - <<'EOF'
import time
import urllib.request

for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        print("API is up.")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("API did not start within 60s")
EOF

exec streamlit run src/app/Home.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
```

- [ ] **Step 4: Garantir LF no start.sh** — criar `.gitattributes` na raiz com:

```
start.sh text eol=lf
*.sh text eol=lf
```

- [ ] **Step 5: Verificação** — se `docker --version` funcionar na máquina: `docker build -t ai-research-assistant .` e depois `docker run --rm -p 7860:7860 -e GROQ_API_KEY=dummy ai-research-assistant`, aguardar e conferir `http://localhost:7860` (UI abre; perguntas falharão com "LLM unavailable" por causa da chave dummy — esperado). Se Docker não estiver disponível: validar `bash -n start.sh` (sintaxe) e deixar o build real para o Space (Task 4) — anotar isso no report.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile start.sh .dockerignore .gitattributes
git commit -m "feat: Docker image with baked index for Hugging Face Spaces"
```

---

### Task 3: GitHub Action de deploy + README (front-matter + Live Demo)

**Files:**
- Create: `.github/workflows/deploy-to-hf.yml`
- Modify: `README.md` (front-matter no topo + seção Live Demo + nota de disco efêmero)

**Interfaces:**
- Consumes: Secrets do repo GitHub `HF_TOKEN` e `HF_USERNAME` (criados na Task 4)
- Produces: push na `master` → Space atualizado. O placeholder `YOUR_HF_USERNAME` no README é substituído na Task 4.

- [ ] **Step 1: Criar `.github/workflows/deploy-to-hf.yml`**

```yaml
name: Deploy to Hugging Face Space

on:
  push:
    branches: [master]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
        run: |
          git push --force \
            "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/ai-research-assistant" \
            master:main
```

- [ ] **Step 2: Editar `README.md`** — inserir NO TOPO (antes de tudo):

```markdown
---
title: AI Research Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

Logo após o título `# 📚 AI Research Assistant — Advanced RAG`, inserir:

```markdown
**🔴 Live demo:** <https://huggingface.co/spaces/YOUR_HF_USERNAME/ai-research-assistant> — no setup needed, just open and ask.
```

E ao final da seção de features (após o bullet "Bilingual UI"), adicionar:

```markdown
> **Note on the hosted demo:** the Space's disk is ephemeral — uploaded PDFs and
> feedback are cleared whenever the Space restarts. Ask up to 10 questions per
> minute (public rate limit).
```

- [ ] **Step 3: Validar YAML** — `python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/deploy-to-hf.yml').read_text())"` (PyYAML vem com o venv via dependências transitivas; se não, validação visual basta — o Action é curto).

- [ ] **Step 4: Suíte** — `.venv\Scripts\pytest -m "not integration" -q` → verde (mudanças são só docs/CI).

- [ ] **Step 5: Commit**

```bash
git add .github README.md
git commit -m "ci: auto-deploy master to Hugging Face Space; README demo section"
```

---

### Task 4: Publicação (INTERATIVA — controller + usuário)

**Files/Recursos:** repo GitHub público, Space HF, secrets. Sem código novo além da substituição do placeholder no README.

**Pré-requisitos do usuário (guiar um de cada vez):**
1. Conta em <https://huggingface.co> (grátis) e token de escrita em Settings → Access Tokens → New token (type: **Write**)
2. `gh auth login` no terminal (GitHub CLI) — ou repo criado manualmente no site

- [ ] **Step 1: Criar e subir o repo GitHub**

```bash
gh repo create ai-research-assistant --public --source . --push
```

(Confirma: branch `master` publicada; `.env`, `data/` gerado e `.superpowers/` fora do git pelo .gitignore.)

- [ ] **Step 2: Criar o Space** — com o token HF do usuário (pedir que cole no momento; NÃO gravar em arquivo):

```bash
.venv\Scripts\python -c "from huggingface_hub import create_repo; create_repo('USERNAME/ai-research-assistant', repo_type='space', space_sdk='docker', token='TOKEN_AQUI')"
```

- [ ] **Step 3: Configurar secrets**
- No Space: Settings → Variables and secrets → New secret → `GROQ_API_KEY` = chave do usuário (ou via `huggingface_hub.add_space_secret`)
- No repo GitHub: `gh secret set HF_TOKEN` (colar o token) e `gh secret set HF_USERNAME --body "USERNAME"`

- [ ] **Step 4: Substituir `YOUR_HF_USERNAME` no README** pelo username real, commitar (`docs: point live demo link to the real Space`) e `git push`.

- [ ] **Step 5: Acompanhar o deploy** — o push dispara o Action (conferir `gh run watch`); no Space, acompanhar o build da imagem (download de papers + modelos leva ~5-10 min na primeira vez). Quando ficar **Running**: abrir a URL, fazer uma pergunta em inglês, trocar para PT, conferir citações.

- [ ] **Step 6: Critérios de sucesso do spec** — validar os 5 (URL pública funciona; zero setup p/ visitante; push atualiza o Space; 429 amigável; suíte local verde).

---

## Cobertura do spec → tasks

| Requisito do spec | Task |
|---|---|
| max_length=500 + rate limit 10/min/IP → 429 (injetável p/ teste) | 1 |
| Dockerfile (non-root, índice+modelos no build) | 2 |
| start.sh (API interna + wait health + Streamlit 7860) | 2 |
| .dockerignore | 2 |
| GitHub Action master → Space main | 3 |
| README: front-matter Space + Live Demo + nota efêmero | 3, 4 (username real) |
| Repo GitHub público + Space + secrets | 4 |
| Critérios de sucesso 1-5 | 4 |
