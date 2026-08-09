# AI Research Assistant — Design

**Data:** 2026-08-09
**Status:** Aprovado em conversa; aguardando revisão final do spec
**Objetivo:** Primeiro projeto RAG de portfólio para vagas de engenheiro de AI

## Visão geral

Sistema de perguntas e respostas sobre uma coleção de documentos (RAG avançado). O usuário faz perguntas em um chat e recebe respostas fundamentadas nos documentos, com citações de fonte (documento + página). Vem com uma coleção padrão de papers clássicos de IA e aceita upload de novos PDFs. Feedback 👍/👎 é armazenado e exibido em um dashboard de métricas.

**Decisões fixadas:**

| Decisão | Escolha |
|---|---|
| Linguagem | Python |
| Interface | Streamlit (UI) + FastAPI (API) |
| LLM | Groq (free tier) — `llama-3.3-70b-versatile` (geração), `llama-3.1-8b-instant` (reescrita de query) |
| Embeddings | `sentence-transformers` local — `all-MiniLM-L6-v2` (384 dim, CPU) |
| Índice vetorial | FAISS (`faiss-cpu`, `IndexFlatIP` com vetores normalizados) |
| Busca lexical | BM25 (`rank-bm25`) |
| Re-ranking | Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Persistência de feedback | SQLite (stdlib `sqlite3`) |
| Abordagem | Pipeline explícito ("from scratch") com bibliotecas focadas — sem LangChain/LlamaIndex |
| Docs padrão | Papers clássicos de IA baixados do arXiv por script (não versionados no git) |

**Custo total de operação: R$ 0** — Groq free tier + modelos locais em CPU.

## Arquitetura

Três camadas, cada uma testável de forma independente:

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│  UI Streamlit    │ ───────────► │   API FastAPI     │
│  - Chat          │              │  /ask  /upload    │
│  - Upload PDFs   │              │  /feedback        │
│  - Dashboard     │              │  /metrics         │
└─────────────────┘              └────────┬─────────┘
                                          │ chama
                                 ┌────────▼─────────┐
                                 │  Núcleo RAG (lib) │
                                 │  ingestão │ busca  │
                                 │  re-rank  │ geração│
                                 └──────────────────┘
```

O núcleo RAG é uma biblioteca Python pura: não conhece API nem UI, funciona em notebook, é testável sem servidor. A API é uma casca fina por cima; a UI Streamlit só fala HTTP com a API.

### Estrutura de pastas

```
ai-research-assistant/
├── src/rag/            # núcleo RAG — sem saber que API/UI existem
│   ├── ingestion/      # extração de PDF (pypdf), chunking
│   ├── retrieval/      # FAISS, BM25, fusão RRF, re-ranking
│   ├── generation/     # cliente Groq, prompts, citações
│   └── feedback/       # armazenamento SQLite, métricas
├── src/api/            # FastAPI — endpoints finos que chamam o núcleo
├── src/app/            # Streamlit — só UI, fala com a API via HTTP
├── data/documents/     # PDFs padrão (baixados por script)
├── data/index/         # índice FAISS + metadados (gerado, fora do git)
├── scripts/            # build_index.py, download_papers.py
└── tests/              # testes do núcleo RAG e da API
```

## Pipeline de ingestão (documento → índice)

Aplica-se aos PDFs padrão e a uploads do usuário:

1. **Extração** — `pypdf` extrai texto página por página, preservando o número da página em cada trecho (base das citações).
2. **Chunking** — chunks de ~800 tokens com overlap de ~150, respeitando limites de parágrafo quando possível. Metadados por chunk: `doc_id`, título do documento, página, posição.
3. **Embeddings** — `all-MiniLM-L6-v2` gera vetor de 384 dimensões por chunk.
4. **Indexação dupla** — vetores normalizados entram no FAISS (`IndexFlatIP` = cosseno; busca exata, adequada para poucos milhares de chunks); textos dos chunks alimentam o índice BM25.
5. **Persistência** — índice FAISS em disco + store de chunks (textos e metadados) em `chunks.json`. A API carrega tudo em memória na inicialização. (SQLite é usado apenas para interações/feedback.)

**Scripts:**

- `scripts/download_papers.py` — baixa do arXiv os papers clássicos (Attention Is All You Need; RAG original; BERT; GPT-3; e afins) para `data/documents/`.
- `scripts/build_index.py` — roda o pipeline de ingestão sobre `data/documents/`.

Setup de quem clona o repo: instalar dependências → rodar os dois scripts → subir API e UI.

Uploads pela UI passam pelo mesmo pipeline de forma **incremental**: adiciona chunks aos índices em memória e re-persiste, sem reconstruir o que já existe.

## Pipeline de consulta (pergunta → resposta)

1. **Reescrita de query** — `llama-3.1-8b-instant` recebe pergunta + histórico do chat e produz uma query de busca autocontida (resolve anáforas tipo "e as limitações disso?", expande siglas). Pergunta já autocontida passa quase inalterada.
2. **Busca híbrida** — a query reescrita roda nos dois índices: FAISS top-20 (semântica) e BM25 top-20 (lexical). **Reciprocal Rank Fusion** combina: `score(chunk) = Σ 1/(60 + posição_na_lista)`.
3. **Re-ranking** — os candidatos da fusão (~30 após dedup) passam pelo cross-encoder, que pontua (pergunta, chunk) em conjunto. Sobrevivem os **top-5**.
4. **Geração com citações** — top-5 chunks numerados `[1]..[5]` (com título e página) no prompt do `llama-3.3-70b-versatile`. Instruções do prompt: responder somente com base no contexto; citar `[n]` em cada afirmação; responder "não encontrei essa informação nos documentos" quando o contexto não cobre a pergunta.
5. **Resposta na UI** — texto com citações + painel expansível "Fontes" (chunk, documento, página) + botões 👍/👎.

Toda consulta gera um registro em `interactions` (ver abaixo) e a API retorna o `interaction_id` para a UI vincular o feedback.

## Feedback e métricas

**SQLite, duas tabelas:**

- `interactions` — id, timestamp, query original, query reescrita, resposta, chunks usados (JSON: doc, página, score), modelo, latência total.
- `feedback` — interaction_id (FK), rating (+1/−1), comentário opcional, timestamp.

**Endpoints:** `POST /feedback` (grava rating), `GET /metrics` (agregados para o dashboard).

**Dashboard (página "Métricas" no Streamlit):**

- Taxa de aprovação geral e últimos 7 dias
- Total de perguntas e latência média
- Perguntas com 👎, com resposta e fontes usadas (fila de investigação)
- Documentos mais citados nas respostas

Sem gráficos complexos: números grandes + tabelas. O feedback **não** altera o ranking automaticamente (fase futura possível); o valor aqui é coletar dados vinculados aos chunks usados.

## API — endpoints

| Método | Rota | Função |
|---|---|---|
| POST | `/ask` | Pergunta + histórico → resposta com citações, fontes e `interaction_id` |
| POST | `/upload` | Recebe PDF, roda ingestão incremental |
| POST | `/feedback` | Grava 👍/👎 de uma interação |
| GET | `/metrics` | Agregados para o dashboard |
| GET | `/documents` | Lista documentos indexados |
| GET | `/health` | Status (índice carregado, nº de chunks/documentos) |

Documentação automática via Swagger (`/docs`) — vitrine de portfólio.

## Tratamento de erros

- **Groq indisponível / rate limit** → retry com backoff (2 tentativas); depois, erro claro da API e mensagem amigável na UI ("serviço de geração indisponível"). Nunca stacktrace na tela.
- **Falha na reescrita de query** → degradação graciosa: busca com a query original.
- **PDF corrompido/sem texto** → mensagem específica no upload; sistema segue funcionando.
- **Índice ausente na inicialização** → erro instruindo a rodar `scripts/build_index.py`.

## Testes (pytest)

- **Chunking** — tamanhos, overlap, preservação de metadados.
- **RRF** — listas de entrada conhecidas produzem o ranking esperado.
- **Retrieval ponta a ponta** — mini-corpus fixo de teste; "pergunta X traz chunk Y no top-5" (sem LLM envolvido).
- **Montagem de prompt** — contexto numerado corretamente com fontes.
- **API** — endpoints com núcleo RAG mockado; sem chamadas reais ao Groq nos testes.

## Fora de escopo (YAGNI / fases futuras)

- Avaliação automática com RAGAS (candidata a fase 2)
- Feedback influenciando o ranking
- Autenticação/multiusuário
- Suporte a formatos além de PDF
- Deploy em nuvem (o projeto roda local; deploy pode ser fase futura)

## Critérios de sucesso

1. Clonar o repo, rodar 2 scripts e 2 comandos → sistema funcionando com os papers padrão.
2. Perguntas sobre os papers retornam respostas corretas com citações verificáveis (documento + página).
3. Pergunta fora do escopo dos documentos → sistema admite que não sabe, em vez de alucinar.
4. Upload de um PDF novo → perguntas sobre ele funcionam imediatamente.
5. Feedback registrado aparece no dashboard.
6. Suíte de testes verde, rodando sem rede (LLM mockado).
