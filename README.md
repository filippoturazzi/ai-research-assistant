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
