---
title: AI Research Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📚 AI Research Assistant — Advanced RAG

**🔴 Live demo:** <https://huggingface.co/spaces/YOUR_HF_USERNAME/ai-research-assistant> — no setup needed, just open and ask.

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

> **Note on the hosted demo:** the Space's disk is ephemeral — uploaded PDFs and
> feedback are cleared whenever the Space restarts. Ask up to 10 questions per
> minute (public rate limit).

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
