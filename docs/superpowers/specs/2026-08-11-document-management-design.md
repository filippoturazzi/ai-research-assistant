# Document Management (add/remove/reset) — Design

Date: 2026-08-11
Status: approved (user picked FAISS vector-reconstruction approach)

## Goal

Let the user change the knowledge base at runtime: remove individual papers,
clear the whole collection, and restore the default arXiv collection — from the
existing Streamlit Documents page, in both HTTP and embedded backend modes.
Upload (add) already exists end-to-end; this adds the removal side.

## Approach for removal

FAISS `IndexFlatIP` cannot delete rows, but it can read stored vectors back
(`reconstruct_n`). Removing a document therefore: filters the chunk list,
copies the surviving vectors into a fresh `IndexFlatIP`, and rebuilds BM25 from
the surviving texts. No re-embedding, no embedding model needed.

## Changes by layer

### RAG core (`src/rag`)

- `errors.py`: new `DocumentNotFoundError`, `DownloadError`, and
  `EmptyIndexError(GenerationError)` (subclass so existing 503/ApiError
  handling applies without new plumbing).
- `VectorIndex.reconstruct_all() -> np.ndarray` — all stored vectors.
- `IndexStore.remove(doc_id) -> int` — drop the doc's chunks, rebuild vectors
  (via reconstruct) and BM25; returns removed chunk count (0 if absent).
- `IndexStore.clear()` — empty chunks, fresh vector + BM25 indexes.
- `rag/ingestion/default_papers.py` — `PAPERS` dict (moved from
  `scripts/download_papers.py`, which now imports it) and
  `fetch_default_papers()` yielding `(filename, bytes)`; wraps network errors
  in `DownloadError`.
- `RAGService.remove_document(doc_id) -> int` — raise `DocumentNotFoundError`
  if not indexed; `store.remove`, delete `documents_dir/<doc_id>.pdf`, save
  index.
- `RAGService.reset_documents() -> int` — `store.clear()`, delete all PDFs in
  `documents_dir`, save index; returns chunks removed.
- `RAGService.restore_default_documents(fetch=None) -> dict` — reset, then
  download + ingest each default paper; returns
  `{"documents_added": n, "chunks_added": total}`. `fetch` injectable for
  tests.
- `RAGService.ask` guard: empty store raises `EmptyIndexError` with a friendly
  message instead of calling the LLM with no context.

### API (`src/api/main.py`)

All rate-limited like the existing endpoints:

- `DELETE /documents/{doc_id}` → `{"doc_id", "chunks_removed"}`; 404 on
  `DocumentNotFoundError`.
- `POST /documents/reset` → `{"chunks_removed"}`.
- `POST /documents/restore-defaults` → restore result; 502 on `DownloadError`.

### Client + embedded backend (`src/app`)

- `api_client.py`: `remove_document(doc_id)`, `reset_documents()`,
  `restore_defaults()`.
- `backend.py`: embedded-mode equivalents with `_check_rate` and the existing
  error → `ApiError` mapping.

### UI (`src/app/pages/1_Documents.py` + `translations.py`)

- 🗑️ button per listed document (single click; re-upload recovers).
- "Clear all" and "Restore default collection" buttons with a two-step
  confirmation (session-state flag).
- `st.rerun()` after successful mutations; new en/pt strings.

## Testing

- `test_vector_index.py`: reconstruct_all roundtrip.
- `test_store.py`: remove keeps chunk/vector/BM25 alignment (search still hits
  the right rows), remove absent returns 0, clear, save/load after remove.
- `test_service.py`: remove deletes PDF + persists; missing doc raises;
  reset; restore with fake fetch; ask on empty base raises `EmptyIndexError`.
- `test_api.py`: new endpoints incl. 404/502 and shared rate limit.
- `test_backend_embedded.py`: embedded equivalents.

## Notes

- Hosted demo (Streamlit Community Cloud) has an ephemeral filesystem: base
  changes persist only until restart. Local changes are permanent.
