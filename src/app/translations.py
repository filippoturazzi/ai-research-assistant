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
    "upload_label": {
        "en": "Add PDFs to the collection (select or drag several at once)",
        "pt": "Adicionar PDFs à coleção (selecione ou arraste vários de uma vez)",
    },
    "index_button": {"en": "Index documents", "pt": "Indexar documentos"},
    "indexing": {
        "en": "Extracting, chunking and indexing...",
        "pt": "Extraindo, chunkeando e indexando...",
    },
    "indexed_ok": {
        "en": "'{doc}' indexed: {n} chunks.",
        "pt": "'{doc}' indexado: {n} chunks.",
    },
    "collection": {"en": "Current collection", "pt": "Coleção atual"},
    "no_documents": {
        "en": "No documents indexed — upload a PDF above or restore the default collection.",
        "pt": "Nenhum documento indexado — envie um PDF acima ou restaure a coleção padrão.",
    },
    "remove_doc": {"en": "🗑️ Remove", "pt": "🗑️ Remover"},
    "removing": {"en": "Removing...", "pt": "Removendo..."},
    "removed_ok": {
        "en": "'{doc}' removed ({n} chunks).",
        "pt": "'{doc}' removido ({n} chunks).",
    },
    "manage_collection": {"en": "Replace the whole collection", "pt": "Trocar a base inteira"},
    "clear_all": {"en": "🗑️ Clear all", "pt": "🗑️ Limpar tudo"},
    "restore_defaults_btn": {
        "en": "📥 Restore default collection",
        "pt": "📥 Restaurar coleção padrão",
    },
    "confirm_clear": {
        "en": "This removes ALL documents from the knowledge base. Continue?",
        "pt": "Isso remove TODOS os documentos da base de conhecimento. Continuar?",
    },
    "confirm_restore": {
        "en": "This replaces the whole collection with the 5 classic papers "
              "(downloaded from arXiv). Continue?",
        "pt": "Isso substitui a coleção inteira pelos 5 papers clássicos "
              "(baixados do arXiv). Continuar?",
    },
    "confirm_yes": {"en": "Confirm", "pt": "Confirmar"},
    "cancel": {"en": "Cancel", "pt": "Cancelar"},
    "clearing": {"en": "Removing all documents...", "pt": "Removendo todos os documentos..."},
    "restoring": {
        "en": "Downloading and indexing the default papers (this can take a few minutes)...",
        "pt": "Baixando e indexando os papers padrão (pode levar alguns minutos)...",
    },
    "cleared_ok": {
        "en": "Collection cleared ({n} chunks removed).",
        "pt": "Coleção limpa ({n} chunks removidos).",
    },
    "restored_ok": {
        "en": "Default collection restored: {docs} papers, {n} chunks.",
        "pt": "Coleção padrão restaurada: {docs} papers, {n} chunks.",
    },
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
