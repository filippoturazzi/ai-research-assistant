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
