import streamlit as st

TRANSLATIONS = {
    "language_label": {"en": "Language", "pt": "Idioma"},
    "page_title": {"en": "AI Research Assistant", "pt": "AI Research Assistant"},

    # navigation
    "nav_chat": {"en": "Chat", "pt": "Chat"},
    "nav_documents": {"en": "Documents", "pt": "Documentos"},
    "nav_metrics": {"en": "Metrics", "pt": "Métricas"},
    "status_label": {"en": "documents · indexed chunks",
                     "pt": "documentos · trechos indexados"},

    # chat page
    "hero_chat_title": {"en": "Chat with your documents",
                        "pt": "Converse com os seus documentos"},
    "hero_chat_sub": {
        "en": "Every answer cites its sources — trust, but verify.",
        "pt": "Cada resposta vem com as fontes — confie, mas confira.",
    },
    "step1_title": {"en": "1 · ADD", "pt": "1 · COLOQUE"},
    "step1_text": {"en": "Upload your PDFs on the Documents page",
                   "pt": "Envie seus PDFs na página Documentos"},
    "step2_title": {"en": "2 · ASK", "pt": "2 · PERGUNTE"},
    "step2_text": {"en": "In English or Portuguese, about their content",
                   "pt": "Em português ou inglês, sobre o conteúdo"},
    "step3_title": {"en": "3 · RATE", "pt": "3 · AVALIE"},
    "step3_text": {"en": "Thumbs help measure answer quality",
                   "pt": "O polegar ajuda a medir a qualidade"},
    "try_asking": {"en": "Try asking:", "pt": "Experimente perguntar:"},
    "example_q1": {"en": "What is the attention mechanism?",
                   "pt": "O que é o mecanismo de atenção?"},
    "example_q2": {"en": "How does retrieval reduce hallucination?",
                   "pt": "Como a recuperação reduz alucinação?"},
    "example_q3": {"en": "What does BERT do differently from GPT-3?",
                   "pt": "O que o BERT faz de diferente do GPT-3?"},
    "empty_base_title": {"en": "Start here", "pt": "Comece aqui"},
    "empty_base_text": {
        "en": "The knowledge base is empty. Upload your PDFs or restore the "
              "default collection on the Documents page.",
        "pt": "A base está vazia. Envie seus PDFs ou restaure a coleção "
              "padrão na página Documentos.",
    },
    "go_to_documents": {"en": "Open Documents", "pt": "Abrir Documentos"},
    "chat_placeholder": {
        "en": "Ask a question about the documents...",
        "pt": "Faça uma pergunta sobre os documentos...",
    },
    "searching": {"en": "Searching the documents...", "pt": "Buscando nos documentos..."},
    "sources": {"en": "Sources ({n})", "pt": "Fontes ({n})"},
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

    # documents page
    "docs_hero_title": {"en": "Your knowledge base", "pt": "Sua base de conhecimento"},
    "docs_hero_sub": {
        "en": "The assistant only answers from what's here. Swap the whole "
              "base anytime: clear the documents and add your own.",
        "pt": "O assistente só responde com o que está aqui. Troque a base "
              "inteira quando quiser: limpe os documentos e coloque os seus.",
    },
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
    "chunks": {"en": "indexed chunks", "pt": "trechos indexados"},
    "no_documents": {
        "en": "No documents indexed yet.",
        "pt": "Nenhum documento indexado ainda.",
    },
    "remove_doc_help": {"en": "Remove from the base", "pt": "Remover da base"},
    "removing": {"en": "Removing...", "pt": "Removendo..."},
    "removed_ok": {
        "en": "'{doc}' removed ({n} chunks).",
        "pt": "'{doc}' removido ({n} chunks).",
    },
    "manage_collection": {"en": "Replace the whole collection", "pt": "Trocar a base inteira"},
    "clear_all": {"en": "Clear all", "pt": "Limpar tudo"},
    "restore_defaults_btn": {
        "en": "Restore default collection",
        "pt": "Restaurar coleção padrão",
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

    # metrics page
    "metrics_title": {"en": "Metrics", "pt": "Métricas"},
    "metrics_sub": {
        "en": "How the assistant is performing, according to its users.",
        "pt": "Como o assistente está se saindo, segundo quem usa.",
    },
    "m_questions": {"en": "Questions", "pt": "Perguntas"},
    "m_approval": {"en": "Approval", "pt": "Aprovação"},
    "m_approval_7d": {"en": "Approval (7d)", "pt": "Aprovação (7d)"},
    "m_latency": {"en": "Avg latency", "pt": "Latência média"},
    "negatives_title": {
        "en": "Thumbs-down questions (investigation queue)",
        "pt": "Perguntas com polegar para baixo (fila de investigação)",
    },
    "no_negatives": {"en": "No negative feedback.", "pt": "Nenhum feedback negativo."},
    "top_docs": {"en": "Most cited documents", "pt": "Documentos mais citados"},
    "citations": {"en": "citations", "pt": "citações"},
}

_LANGUAGE_NAMES = {"en": "English", "pt": "Português"}


def t(key: str, language: str) -> str:
    return TRANSLATIONS[key][language]


def current_language() -> str:
    if "language" not in st.session_state:
        st.session_state.language = "en"
    return st.session_state.language


def language_selector() -> str:
    current_language()
    st.sidebar.selectbox(
        t("language_label", st.session_state.language),
        options=["en", "pt"],
        format_func=lambda code: _LANGUAGE_NAMES[code],
        key="language",
    )
    return st.session_state.language
