"""Entry point: router with themed navigation and the shared sidebar."""
import streamlit as st

from app.backend import ApiError, documents
from app.theme import inject_css, sidebar_status
from app.translations import current_language, language_selector, t

st.set_page_config(page_title="AI Research Assistant",
                   page_icon=":material/science:", layout="wide")
inject_css()
lang = current_language()

navigation = st.navigation([
    st.Page("views/chat.py", title=t("nav_chat", lang),
            icon=":material/forum:", default=True),
    st.Page("views/documents.py", title=t("nav_documents", lang),
            icon=":material/description:"),
    st.Page("views/metrics.py", title=t("nav_metrics", lang),
            icon=":material/monitoring:"),
])

language_selector()
try:
    docs = documents()
    sidebar_status(len(docs), sum(d["chunks"] for d in docs),
                   t("status_label", lang))
except ApiError:
    pass  # sem indicador quando o backend está fora — a página mostra o erro

navigation.run()
