import streamlit as st

from app.backend import ApiConnectionError, ApiError, metrics
from app.theme import hero
from app.translations import current_language, t

lang = current_language()
hero(t("metrics_title", lang), t("metrics_sub", lang))

try:
    data = metrics()
except ApiConnectionError:
    st.error(t("api_unreachable", lang))
    st.stop()
except ApiError as exc:
    st.error(str(exc))
    st.stop()


def _pct(value):
    return f"{value * 100:.0f}%" if value is not None else "—"


st.caption(t("metrics_scope_note", lang))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(t("m_questions", lang), data["total_questions"])
col2.metric(t("m_feedback", lang), data["feedback_count"])
col3.metric(t("m_approval", lang), _pct(data["approval_rate"]))
col4.metric(t("m_approval_7d", lang), _pct(data["approval_rate_7d"]))
col5.metric(t("m_latency", lang),
            f"{data['avg_latency_ms']:.0f} ms" if data["avg_latency_ms"] else "—")

st.subheader(":material/format_quote: " + t("top_docs", lang))
for doc in data["top_documents"]:
    st.markdown(f"- **{doc['doc_title']}** — {doc['citations']} {t('citations', lang)}")
