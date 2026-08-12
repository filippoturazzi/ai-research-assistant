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


col1, col2, col3, col4 = st.columns(4)
col1.metric(t("m_questions", lang), data["total_questions"])
col2.metric(t("m_approval", lang), _pct(data["approval_rate"]))
col3.metric(t("m_approval_7d", lang), _pct(data["approval_rate_7d"]))
col4.metric(t("m_latency", lang),
            f"{data['avg_latency_ms']:.0f} ms" if data["avg_latency_ms"] else "—")

st.subheader(":material/thumb_down: " + t("negatives_title", lang))
if not data["negatives"]:
    st.caption(t("no_negatives", lang))
for item in data["negatives"]:
    with st.expander(f"{item['created_at']} — {item['query']}"):
        st.markdown(item["answer"])
        st.json(item["sources"])

st.subheader(":material/format_quote: " + t("top_docs", lang))
for doc in data["top_documents"]:
    st.markdown(f"- **{doc['doc_title']}** — {doc['citations']} {t('citations', lang)}")
