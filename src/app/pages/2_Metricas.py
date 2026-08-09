import streamlit as st

from app.api_client import ApiError, metrics

st.title("📊 Métricas")

try:
    data = metrics()
except ApiError as exc:
    st.error(str(exc))
    st.stop()


def _pct(value):
    return f"{value * 100:.0f}%" if value is not None else "—"


col1, col2, col3, col4 = st.columns(4)
col1.metric("Perguntas", data["total_questions"])
col2.metric("Aprovação", _pct(data["approval_rate"]))
col3.metric("Aprovação (7d)", _pct(data["approval_rate_7d"]))
col4.metric("Latência média",
            f"{data['avg_latency_ms']:.0f} ms" if data["avg_latency_ms"] else "—")

st.subheader("Perguntas com 👎 (fila de investigação)")
if not data["negatives"]:
    st.caption("Nenhum feedback negativo. 🎉")
for item in data["negatives"]:
    with st.expander(f"{item['created_at']} — {item['query']}"):
        st.markdown(item["answer"])
        st.json(item["sources"])

st.subheader("Documentos mais citados")
for doc in data["top_documents"]:
    st.markdown(f"- **{doc['doc_title']}** — {doc['citations']} citações")
