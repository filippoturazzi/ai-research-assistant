import streamlit as st

from app.api_client import ApiError, documents, upload

st.title("📄 Documentos")

uploaded = st.file_uploader("Adicionar PDF à coleção", type=["pdf"])
if uploaded is not None and st.button("Indexar documento"):
    try:
        with st.spinner("Extraindo, chunkeando e indexando..."):
            result = upload(uploaded.name, uploaded.getvalue())
        st.success(f"'{result['doc_id']}' indexado: {result['chunks_added']} chunks.")
    except ApiError as exc:
        st.error(str(exc))

st.divider()
st.subheader("Coleção atual")
try:
    for doc in documents():
        st.markdown(f"- **{doc['doc_title']}** — {doc['chunks']} chunks")
except ApiError as exc:
    st.error(str(exc))
