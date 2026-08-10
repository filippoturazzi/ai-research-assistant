import streamlit as st

from app.backend import ApiConnectionError, ApiError, documents, upload
from app.translations import language_selector, t

lang = language_selector()
st.title(t("docs_title", lang))

uploaded = st.file_uploader(t("upload_label", lang), type=["pdf"])
if uploaded is not None and st.button(t("index_button", lang)):
    try:
        with st.spinner(t("indexing", lang)):
            result = upload(uploaded.name, uploaded.getvalue())
        st.success(t("indexed_ok", lang).format(doc=result["doc_id"],
                                                n=result["chunks_added"]))
    except ApiConnectionError:
        st.error(t("api_unreachable", lang))
    except ApiError as exc:
        st.error(str(exc))

st.divider()
st.subheader(t("collection", lang))
try:
    for doc in documents():
        st.markdown(f"- **{doc['doc_title']}** — {doc['chunks']} {t('chunks', lang)}")
except ApiConnectionError:
    st.error(t("api_unreachable", lang))
except ApiError as exc:
    st.error(str(exc))
