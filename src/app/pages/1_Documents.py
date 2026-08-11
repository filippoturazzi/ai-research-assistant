import streamlit as st

from app.backend import (ApiConnectionError, ApiError, documents,
                         remove_document, reset_documents, restore_defaults,
                         upload)
from app.translations import language_selector, t

lang = language_selector()
st.title(t("docs_title", lang))

if message := st.session_state.pop("docs_flash", None):
    st.success(message)


def _show_error(exc: Exception) -> None:
    if isinstance(exc, ApiConnectionError):
        st.error(t("api_unreachable", lang))
    else:
        st.error(str(exc))


uploaded = st.file_uploader(t("upload_label", lang), type=["pdf"])
if uploaded is not None and st.button(t("index_button", lang)):
    try:
        with st.spinner(t("indexing", lang)):
            result = upload(uploaded.name, uploaded.getvalue())
    except ApiError as exc:
        _show_error(exc)
    else:
        st.session_state.docs_flash = t("indexed_ok", lang).format(
            doc=result["doc_id"], n=result["chunks_added"])
        st.rerun()

st.divider()
st.subheader(t("collection", lang))
try:
    docs = documents()
except ApiError as exc:
    _show_error(exc)
    docs = []

if not docs:
    st.info(t("no_documents", lang))
for doc in docs:
    col_info, col_remove = st.columns([5, 1])
    col_info.markdown(f"**{doc['doc_title']}** — {doc['chunks']} {t('chunks', lang)}")
    if col_remove.button(t("remove_doc", lang), key=f"rm-{doc['doc_id']}"):
        try:
            with st.spinner(t("removing", lang)):
                result = remove_document(doc["doc_id"])
        except ApiError as exc:
            _show_error(exc)
        else:
            st.session_state.docs_flash = t("removed_ok", lang).format(
                doc=result["doc_id"], n=result["chunks_removed"])
            st.rerun()

st.divider()
st.subheader(t("manage_collection", lang))
if "docs_confirm" not in st.session_state:
    st.session_state.docs_confirm = None

col_clear, col_restore, _ = st.columns([1, 2, 2])
if col_clear.button(t("clear_all", lang)):
    st.session_state.docs_confirm = "clear"
if col_restore.button(t("restore_defaults_btn", lang)):
    st.session_state.docs_confirm = "restore"

if st.session_state.docs_confirm == "clear":
    st.warning(t("confirm_clear", lang))
    col_yes, col_no, _ = st.columns([1, 1, 4])
    if col_yes.button(t("confirm_yes", lang), key="confirm-clear"):
        st.session_state.docs_confirm = None
        try:
            with st.spinner(t("clearing", lang)):
                result = reset_documents()
        except ApiError as exc:
            _show_error(exc)
        else:
            st.session_state.docs_flash = t("cleared_ok", lang).format(
                n=result["chunks_removed"])
            st.rerun()
    if col_no.button(t("cancel", lang), key="cancel-clear"):
        st.session_state.docs_confirm = None
        st.rerun()

if st.session_state.docs_confirm == "restore":
    st.warning(t("confirm_restore", lang))
    col_yes, col_no, _ = st.columns([1, 1, 4])
    if col_yes.button(t("confirm_yes", lang), key="confirm-restore"):
        st.session_state.docs_confirm = None
        try:
            with st.spinner(t("restoring", lang)):
                result = restore_defaults()
        except ApiError as exc:
            _show_error(exc)
        else:
            st.session_state.docs_flash = t("restored_ok", lang).format(
                docs=result["documents_added"], n=result["chunks_added"])
            st.rerun()
    if col_no.button(t("cancel", lang), key="cancel-restore"):
        st.session_state.docs_confirm = None
        st.rerun()
