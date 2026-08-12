import streamlit as st

from app.backend import (ApiConnectionError, ApiError, documents,
                         remove_document, reset_documents, restore_defaults,
                         upload)
from app.theme import hero
from app.translations import current_language, t

lang = current_language()
hero(t("docs_hero_title", lang), t("docs_hero_sub", lang))
st.caption(":material/info: " + t("session_scope_note", lang))

for message in st.session_state.pop("docs_flash", []):
    if message.get("kind") == "error":
        st.error(message["text"])
    else:
        st.success(message["text"])


def _show_error(exc: Exception) -> None:
    if isinstance(exc, ApiConnectionError):
        st.error(t("api_unreachable", lang))
    else:
        st.error(str(exc))


uploaded_files = st.file_uploader(t("upload_label", lang), type=["pdf"],
                                  accept_multiple_files=True)
if uploaded_files and st.button(":material/library_add: " + t("index_button", lang),
                                type="primary"):
    try:
        with st.spinner(t("indexing", lang)):
            result = upload([(f.name, f.getvalue()) for f in uploaded_files])
    except ApiError as exc:
        _show_error(exc)
    else:
        flash = []
        for r in result["results"]:
            if r["error"] is None:
                flash.append({"kind": "success", "text": t("indexed_ok", lang).format(
                    doc=r["doc_id"], n=r["chunks_added"])})
            else:
                flash.append({"kind": "error", "text": f"'{r['filename']}': {r['error']}"})
        st.session_state.docs_flash = flash
        st.rerun()

st.divider()
st.subheader(t("collection", lang))
try:
    docs = documents()
except ApiError as exc:
    _show_error(exc)
    docs = []

if not docs:
    with st.container(border=True):
        st.markdown(f"**{t('empty_base_title', lang)}**  \n"
                    f"{t('empty_base_text', lang)}")
for doc in docs:
    with st.container(border=True):
        col_info, col_remove = st.columns([8, 1], vertical_alignment="center")
        col_info.markdown(
            f":material/description: **{doc['doc_title']}**  \n"
            f":gray[{doc['chunks']} {t('chunks', lang)}]")
        if col_remove.button(":material/delete:", key=f"rm-{doc['doc_id']}",
                             help=t("remove_doc_help", lang)):
            try:
                with st.spinner(t("removing", lang)):
                    result = remove_document(doc["doc_id"])
            except ApiError as exc:
                _show_error(exc)
            else:
                st.session_state.docs_flash = [{"kind": "success", "text": t(
                    "removed_ok", lang).format(doc=result["doc_id"],
                                               n=result["chunks_removed"])}]
                st.rerun()

st.divider()
st.subheader(t("manage_collection", lang))
if "docs_confirm" not in st.session_state:
    st.session_state.docs_confirm = None

col_clear, col_restore, _ = st.columns([1, 2, 2])
if col_clear.button(":material/delete_sweep: " + t("clear_all", lang)):
    st.session_state.docs_confirm = "clear"
if col_restore.button(":material/history: " + t("restore_defaults_btn", lang)):
    st.session_state.docs_confirm = "restore"

if st.session_state.docs_confirm == "clear":
    st.warning(t("confirm_clear", lang))
    col_yes, col_no, _ = st.columns([1, 1, 4])
    if col_yes.button(t("confirm_yes", lang), key="confirm-clear", type="primary"):
        st.session_state.docs_confirm = None
        try:
            with st.spinner(t("clearing", lang)):
                result = reset_documents()
        except ApiError as exc:
            _show_error(exc)
        else:
            st.session_state.docs_flash = [{"kind": "success", "text": t(
                "cleared_ok", lang).format(n=result["chunks_removed"])}]
            st.rerun()
    if col_no.button(t("cancel", lang), key="cancel-clear"):
        st.session_state.docs_confirm = None
        st.rerun()

if st.session_state.docs_confirm == "restore":
    st.warning(t("confirm_restore", lang))
    col_yes, col_no, _ = st.columns([1, 1, 4])
    if col_yes.button(t("confirm_yes", lang), key="confirm-restore", type="primary"):
        st.session_state.docs_confirm = None
        try:
            with st.spinner(t("restoring", lang)):
                result = restore_defaults()
        except ApiError as exc:
            _show_error(exc)
        else:
            st.session_state.docs_flash = [{"kind": "success", "text": t(
                "restored_ok", lang).format(docs=result["documents_added"],
                                            n=result["chunks_added"])}]
            st.rerun()
    if col_no.button(t("cancel", lang), key="cancel-restore"):
        st.session_state.docs_confirm = None
        st.rerun()
