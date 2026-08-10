import streamlit as st

from app.backend import ApiConnectionError, ApiError, ask, send_feedback
from app.translations import language_selector, t

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")
lang = language_selector()
st.title("📚 " + t("page_title", lang))
st.caption(t("tagline", lang))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "voted" not in st.session_state:
    st.session_state.voted = set()


def _render_sources(sources):
    with st.expander(t("sources", lang).format(n=len(sources))):
        for i, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {s['doc_title']}** — {t('page_abbrev', lang)} {s['page']} "
                f"({t('score', lang)} {s['score']:.2f})"
            )
            st.text(s["text"][:500])


def _render_feedback(interaction_id):
    if interaction_id in st.session_state.voted:
        st.caption(t("feedback_thanks", lang))
        return
    col_up, col_down, _ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"up-{interaction_id}"):
        try:
            send_feedback(interaction_id, 1)
        except ApiConnectionError:
            st.error(t("api_unreachable", lang))
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state.voted.add(interaction_id)
            st.rerun()
    if col_down.button("👎", key=f"down-{interaction_id}"):
        try:
            send_feedback(interaction_id, -1)
        except ApiConnectionError:
            st.error(t("api_unreachable", lang))
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state.voted.add(interaction_id)
            st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])
        if message.get("interaction_id"):
            _render_feedback(message["interaction_id"])

if question := st.chat_input(t("chat_placeholder", lang), max_chars=500):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]][-6:]
        try:
            with st.spinner(t("searching", lang)):
                result = ask(question, history, lang)
        except ApiConnectionError:
            st.error(t("api_unreachable", lang))
        except ApiError as exc:
            st.error(f"{t('ask_failed', lang)} {exc}")
        else:
            st.markdown(result["answer"])
            _render_sources(result["sources"])
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"],
                "sources": result["sources"],
                "interaction_id": result["interaction_id"],
            })
            _render_feedback(result["interaction_id"])
