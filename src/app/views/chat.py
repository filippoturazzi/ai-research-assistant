import streamlit as st

from app.backend import (ApiConnectionError, ApiError, ask, documents,
                         send_feedback, suggestions)
from app.theme import hero, step_cards
from app.translations import current_language, t

lang = current_language()
hero(t("hero_chat_title", lang), t("hero_chat_sub", lang))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "voted" not in st.session_state:
    st.session_state.voted = set()


def _render_sources(sources):
    with st.expander(":material/description: " + t("sources", lang).format(n=len(sources))):
        for i, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {s['doc_title']}** — {t('page_abbrev', lang)} {s['page']} "
                f"({t('score', lang)} {s['score']:.2f})"
            )
            st.text(s["text"][:500])


def _render_feedback(interaction_id):
    voted = interaction_id in st.session_state.voted
    selected = st.feedback("thumbs", key=f"fb-{interaction_id}", disabled=voted)
    if voted:
        st.caption(t("feedback_thanks", lang))
        return
    if selected is None:
        return
    try:
        send_feedback(interaction_id, 1 if selected == 1 else -1)
    except ApiConnectionError:
        st.error(t("api_unreachable", lang))
    except ApiError as exc:
        st.error(str(exc))
    else:
        st.session_state.voted.add(interaction_id)
        st.rerun()


def _pick_example():
    st.session_state.pending_question = st.session_state.example_pills
    st.session_state.example_pills = None


question = st.chat_input(t("chat_placeholder", lang), max_chars=500)
if not question:
    question = st.session_state.pop("pending_question", None)

if not st.session_state.messages and question is None:
    step_cards([
        ("#34D399", t("step1_title", lang), t("step1_text", lang)),
        ("#22D3EE", t("step2_title", lang), t("step2_text", lang)),
        ("#A78BFA", t("step3_title", lang), t("step3_text", lang)),
    ])
    try:
        base_empty = documents() == []
    except ApiError:
        base_empty = False
    if base_empty:
        with st.container(border=True):
            st.markdown(f"**{t('empty_base_title', lang)}**  \n"
                        f"{t('empty_base_text', lang)}")
            st.page_link("views/documents.py", label=t("go_to_documents", lang),
                         icon=":material/upload_file:")
    else:
        try:
            examples = suggestions(lang)
        except ApiError:
            examples = []  # decorative: no suggestions, the page still works
        if examples:
            st.pills(t("try_asking", lang), examples,
                     key="example_pills", on_change=_pick_example)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            _render_sources(message["sources"])
        if message.get("interaction_id"):
            _render_feedback(message["interaction_id"])

if question:
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
