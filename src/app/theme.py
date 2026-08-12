"""Dark Research Lab identity: injected CSS and shared HTML fragments.

Palette (see docs/superpowers/specs/2026-08-11-dark-lab-redesign-design.md):
bg #0B1120 · surface #0F172A · border #1E293B · text #E2E8F0 · muted #64748B
emerald #34D399 (primary) · cyan #22D3EE · violet #A78BFA · red #F87171
"""
import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');

html, body,
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {
    font-family: 'Inter', 'Source Sans Pro', sans-serif;
}

/* chat bubbles: emerald spine on assistant, plain surface on user */
[data-testid="stChatMessage"] {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 2px 12px 12px 12px;
    padding: 0.9rem 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #34D399;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #14233B;
    border-radius: 12px 12px 2px 12px;
}

/* example-question pills in cyan */
[data-testid="stPills"] button {
    border: 1px solid #164E63 !important;
    color: #22D3EE !important;
    border-radius: 999px !important;
    background: transparent !important;
}
[data-testid="stPills"] button:hover {
    border-color: #22D3EE !important;
    background: #0E2A38 !important;
}

/* secondary buttons: quiet surface look */
button[kind="secondary"] {
    background: #0F172A;
    border: 1px solid #1E293B;
}

/* uploader as a dashed drop zone */
[data-testid="stFileUploaderDropzone"] {
    background: #0F172A;
    border: 2px dashed #164E63;
    border-radius: 12px;
}

/* bordered containers (document cards) */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px;
}

[data-testid="stExpander"] details {
    border: 1px solid #1E293B;
    border-radius: 10px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #1E293B;
}

h1, h2, h3 { font-weight: 800 !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div style="margin-bottom:0.6rem;">
          <div style="font-size:1.7rem; font-weight:800; color:#F1F5F9;">{title}</div>
          <div style="font-size:0.9rem; color:#64748B;">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def step_cards(steps: list[tuple[str, str, str]]) -> None:
    """steps: list of (accent_hex, title, text)."""
    cells = "".join(
        f"""<div style="flex:1; background:#0F172A; border:1px solid #1E293B;
                    border-radius:10px; padding:0.7rem 0.8rem;">
              <div style="color:{accent}; font-size:0.72rem; font-weight:700;
                          letter-spacing:0.06em;">{title}</div>
              <div style="color:#94A3B8; font-size:0.8rem;">{text}</div>
            </div>"""
        for accent, title, text in steps
    )
    st.markdown(
        f'<div style="display:flex; gap:0.6rem; margin:0.4rem 0 1rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


def sidebar_status(documents: int, chunks: int, label: str) -> None:
    st.sidebar.markdown(
        f"""<div style="background:#0D2622; border:1px solid #14532D; border-radius:8px;
                    padding:0.55rem 0.7rem; font-size:0.75rem; color:#86EFAC;">
          <span style="font-family:'JetBrains Mono',monospace; font-weight:500;">
            {documents} · {chunks}</span> {label}
        </div>""",
        unsafe_allow_html=True,
    )
