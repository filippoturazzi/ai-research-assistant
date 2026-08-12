# Dark Research Lab — UI redesign

Date: 2026-08-11
Status: approved (user picked direction B in the visual companion, then the
icon-only action variant; mockups in `.superpowers/brainstorm/`)

## Goal

Replace the default-looking Streamlit UI with a distinctive dark identity and
guided UX: onboarding steps, example questions, instructive empty states, and
vector icons (no emojis) for all actions.

## Visual identity

- Palette: background `#0B1120`, surface `#0F172A`, borders `#1E293B`,
  text `#E2E8F0`/`#94A3B8`, accents emerald `#34D399` (primary), cyan
  `#22D3EE` (links/citations/example chips), violet `#A78BFA`, red `#F87171`
  (destructive).
- Icons: Streamlit Material Symbols on native widgets (`:material/delete:`
  icon-only remove buttons, `st.feedback("thumbs")` for 👍/👎); inline SVG
  (Lucide-style, stroke 2, currentColor) inside custom HTML blocks.

## Structure

- `Home.py` becomes a router: `st.set_page_config` + shared sidebar (language
  selector, "N documentos · M trechos" indicator) + `st.navigation` over
  `st.Page` entries with Material icons. Deploy entrypoint unchanged.
- Page bodies move from `pages/` (auto-nav, no icon support) to `views/`:
  `views/chat.py`, `views/documents.py`, `views/metrics.py`. `pages/` dir is
  removed to avoid double navigation.
- `.streamlit/config.toml`: dark base theme with the palette above.
- `src/app/theme.py`: `inject_css()` (called once by the router) with the
  custom styling; small helpers for icon HTML used by views.

## UX per page

Chat (Home): header "Converse com os seus documentos" + subtitle; when the
conversation is empty show the 3 onboarding step cards (Coloque → Pergunte →
Avalie) and clickable example-question chips that submit directly; sources
expander and thumbs feedback as icon-only controls; empty knowledge base shows
a "comece aqui" card pointing to Documentos.

Documentos: header "Sua base de conhecimento" + copy "limpe os documentos e
coloque os seus"; uploader; each document as a bordered card row with an
icon-only remove button; bulk section "trocar a base inteira" (clear all /
restore defaults, existing two-step confirm); guided empty state.

Métricas: same theme, Material icons on section headers; layout unchanged.

Translations: new/adjusted en/pt strings for all copy above.

## Out of scope

Animations, replacing the native uploader dropzone look beyond CSS, React
front-end. Backend/API untouched.

## Verification

Full pytest suite (no page tests exist; backend untouched), then run the app
locally and inspect both pages in the browser before pushing (push =
Streamlit Cloud deploy).
