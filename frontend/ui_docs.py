# frontend/ui_docs.py
"""
Docs Page (Streamlit) – przegląd dokumentacji TMIV w UI.

Funkcje:
- render_docs(): strona "📚 Dokumentacja" z wyborem pliku i podglądem Markdown.
- Wspiera: docs/README.md, docs/ARCHITECTURE.md, docs/UX_GUIDE.md, docs/API_INTERNALS.md
- Wyszukiwarka (prostą frazą) + przyciski pobierania.

Użycie:
    from frontend.ui_docs import render_docs
    render_docs()
"""

from __future__ import annotations

from pathlib import Path
import io
import streamlit as st

DOCS_MAP = {
    "README": Path("docs/README.md"),
    "ARCHITECTURE": Path("docs/ARCHITECTURE.md"),
    "UX GUIDE": Path("docs/UX_GUIDE.md"),
    "API INTERNALS": Path("docs/API_INTERNALS.md"),
}

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def _filter_text(text: str, query: str) -> str:
    """Prosta filtracja – jeśli jest zapytanie, pokaż tylko dopasowane sekcje (nagłówek + kilka linii)."""
    if not query:
        return text
    lines = text.splitlines()
    q = query.lower().strip()
    if not q:
        return text

    out: list[str] = []
    window = 4  # ile linii po dopasowaniu
    i = 0
    while i < len(lines):
        line = lines[i]
        if q in line.lower():
            # dodaj kontekst (do najbliższego nagłówka wstecz + kilka linii wprzód)
            start = max(0, i - 1)
            # znajdź początek sekcji (nagłówek markdown)
            while start > 0 and not lines[start].startswith("#"):
                start -= 1
            end = min(len(lines), i + 1 + window)
            if out and out[-1] != "":
                out.append("")
            out.extend(lines[start:end])
            out.append("...")
            i = end
        else:
            i += 1
    return "\n".join(out) if out else f"> Brak wyników dla: `{query}`"

def render_docs() -> None:
    st.header("📚 Dokumentacja TMIV")

    col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    with col1:
        choice = st.selectbox("Wybierz dokument", options=list(DOCS_MAP.keys()), index=0)
    with col2:
        query = st.text_input("Szukaj w treści (fraza, opcjonalnie)", value="")
    with col3:
        raw_toggle = st.toggle("Pokaż surowy Markdown", value=False)

    path = DOCS_MAP.get(choice)
    content = _read_text(path) if path else ""
    if not content:
        st.warning(f"Nie znaleziono pliku dokumentacji: {path}")
        return

    filtered = _filter_text(content, query)

    st.caption(f"Plik: {path}")
    if raw_toggle:
        st.code(filtered, language="markdown")
    else:
        st.markdown(filtered, unsafe_allow_html=False)

    # Akcje: pobierz / otwórz plik lokalnie
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        buf = io.BytesIO(content.encode("utf-8"))
        st.download_button(
            label="💾 Pobierz plik",
            data=buf,
            file_name=path.name,
            mime="text/markdown",
        )
    with btn_col2:
        try:
            st.markdown(f"[Otwórz w przeglądarce]({path.resolve().as_uri()})")
        except Exception:
            pass
