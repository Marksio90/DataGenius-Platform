from __future__ import annotations
# -*- coding: utf-8 -*-

# backend/__init__.py

from backend.safe_utils import truthy_df_safe

# ------------------------------------------------------------
# Importy podstawowe: preferuj relatywne, fallback na absolutne
# ------------------------------------------------------------
try:
    # Gdy backend jest używany jako pakiet
    from .eda_integration import EDAAnalyzer, apply_ai_dataprep  # type: ignore
except Exception:  # pragma: no cover
    # Gdy uruchamiasz z root projektu
    from backend.eda_integration import EDAAnalyzer, apply_ai_dataprep  # type: ignore

# Streamlit jest opcjonalny – funkcje renderujące działają bez niego
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore


# ------------------------------------------------------------
# Kompatybilność wsteczna: DataProcessor => EDAAnalyzer
# ------------------------------------------------------------
DataProcessor = EDAAnalyzer  # alias 1:1 (zachowuje poprzedni interfejs)


def get_data_processor() -> DataProcessor:
    """
    Zwraca instancję procesora danych (kompatybilność wsteczna).
    W nowszych wersjach używaj bezpośrednio EDAAnalyzer().
    """
    return DataProcessor()


# ------------------------------------------------------------
# Minimalistyczne funkcje renderujące (bezpieczne placeholdery)
# ------------------------------------------------------------
def render_eda_dashboard(df, *, analyzer: EDAAnalyzer | None = None):
    """
    Prosty dashboard EDA:
      - raport EDA (basic info, braki, korelacje, outliery, rekomendacje),
      - wizualizacje (missing bar, corr heatmap, distributions).
    Działa bez Streamlit (wtedy zwraca słownik wyników).
    """
    analyzer = analyzer or EDAAnalyzer()
    report = analyzer.generate_comprehensive_eda_report(df)
    visuals = analyzer.create_eda_visualizations(df)

    # jeśli jest Streamlit – lekko wyrenderuj
    if st is not None:
        try:
            st.subheader("📊 Podstawowy raport EDA")
            basic = report.get("basic_info", {}) or {}
            shape = basic.get("shape", (None, None))
            mem_mb = basic.get("memory_usage_mb", 0.0) or 0.0
            dups = basic.get("duplicate_rows", 0) or 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Wiersze × Kolumny", f"{shape[0]} × {shape[1]}")
            with c2:
                st.metric("Pamięć (MB)", f"{float(mem_mb):.2f}")
            with c3:
                st.metric("Duplikaty", int(dups))

            # brakujące
            if "missing_data" in visuals and truthy_df_safe(visuals["missing_data"]):
                st.plotly_chart(visuals["missing_data"], use_container_width=True)
            # korelacje
            if "correlation_matrix" in visuals and truthy_df_safe(visuals["correlation_matrix"]):
                st.plotly_chart(visuals["correlation_matrix"], use_container_width=True)
            # rozkłady
            if "distributions" in visuals and truthy_df_safe(visuals["distributions"]):
                st.plotly_chart(visuals["distributions"], use_container_width=True)

            # jakościowe problemy + rekomendacje
            issues = report.get("data_quality_issues", []) or []
            st.markdown("### 🧹 Wykryte problemy jakości danych")
            if issues:
                for it in issues[:10]:
                    st.write(f"- **{it.get('type','?')}**: {it.get('description','')}")
            else:
                st.caption("Brak istotnych problemów.")

            recs = report.get("recommendations", []) or []
            st.markdown("### 💡 Rekomendacje")
            if recs:
                for r in recs[:10]:
                    st.write(f"- {r}")
            else:
                st.caption("Brak rekomendacji.")
        except Exception:
            # Nie przerywaj działania, nawet jeśli UI nie zrenderuje się poprawnie
            pass

    # Zwróć dane (przydaje się w testach / API)
    return {"report": report, "visuals": visuals}


def render_advanced_visualizations(df, *, analyzer: EDAAnalyzer | None = None):
    """
    Rozszerzone wizualizacje – w tej wersji deleguje do create_eda_visualizations().
    W razie potrzeby rozbuduj (np. par-plots, PDP/ICE).
    """
    analyzer = analyzer or EDAAnalyzer()
    visuals = analyzer.create_eda_visualizations(df)

    if st is not None:
        try:
            st.subheader("📈 Zaawansowane wizualizacje")
            for name, fig in visuals.items():
                if truthy_df_safe(fig):
                    st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    return visuals


def render_feature_engineering_section(df, target: str | None = None, *, analyzer: EDAAnalyzer | None = None):
    """
    Sekcja „feature engineering”: pokazuje efekt lekkiego data-prepu.
    """
    analyzer = analyzer or EDAAnalyzer()
    # używamy helpera z eda_integration.py
    df_prep, steps = apply_ai_dataprep(df, target or "__no_target__", plan={})

    if st is not None:
        try:
            st.subheader("🛠️ Feature Engineering (AI plan – lekka wersja)")
            st.write(f"Zmieniono kształt: {df.shape} → {df_prep.shape}")
            with st.expander("Kroki przetwarzania", expanded=False):
                for s in steps:
                    st.write(f"- **{s.get('name','step')}** — {s.get('detail','')}")
            # szybkie podglądy
            st.markdown("**Podgląd danych (po):**")
            st.dataframe(df_prep.head(20))
        except Exception:
            pass

    return {"prepared_df": df_prep, "log": steps}


def render_data_quality_report(df, *, analyzer: EDAAnalyzer | None = None):
    """
    Skrócony raport jakości danych.
    """
    analyzer = analyzer or EDAAnalyzer()
    report = analyzer.generate_comprehensive_eda_report(df)
    issues = report.get("data_quality_issues", []) or []

    if st is not None:
        try:
            st.subheader("🧪 Raport jakości danych")
            if not issues:
                st.success("Brak istotnych problemów jakościowych.")
            else:
                for i in issues:
                    st.write(f"- **{i.get('type','')}**: {i.get('description','')}")
                    if i.get("recommendation"):
                        st.caption(f"Rekomendacja: {i['recommendation']}")
        except Exception:
            pass

    return issues


def render_dataset_comparison(df_left, df_right, *, title_left="Zbiór A", title_right="Zbiór B"):
    """
    Bardzo prosty porównywacz dwóch zbiorów (shape, kolumny wspólne/unikalne).
    """
    left_cols = set(map(str, df_left.columns))
    right_cols = set(map(str, df_right.columns))
    common = sorted(left_cols & right_cols)
    only_left = sorted(left_cols - right_cols)
    only_right = sorted(right_cols - left_cols)

    result = {
        "shape_left": tuple(df_left.shape),
        "shape_right": tuple(df_right.shape),
        "common_columns": common,
        "only_left": only_left,
        "only_right": only_right,
    }

    if st is not None:
        try:
            st.subheader("🔀 Porównanie zbiorów")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(title_left, f"{df_left.shape[0]} × {df_left.shape[1]}")
            with c2:
                st.metric(title_right, f"{df_right.shape[0]} × {df_right.shape[1]}")
            with c3:
                st.metric("Wspólne kolumny", len(common))
            with st.expander("Wspólne kolumny", expanded=False):
                st.write(common[:200])
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Unikalne w {title_left}**")
                st.write(only_left[:200])
            with colB:
                st.markdown(f"**Unikalne w {title_right}**")
                st.write(only_right[:200])
        except Exception:
            pass

    return result


def render_interactive_data_explorer(df, *, height: int = 420):
    """
    Bardzo lekki „eksplorator danych” – po prostu interaktywny podgląd danych.
    """
    if st is not None:
        try:
            st.subheader("🔎 Interaktywny podgląd danych")
            st.dataframe(df, height=height)
        except Exception:
            pass
    # zawsze coś zwracamy (np. do testów)
    return {"preview_rows": min(len(df), 20), "columns": list(map(str, df.columns))}


# ------------------------------------------------------------
# Publiczny interfejs modułu
# ------------------------------------------------------------
__all__ = [
    # Nowe API
    "EDAAnalyzer",
    "apply_ai_dataprep",

    # Kompatybilność wsteczna
    "DataProcessor",
    "get_data_processor",

    # Funkcje renderujące (bezpieczne placeholdery)
    "render_eda_dashboard",
    "render_advanced_visualizations",
    "render_feature_engineering_section",
    "render_data_quality_report",
    "render_dataset_comparison",
    "render_interactive_data_explorer",
]
