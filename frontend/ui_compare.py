# frontend/ui_compare.py
"""
Compare View (Streamlit) – porównywanie modeli i metryk w TMIV.

Użycie w `app.py`:
    from frontend.ui_compare import render_compare

    # w sekcji Wyniki:
    tr = st.session_state.train_result
    if tr and not tr["leaderboard"].empty:
        render_compare(tr["leaderboard"], explain_service=explain_svc)

Funkcje:
- render_compare(leaderboard, explain_service=None): UI do wyboru metryk i modeli,
  tabela zestawienia oraz radar (matplotlib). Opcjonalnie potrafi wygenerować
  PNG radaru przez ExplainService.plot_radar_leaderboard (artefakt na dysku).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Matplotlib ONLY (bez seaborn)
import matplotlib.pyplot as plt


# --- Konwencje metryk (większe-lepsze / mniejsze-lepsze) ---
BIGGER_BETTER = {
    "accuracy",
    "f1",
    "f1_weighted",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "r2",
    "aps",
    "score",  # wewnętrzna kolumna sortująca (jeśli jest)
}
LOWER_BETTER = {"rmse", "mae", "logloss"}


def render_compare(
    leaderboard: pd.DataFrame,
    *,
    explain_service=None,
    default_top_n: int = 5,
    min_models: int = 2,
) -> None:
    """Główna funkcja widoku porównania."""
    if leaderboard is None or len(leaderboard) == 0:
        st.info("Brak danych do porównania.")
        return

    st.subheader("🔬 Porównanie modeli")

    # --- Wybór metryk i modeli ---
    metric_candidates = _metric_columns(leaderboard)
    if not metric_candidates:
        st.warning("Nie znaleziono kolumn metrycznych w leaderboardzie.")
        st.dataframe(leaderboard)
        return

    with st.expander("Ustawienia porównania", expanded=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            metrics = st.multiselect(
                "Metryki do porównania (min. 2)",
                options=metric_candidates,
                default=_default_metrics(metric_candidates),
                help="Metryki zostaną znormalizowane do 0–1 (większe = lepsze).",
            )
        with c2:
            top_n = int(
                st.slider(
                    "Liczba porównywanych modeli (Top-N wg rank/score)",
                    min_value=max(min_models, 2),
                    max_value=min(10, int(len(leaderboard))),
                    value=min(default_top_n, int(len(leaderboard))),
                    step=1,
                )
            )
        with c3:
            invert_low = st.checkbox("Automatycznie odwróć metryki, gdzie mniejsze=lepsze", value=True)

    if len(metrics) < 2:
        st.warning("Wybierz co najmniej dwie metryki.")
        return

    # --- Wybor najlepszych modeli (rank->score->primary) ---
    lb = leaderboard.copy()
    for c in ["rank", "score", "primary_metric"]:
        if c not in lb.columns:
            lb[c] = np.nan
    lb = lb.sort_values(
        by=["rank", "score", "primary_metric"], ascending=[True, False, False], na_position="last"
    ).reset_index(drop=True)
    lb_top = lb.head(top_n).copy()

    # --- Tabela surowa + download ---
    st.markdown("**Tabela porównawcza (surowe wartości)**")
    st.dataframe(lb_top[["model"] + metrics + ["rank"]].reset_index(drop=True))
    _download_csv_button(lb_top[["model"] + metrics + ["rank"]], label="Pobierz CSV porównania")

    # --- Normalizacja 0–1 (większe=lepsze) ---
    norm_df = _normalize_metrics(lb_top, metrics, auto_invert=invert_low)

    # --- Radar: inline (matplotlib) ---
    st.markdown("**Radar (0–1, większe lepsze)**")
    fig = _radar_chart(norm_df, metrics, label_col="model")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    # --- (Opcjonalnie) Radar przez ExplainService → artefakt PNG ---
    if explain_service is not None:
        if st.button("Wygeneruj PNG radaru (artefakt przez ExplainService)"):
            try:
                png = explain_service.plot_radar_leaderboard(lb_top, metrics=metrics, name="radar_compare")
                st.success("Zapisano wykres radaru jako PNG.")
                st.code(png)
                st.image(png, caption="Radar (ExplainService PNG)")
            except Exception as e:
                st.error(f"Nie udało się wygenerować PNG przez ExplainService: {e}")


# =========================
# Helpers
# =========================

def _metric_columns(df: pd.DataFrame) -> List[str]:
    """Zwróć listę kolumn kandydatów na metryki numeryczne."""
    skip = {"model", "primary_metric", "rank"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    # porządek: najczęściej spotykane metryki z przodu, reszta alfabetycznie
    common = [m for m in [
        "accuracy", "f1_weighted", "f1", "roc_auc", "roc_auc_ovr", "r2", "rmse", "mae", "logloss", "aps"
    ] if m in cols]
    rest = sorted([c for c in cols if c not in common])
    return common + rest


def _default_metrics(candidates: Sequence[str]) -> List[str]:
    """Proponuj sensowny zestaw startowy."""
    preferred = ["r2", "rmse", "mae", "accuracy", "f1_weighted", "roc_auc", "aps", "logloss"]
    out = [m for m in preferred if m in candidates]
    if len(out) >= 2:
        return out[:4]
    # fallback: pierwsze dwie z listy
    return list(candidates[:2])


def _normalize_metrics(df: pd.DataFrame, metrics: Sequence[str], *, auto_invert: bool = True) -> pd.DataFrame:
    """
    Normalizuj metryki do [0,1] z konwencją 'większe=lepsze'.
    Dla metryk z LOWER_BETTER odwracamy skalę (1 - minmax).
    """
    out = pd.DataFrame()
    out["model"] = df["model"].astype(str).values
    for m in metrics:
        s = pd.to_numeric(df[m], errors="coerce")
        if s.notna().sum() == 0:
            out[m] = np.nan
            continue
        # kierunek
        invert = auto_invert and (m in LOWER_BETTER)
        if invert:
            s_eff = -s  # mniejsze=lepsze → odwróć, by większe=lepsze
        else:
            s_eff = s
        # min-max
        mn, mx = float(np.nanmin(s_eff)), float(np.nanmax(s_eff))
        if mx - mn < 1e-12:
            out[m] = 1.0  # identyczne wartości → pełne 1.0, by nie generować NaN
        else:
            out[m] = (s_eff - mn) / (mx - mn)
    return out


def _radar_chart(norm_df: pd.DataFrame, metrics: Sequence[str], *, label_col: str = "model"):
    """
    Rysuj radar (jeden wykres) porównujący modele w przestrzeni metryk 0–1.
    Zasady:
    - matplotlib only
    - jeden wykres = jedna figura
    - brak wymuszania kolorów (użyj domyślnych)
    """
    # Przygotuj dane
    labels = list(map(str, metrics))
    K = len(labels)
    if K < 2:
        K = 2
        labels = list(labels) + ["(dummy)"]
        norm_df["(dummy)"] = 0.5

    angles = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist()
    angles += angles[:1]  # zamknij okrąg

    fig = plt.figure(figsize=(6.8, 6.8))
    ax = plt.subplot(111, polar=True)

    # rysuj każdy model
    for _, row in norm_df.iterrows():
        vals = [float(row.get(m, np.nan)) for m in metrics]
        if len(vals) < K:
            vals += [0.5] * (K - len(vals))
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1)
        ax.fill(angles, vals, alpha=0.08)
        # podpis w legendzie
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1)

    # legenda
    ax.legend(norm_df[label_col].tolist(), loc="upper right", bbox_to_anchor=(1.18, 1.08))
    ax.set_title("Porównanie modeli – radar (0–1, większe lepsze)")
    fig.tight_layout()
    return fig


def _download_csv_button(df: pd.DataFrame, *, label: str = "Pobierz CSV") -> None:
    try:
        import io

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            label=label,
            data=buf.getvalue().encode("utf-8"),
            file_name="tmiv_compare.csv",
            mime="text/csv",
        )
    except Exception:
        pass
