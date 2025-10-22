from __future__ import annotations
# -*- coding: utf-8 -*-

from backend.safe_utils import truthy_df_safe

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Akceptowane nazwy metryk (case-insensitive)
_ALLOWED_METRICS = {
    "rmse", "mae", "r2", "mape", "medae", "rmsle",
    "accuracy", "f1", "f1_weighted", "auc", "roc_auc_ovr", "average_precision"
}
# Metryki, gdzie "mniej = lepiej"
_LOWER_IS_BETTER = {"rmse", "mae", "mape", "medae", "rmsle"}


def _to_df(obj: Any) -> pd.DataFrame:
    """Bezpieczne rzutowanie na DataFrame z kilku typów wejścia."""
    try:
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            # Jeśli to rekord jednego modelu (np. zawiera 'model'/'name'), zbuduj DF z pojedynczym wierszem
            return pd.DataFrame([obj]) if ('model' in obj or 'name' in obj) else pd.DataFrame(obj)
    except Exception:
        pass
    return pd.DataFrame()


def _pick_leaderboard(results: Dict[str, Any]) -> pd.DataFrame:
    """Wybierz strukturę leaderboardu z kilku możliwych kluczy i zwróć jako DataFrame."""
    candidates = (
        results.get("leaderboard")
        or results.get("models")
        or results.get("candidates")
        or results.get("model_results")
    )
    df = _to_df(candidates)
    # Normalizacja kolumn na str (bez modyfikowania oryginalnych nazw/metryk)
    try:
        df.columns = [str(c) for c in df.columns]
    except Exception:
        pass
    return df


def _find_primary_metric(results: Dict[str, Any], df: pd.DataFrame) -> Optional[str]:
    """
    Znajdź kolumnę metryki do sortowania:
      - najpierw training_plan.metrics_primary,
      - aliasy,
      - fallback: f1 / accuracy / auc / r2 / rmse / mae (pierwsza dostępna).
    Dopasowanie case-insensitive do kolumn DataFrame.
    """
    primary = None
    tp = (results.get("training_plan") or {})
    cand = tp.get("metrics_primary")

    aliases = {"f1_weighted": "f1", "roc_auc_ovr": "auc"}

    def _match_col(name: str | None) -> Optional[str]:
        if not name:
            return None
        name_l = str(name).lower()
        for c in df.columns:
            if str(c).lower() == name_l:
                return c
        return None

    primary = _match_col(cand) or _match_col(aliases.get(str(cand).lower()))
    if primary:
        return primary

    for fallback in ["f1", "accuracy", "auc", "r2", "rmse", "mae"]:
        hit = _match_col(fallback)
        if hit:
            return hit
    return None


def _metric_columns(df: pd.DataFrame) -> List[str]:
    """Zwróć listę kolumn będących metrykami (po dopasowaniu case-insensitive do akceptowanych nazw)."""
    cols = []
    for c in df.columns:
        if str(c).lower() in _ALLOWED_METRICS:
            cols.append(c)
    return cols


def _name_columns(df: pd.DataFrame) -> List[str]:
    """Preferowana kolejność kolumn opisowych."""
    out = [c for c in ["name", "model", "algorithm"] if c in df.columns]
    return out


def _sort_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Posortuj DataFrame według metryki, respektując kierunek (lower-is-better dla błędów)."""
    asc = str(metric).lower() in _LOWER_IS_BETTER
    try:
        # postaraj się posortować po wartości numerycznej
        s = pd.to_numeric(df[metric], errors="coerce")
        return df.assign(_m=s).sort_values("_m", ascending=asc).drop(columns=["_m"])
    except Exception:
        return df.sort_values(metric, ascending=asc)


def _safe_st_dataframe(df: pd.DataFrame) -> None:
    """Wyświetl dataframe, zachowując zgodność z różnymi wersjami Streamlit."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)  # nowsze wersje
    except TypeError:
        st.dataframe(df, use_container_width=True)  # starsze wersje


def render_compare_models(results: dict) -> None:
    st.subheader("🏆 Compare Models (Top-N)")

    if not truthy_df_safe(results):
        st.info("Brak wyników do porównania.")
        return

    df = _pick_leaderboard(results)
    if df.empty:
        st.info(
            "Brak tabeli kandydatów/leaderboardu w results. "
            "Jeśli używasz własnej struktury, dodaj ją do results['leaderboard']."
        )
        return

    # Kolumny do pokazania: opisowe + metryki
    metric_cols = _metric_columns(df)
    name_cols = _name_columns(df)
    display_cols = (name_cols + [c for c in metric_cols if c not in name_cols]) or df.columns.tolist()

    # Sortowanie wg metryki primary (lub fallback)
    primary = _find_primary_metric(results, df)
    if primary:
        df = _sort_by_metric(df, primary)

    _safe_st_dataframe(df[display_cols])


def render_compare_charts(results: dict, top_n: int = 5) -> None:
    st.subheader("📊 Compare Charts")
    import matplotlib.pyplot as plt

    df = _pick_leaderboard(results)
    if df.empty:
        st.info("Brak leaderboardu — nie można narysować porównań.")
        return

    primary = _find_primary_metric(results, df)
    if primary:
        df = _sort_by_metric(df, primary)

    top = df.head(max(1, int(top_n))).copy()
    metric_cols = _metric_columns(top)

    # Wykres 1: poziome słupki dla metryki primary (najbardziej czytelne porównanie)
    if primary:
        fig1, ax1 = plt.subplots()
        # wartości numeryczne (bezpiecznie)
        vals = pd.to_numeric(top[primary], errors="coerce")
        labels = (
            top.get("name", top.get("model", top.get("algorithm", top.index)))
            .astype(str)
            .tolist()
        )
        ax1.barh(labels, vals)
        ax1.invert_yaxis()  # najlepszy na górze
        asc = str(primary).lower() in _LOWER_IS_BETTER
        ax1.set_xlabel(primary)
        ax1.set_title(f"Top-{len(top)} według {primary} ({'↓ lepiej' if asc else '↑ lepiej'})")
        st.pyplot(fig1, use_container_width=True)
    else:
        st.info("Brak jednoznacznej metryki primary — pomijam pierwszy wykres.")

    # Wykres 2: porównanie wielometryczne (znormalizowane do 0–1, tak by ↑ oznaczało lepiej)
    if truthy_df_safe(metric_cols):
        # Przygotuj macierz znormalizowanych wyników (↑ lepiej)
        norm = pd.DataFrame(index=top.index)
        for m in metric_cols:
            v = pd.to_numeric(top[m], errors="coerce")
            if v.notna().sum() == 0:
                continue
            # normalizacja min-max; odwracamy skalę dla metryk "mniej=lepiej"
            if str(m).lower() in _LOWER_IS_BETTER:
                # mniejsze → lepiej
                v_norm = (v.max() - v) / (v.max() - v.min() + 1e-12)
            else:
                v_norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
            norm[m] = v_norm.clip(0, 1)

        if not norm.empty:
            fig2, ax2 = plt.subplots()
            labels = (
                top.get("name", top.get("model", top.get("algorithm", top.index)))
                .astype(str)
                .tolist()
            )
            x = np.arange(len(norm))  # modele
            width = max(0.8 / max(1, len(norm.columns)), 0.12)  # szerokość słupka

            for i, m in enumerate(norm.columns):
                ax2.bar(x + i * width, norm[m].fillna(0.0).values, width=width, label=m)

            ax2.set_xticks(x + (len(norm.columns) - 1) * width / 2)
            ax2.set_xticklabels(labels, rotation=30, ha="right")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Znormalizowany wynik (↑ lepiej)")
            ax2.set_title("Porównanie wielu metryk (znormalizowane 0–1)")
            ax2.legend(loc="best", fontsize="small")
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("Brak metryk liczbowych do porównania wielometrycznego.")
    else:
        st.info("Brak wspólnych metryk do wykresu porównawczego.")

    # Wykres 3: ROC overlay (jeśli dostępne probabilistyki)
    y_test = results.get("y_test")
    probas = (results.get("compare") or {}).get("probas")
    if y_test is not None and isinstance(probas, dict) and probas:
        try:
            from sklearn.metrics import roc_curve, auc
            fig3, ax3 = plt.subplots()
            plotted = 0
            for name, y_proba in list(probas.items())[: top_n]:
                try:
                    y_true = pd.Series(y_test).astype(int).values
                    y_p = np.asarray(y_proba).astype(float)
                    # Obsłuż kształty (binarka: p(klasa=1))
                    if y_p.ndim == 2 and y_p.shape[1] == 2:
                        y_p = y_p[:, 1]
                    fpr, tpr, _ = roc_curve(y_true, y_p)
                    ax3.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
                    plotted += 1
                except Exception:
                    continue
            if plotted:
                ax3.plot([0, 1], [0, 1], linestyle="--")
                ax3.set_xlabel("FPR")
                ax3.set_ylabel("TPR")
                ax3.set_title("ROC overlay (Top-N)")
                ax3.legend()
                st.pyplot(fig3, use_container_width=True)
            else:
                st.info("Nie udało się narysować ROC — sprawdź format y_test/prob.")
        except Exception:
            st.info("Brak sklearn.metrics. Pomijam overlay ROC.")
    else:
        st.info("Brak słownika proby per model w results['compare']['probas'] — pomijam ROC overlay.")
