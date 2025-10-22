from __future__ import annotations

# -*- coding: utf-8 -*-
"""frontend/ui_panels.py
Minimalne, gotowe panele UI do wyświetlania wyników z PRO+ Upgrades.
Użycie (w app.py):
    from frontend.ui_panels import (
        render_auto_clean_report,
        render_training_plan_panel,
        render_data_analysis_panel,
        render_model_plots,
        render_explainability_plus,
        render_explainability_tab,
        render_plots_download,
        render_explainability_pdf_download
    )
    ...
    results = trainer.train_model(...)
    render_auto_clean_report(results.get('cleaning_report', {}))
    render_training_plan_panel(results.get('training_plan', {}))
    render_data_analysis_panel(results.get('data_analysis', {}))
    render_model_plots(results, problem_type=results.get('problem_type', 'classification'))
"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from backend.safe_utils import truthy_df_safe

try:
    from backend.export_plus import make_plots_zip
except Exception:  # pragma: no cover
    make_plots_zip = None  # type: ignore


# --- helpers ------------------------------------------------------------------
def _df_from_dict(d: dict) -> pd.DataFrame:
    try:
        return pd.DataFrame(d)
    except Exception:
        return pd.DataFrame([d]) if isinstance(d, dict) else pd.DataFrame()


def _ensure_plots_dir() -> Path:
    p = Path("artifacts/plots")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _safe_fmt_float(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


# --- panels -------------------------------------------------------------------
def render_auto_clean_report(report: Dict[str, Any]) -> None:
    st.subheader("🧼 Auto Clean Report")
    if not truthy_df_safe(report):
        st.info("Brak raportu czyszczenia.")
        return

    cols = st.columns(2)
    with cols[0]:
        dropped = report.get("dropped_cols", []) or []
        st.markdown("**Usunięte kolumny:** " + (", ".join(map(str, dropped)) if dropped else "—"))

        if "imputed" in report and truthy_df_safe(report["imputed"]):
            st.markdown("**Imputacje (podgląd):**")
            imp = _df_from_dict(report["imputed"]).T.reset_index().rename(columns={"index": "column"})
            st.dataframe(imp, use_container_width=True, hide_index=True)

    with cols[1]:
        if "rarelabeled" in report and truthy_df_safe(report["rarelabeled"]):
            st.markdown("**Rare labels:**")
            rl = {k: v.get("n_rare") for k, v in report["rarelabeled"].items()}
            rl_df = _df_from_dict(rl).T.reset_index().rename(columns={"index": "column", 0: "n_rare"})
            st.dataframe(rl_df, use_container_width=True, hide_index=True)

        if "clipped" in report and truthy_df_safe(report["clipped"]):
            st.markdown("**Clipping:**")
            cl = _df_from_dict(report["clipped"]).T.reset_index().rename(columns={"index": "column"})
            st.dataframe(cl, use_container_width=True, hide_index=True)

    steps = report.get("steps", []) or []
    st.markdown("**Kroki:** " + (", ".join(steps) if steps else "—"))


def render_training_plan_panel(plan: Dict[str, Any]) -> None:
    st.subheader("🧭 Training Plan (AI)")
    if not truthy_df_safe(plan):
        st.info("Brak planu treningu.")
        return

    cols = st.columns(3)
    cols[0].metric("n (rows)", plan.get("n", "—"))
    cols[1].metric("p (cols)", plan.get("p", "—"))
    cols[2].metric("problem", plan.get("problem_type", "—"))

    st.markdown("**CV:**")
    st.json(plan.get("cv", {}) or {})

    st.markdown("**Budżet:**")
    st.json(plan.get("budget", {}) or {})

    st.markdown("**Sampling:**")
    st.json(plan.get("sampling", {}) or {})

    st.markdown("**Metryka główna:** `" + str(plan.get("metrics_primary", "—")) + "`")
    st.markdown("**Metryki dodatkowe:** " + (", ".join(plan.get("metrics_extra", []) or []) or "—"))
    st.markdown("**Algorytmy (kolejność):** " + (", ".join(plan.get("algos_order", []) or []) or "—"))


def render_data_analysis_panel(analysis: Dict[str, Any]) -> None:
    st.subheader("🧪 Data Analysis")
    if not truthy_df_safe(analysis):
        st.info("Brak analizy danych.")
        return

    cols = st.columns(4)
    cols[0].metric("Rows", analysis.get("n_rows", "—"))
    cols[1].metric("Cols", analysis.get("n_cols", "—"))
    cols[2].metric("Y unique", analysis.get("y_unique", "—"))
    cols[3].metric("Target", analysis.get("y_name", "—"))

    st.markdown("**Podejrzane ID kolumny:** " + (", ".join(analysis.get("suspected_id_cols", []) or []) or "—"))
    st.markdown("**Datetime kolumny:** " + (", ".join(analysis.get("datetime_cols", []) or []) or "—"))
    st.markdown("**Tekstowe kolumny (dłuższe):** " + (", ".join(analysis.get("text_cols", []) or []) or "—"))

    with st.expander("Pokaż brakujące wartości (ratio)"):
        miss = analysis.get("missing_ratio", {}) or {}
        if truthy_df_safe(miss):
            df = (
                pd.DataFrame({"column": list(miss.keys()), "missing_ratio": list(miss.values())})
                .sort_values("missing_ratio", ascending=False)
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Brak danych o brakach.")

    with st.expander("Pokaż kardynalność"):
        card = analysis.get("cardinality", {}) or {}
        if truthy_df_safe(card):
            df = (
                pd.DataFrame({"column": list(card.keys()), "unique": list(card.values())})
                .sort_values("unique", ascending=False)
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Brak danych o kardynalności.")


def render_model_plots(results: Dict[str, Any], problem_type: str) -> None:
    st.subheader("📈 Model Plots")
    if not truthy_df_safe(results):
        st.info("Brak wyników do wykresów.")
        return

    y_true = results.get("y_test")
    y_pred = results.get("y_pred")
    y_proba = results.get("y_proba")  # dla klasyfikacji

    if y_true is None or y_pred is None:
        st.info("Brak danych (y_test/y_pred).")
        return

    import matplotlib.pyplot as plt

    _ensure_plots_dir()

    problem_str = str(problem_type or "").lower()

    if problem_str.startswith(("class", "klasyf")):
        # Confusion matrix (tabela)
        try:
            from sklearn.metrics import confusion_matrix

            unique_labels = pd.unique(pd.Series(y_true)).tolist()
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            st.markdown("**Confusion matrix:**")
            cm_df = pd.DataFrame(cm, index=[f"true_{v}" for v in unique_labels], columns=[f"pred_{v}" for v in unique_labels])
            st.dataframe(cm_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Confusion matrix niedostępna: {e}")

        # ROC / PR / Calibration
        if y_proba is not None:
            try:
                from backend.plots_plus import plot_pr_curve, plot_roc_proba, calibration_data

                # ROC
                fpr, tpr, aucv = plot_roc_proba(y_true, y_proba)
                fig = plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={aucv:.3f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC")
                plt.legend()
                st.pyplot(fig, use_container_width=True)
                try:
                    fig.savefig("artifacts/plots/roc.png", bbox_inches="tight")
                except Exception:
                    pass
                plt.close(fig)

                # PR Curve
                rec, prec = plot_pr_curve(y_true, y_proba)
                fig2 = plt.figure()
                plt.plot(rec, prec)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("PR Curve")
                st.pyplot(fig2, use_container_width=True)
                try:
                    fig2.savefig("artifacts/plots/pr_curve.png", bbox_inches="tight")
                except Exception:
                    pass
                plt.close(fig2)

                # Calibration
                pt, pp = calibration_data(y_true, y_proba)
                fig3 = plt.figure()
                plt.plot(pp, pt, marker="o")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("Predicted")
                plt.ylabel("Observed")
                plt.title("Calibration")
                st.pyplot(fig3, use_container_width=True)
                try:
                    fig3.savefig("artifacts/plots/calibration.png", bbox_inches="tight")
                except Exception:
                    pass
                plt.close(fig3)
            except Exception as e:
                st.warning(f"Wykresy ROC/PR/Calibration niedostępne: {e}")
        else:
            st.info("Brak `y_proba` — pomijam wykresy ROC/PR/Calibration.")
    else:
        # Regresja → Residuals
        try:
            from backend.plots_plus import residuals_plot

            xp, res = residuals_plot(y_true, y_pred)
            import matplotlib.pyplot as plt

            fig = plt.figure()
            plt.scatter(xp, res, s=8, alpha=0.7)
            plt.axhline(0, linestyle="--")
            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.title("Residuals")
            st.pyplot(fig, use_container_width=True)
            try:
                fig.savefig("artifacts/plots/residuals.png", bbox_inches="tight")
            except Exception:
                pass
            plt.close(fig)
        except Exception as e:
            st.warning(f"Wykres residuals niedostępny: {e}")


def render_explainability_plus(results: Dict[str, Any], max_ice_samples: int = 50) -> None:
    """Export + PDP/ICE + FI (1D)."""
    st.subheader("🧠 Explainability PRO (PDP/ICE + FI)")
    pipe = None
    try:
        pipe = results.get("pipeline")
    except Exception:
        pipe = None

    if pipe is None:
        st.info("Brak `pipeline` w wynikach — pomiń lub włącz artefakt `pipeline`.")
        return

    feat_names = results.get("feature_names") or []
    if not truthy_df_safe(feat_names):
        st.info("Brak listy cech w `results['feature_names']`.")
        return

    # Feature Importance table (jeśli dostępne)
    fi_df = results.get("feature_importance")
    if fi_df is not None:
        st.markdown("**Feature Importance (model-agnostic lub wbudowane):**")
        try:
            if not isinstance(fi_df, pd.DataFrame):
                fi_df = pd.DataFrame(fi_df)
            if "feature" in fi_df.columns and "importance" in fi_df.columns:
                fi_df = fi_df.sort_values("importance", ascending=False)
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Nie udało się wyświetlić FI: {e}")

    # PDP/ICE controls
    st.markdown("**PDP/ICE**")
    feature = st.selectbox("Wybierz cechę", feat_names, index=0)
    grid_size = st.slider("Rozmiar siatki (PDP)", min_value=10, max_value=100, value=20, step=5)
    ice_n = st.slider("Próbek do ICE", min_value=10, max_value=200, value=min(50, int(max_ice_samples)), step=10)

    y_true = results.get("y_test")
    X_test = results.get("X_test")
    if X_test is None:
        st.info("Brak X_test w `results` — PDP/ICE policzone może być ograniczone.")

    # compute pdp/ice via evaluator (bez cache dla bezpieczeństwa hash/serializacji)
    try:
        data = _cached_pdp_ice(pipe, X_test, feature, grid_size=grid_size, ice_n=ice_n)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])

        import matplotlib.pyplot as plt

        # PDP
        fig = plt.figure()
        plt.plot(data["grid"], data["pdp"], linewidth=2)
        plt.xlabel(feature)
        plt.ylabel("Prediction")
        plt.title("PDP")
        st.pyplot(fig, use_container_width=True)
        try:
            _ensure_plots_dir()
            safe_feat = str(feature).replace("/", "_").replace("\\", "_").replace(" ", "_")
            fig.savefig(f"artifacts/plots/pdp_{safe_feat}.png", bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig)

        # ICE (subset)
        fig2 = plt.figure()
        for arr in data.get("ice", []):
            try:
                plt.plot(data["grid"], np.asarray(arr), alpha=0.4)
            except Exception:
                continue
        plt.xlabel(feature)
        plt.ylabel("Prediction")
        plt.title("ICE (subset)")
        st.pyplot(fig2, use_container_width=True)
        try:
            _ensure_plots_dir()
            safe_feat = str(feature).replace("/", "_").replace("\\", "_").replace(" ", "_")
            fig2.savefig(f"artifacts/plots/ice_{safe_feat}.png", bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig2)
    except Exception as e:
        st.warning(f"PDP/ICE niedostępne: {e}")


def render_explainability_tab(results: Dict[str, Any]) -> None:
    st.subheader("🧠 Explainability PRO – Tab")
    if not truthy_df_safe(results):
        st.info("Brak wyników.")
        return

    feat_names = results.get("feature_names") or []

    # Two columns layout
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Feature Importance (z fallbackami)**")
        try:
            from backend.fi_plus import get_feature_importance

            fi_df = get_feature_importance(results)
            if truthy_df_safe(fi_df):
                st.dataframe(fi_df, use_container_width=True, hide_index=True)
            else:
                st.info("Brak FI do wyświetlenia.")
        except Exception as e:
            st.warning(f"FI niedostępne: {e}")

    with c2:
        st.markdown("**PDP/ICE (1D)**")
        if not truthy_df_safe(feat_names):
            st.info("Brak listy cech.")
        else:
            render_explainability_plus(results)

    with st.expander("PDP 2D (opcjonalne)"):
        if not feat_names or len(feat_names) < 2:
            st.info("Za mało cech do 2D.")
        else:
            f1 = st.selectbox("Cecha X", feat_names, index=0, key="pdp2d_f1")
            f2 = st.selectbox("Cecha Y", feat_names, index=1, key="pdp2d_f2")
            grid = st.slider("Rozmiar siatki", 10, 60, 30, 5, key="pdp2d_grid")

            # Opcjonalna selekcja klasy (gdy classifier) – info only
            try:
                classes = getattr(getattr(results.get("pipeline"), "named_steps", {}).get("model", None), "classes_", None)
                if classes is not None and len(classes) >= 2:
                    _ = st.selectbox(
                        "Klasa (dla predict_proba, info)",
                        list(range(len(classes))),
                        format_func=lambda i: str(classes[i]),
                        index=0,
                    )
            except Exception:
                pass

            try:
                pipe = results.get("pipeline")
                X_test = results.get("X_test")
                if pipe is None or X_test is None:
                    st.info("Brak pipeline/X_test – nie można policzyć PDP 2D.")
                else:
                    data2d = _cached_pdp2d(pipe, X_test, f1, f2, grid_size=grid)
                    if isinstance(data2d, dict) and "error" in data2d:
                        raise RuntimeError(data2d["error"])

                    import matplotlib.pyplot as plt
                    import numpy as np

                    # contour
                    fig = plt.figure()
                    plt.contourf(data2d["x_grid"], data2d["y_grid"], data2d["Z"], levels=20)
                    plt.xlabel(f1)
                    plt.ylabel(f2)
                    plt.title("PDP 2D – contour")
                    st.pyplot(fig, use_container_width=True)
                    try:
                        _ensure_plots_dir()
                        fig.savefig(f"artifacts/plots/pdp2d_contour_{f1}_{f2}.png", bbox_inches="tight")
                    except Exception:
                        pass
                    plt.close(fig)

                    # heatmap with vmin/vmax
                    vmin = st.number_input("vmin", value=float(np.nanmin(data2d["Z"])))
                    vmax = st.number_input("vmax", value=float(np.nanmax(data2d["Z"])))
                    fig_hm = plt.figure()
                    im = plt.imshow(
                        data2d["Z"],
                        origin="lower",
                        aspect="auto",
                        extent=[data2d["x_grid"].min(), data2d["x_grid"].max(), data2d["y_grid"].min(), data2d["y_grid"].max()],
                        vmin=vmin,
                        vmax=vmax,
                    )
                    plt.xlabel(f1)
                    plt.ylabel(f2)
                    plt.title("PDP 2D – heatmap")
                    plt.colorbar(im)
                    st.pyplot(fig_hm, use_container_width=True)
                    try:
                        _ensure_plots_dir()
                        fig_hm.savefig(f"artifacts/plots/pdp2d_{f1}_{f2}.png", bbox_inches="tight")
                    except Exception:
                        pass
                    plt.close(fig_hm)
            except Exception as e:
                st.warning(f"PDP 2D niedostępne: {e}")


def _cached_pdp_ice(pipe, X_test, feature: str, grid_size: int, ice_n: int):
    """Lekki wrapper bez cache (unikamy hash problemów na obiektach sklearn)."""
    try:
        from backend.explain_plus import compute_pdp_ice_data

        return compute_pdp_ice_data(pipe, X_test, feature, grid_size=grid_size, ice_samples=ice_n)
    except Exception as e:
        return {"error": str(e)}


def _cached_pdp2d(pipe, X_test, f1: str, f2: str, grid_size: int):
    """Wrapper bez cache (stabilniej przy złożonych obiektach pipe)."""
    try:
        from backend.explain_plus import compute_pdp2d_data

        return compute_pdp2d_data(pipe, X_test, f1, f2, grid_size=grid_size)
    except Exception as e:
        return {"error": str(e)}


# --- Export & downloads -------------------------------------------------------
# (Pozostawione jako aktywne kontrolki – ten blok wykona się przy imporcie pliku
#  w kontekście Streamlit. Jeśli wolisz bez efektów ubocznych, opakuj w funkcję)
st.markdown("---")
if st.button("📦 Eksportuj wszystkie wykresy (ZIP)"):
    try:
        if make_plots_zip is not None:
            _ensure_plots_dir()
            zip_path = make_plots_zip(plots_dir="artifacts/plots", zip_name="plots_bundle.zip")
            st.success(f"Spakowano wykresy: {zip_path}")
        else:
            st.info("Brak funkcji make_plots_zip – sprawdź backend/export_plus.py")
    except Exception as e:
        st.warning(f"Nie udało się spakować wykresów: {e}")


def render_plots_download() -> None:
    """Renderuje przycisk pobrania spakowanych wykresów (jeśli istnieją)."""
    zip_path = os.path.join("artifacts", "plots_bundle.zip")
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            data = f.read()
        st.download_button(
            "⬇️ Pobierz plots_bundle.zip",
            data=data,
            file_name="plots_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.info("Brak 'plots_bundle.zip' — najpierw użyj przycisku eksportu.")


def render_explainability_pdf_download() -> None:
    """Eksport i pobranie PDF z raportem Explainability (jeśli funkcja dostępna)."""
    try:
        from backend.export_explain_pdf import export_explainability_pdf

        _ensure_plots_dir()
        pdf_path = export_explainability_pdf(out_path="artifacts/Explainability_Report.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                data = f.read()
            st.download_button(
                "⬇️ Pobierz Explainability_Report.pdf",
                data=data,
                file_name="Explainability_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Nie udało się wygenerować PDF.")
    except Exception as e:
        st.warning(f"PDF exporter błąd: {e}")
