from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import Optional, Sequence, Tuple, Any, List

import numpy as np
import pandas as pd


# =====================================================================
#  Bezpieczny import permutation_importance (sklearn)
# =====================================================================

def _safe_perm_import():
    try:
        from sklearn.inspection import permutation_importance as _perm
        return _perm
    except Exception:
        return None


# =====================================================================
#  SAFE PERMUTATION IMPORTANCE
# =====================================================================

def safe_permutation_importance(
    model: Any,
    X_df: pd.DataFrame,
    y: pd.Series | np.ndarray | Sequence,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: Optional[str] = None,
    problem: Optional[str] = None,
    max_samples: Optional[int] = None,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Bezpieczna wersja permutation importance.
    Zwraca DataFrame: ['feature', 'importance'] posortowany malejąco.
    W razie błędu zwraca zera (ten sam kształt).
    - Działa z Pipeline (jeśli wejściem są CECHY po transformacji, podaj je w X_df),
    - 'scoring' możesz podać ręcznie; jeśli None, dobieramy sensowny fallback.
    """

    # Przygotowanie X, y
    if X_df is None or len(X_df) == 0:
        return pd.DataFrame({"feature": [], "importance": []})

    X_df = pd.DataFrame(X_df).copy()
    y_arr = np.asarray(y)

    # Ewentualny sampling dla szybkości
    if max_samples is not None and len(X_df) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_df), size=max_samples, replace=False)
        X_df = X_df.iloc[idx]
        y_arr = y_arr[idx]

    # Wybór scoringu jeśli nie podany
    if scoring is None:
        prob = (problem or "").lower()
        if prob.startswith("reg"):
            scoring = "r2"
        else:
            # klasyfikacja – f1_weighted jest uniwersalny (nie wymaga proba)
            scoring = "f1_weighted"

    perm = _safe_perm_import()
    if perm is None:
        # Brak sklearn.inspection.permutation_importance
        return pd.DataFrame(
            {"feature": list(X_df.columns), "importance": np.zeros(len(X_df.columns))}
        )

    # Pierwsza próba z wybranym scoringiem
    try:
        r = perm(
            model, X_df, y_arr,
            n_repeats=int(n_repeats),
            random_state=int(random_state),
            scoring=scoring,
            n_jobs=n_jobs
        )
        out = pd.DataFrame(
            {"feature": list(X_df.columns), "importance": r.importances_mean}
        ).sort_values("importance", ascending=False, ignore_index=True)
        return out
    except Exception:
        # Druga próba z bardziej „odpornym” balanced_accuracy dla klasyfikacji
        try:
            prob = (problem or "").lower()
            if not prob.startswith("reg"):
                r = perm(
                    model, X_df, y_arr,
                    n_repeats=int(n_repeats),
                    random_state=int(random_state),
                    scoring="balanced_accuracy",
                    n_jobs=n_jobs
                )
                out = pd.DataFrame(
                    {"feature": list(X_df.columns), "importance": r.importances_mean}
                ).sort_values("importance", ascending=False, ignore_index=True)
                return out
        except Exception:
            pass

    # Ostateczny fallback – zwróć zera (ten sam kształt)
    return pd.DataFrame(
        {"feature": list(X_df.columns), "importance": np.zeros(len(X_df.columns))}
    )


# =====================================================================
#  SHAP
# =====================================================================

def _unwrap_pipeline(pipe_or_model: Any) -> Tuple[Optional[Any], Any]:
    """
    Jeśli to Pipeline, zwróć (pre, est) gdzie:
      pre  – wszystkie kroki bez ostatniego,
      est  – ostatni krok (lub 'model' jeśli istnieje).
    W przeciwnym razie (None, pipe_or_model).
    """
    try:
        from sklearn.pipeline import Pipeline  # type: ignore
    except Exception:
        Pipeline = None  # type: ignore

    if Pipeline is None or not isinstance(pipe_or_model, Pipeline):
        return None, pipe_or_model

    try:
        # popularne nazwy
        pre = pipe_or_model.named_steps.get("pre") or pipe_or_model.named_steps.get("transformer")
        est = pipe_or_model.named_steps.get("model", pipe_or_model.steps[-1][1])
        if pre is None and len(pipe_or_model.steps) >= 2:
            # zbuduj pod-pipeline bez ostatniego estymatora
            from sklearn.pipeline import Pipeline as _P
            pre = _P(pipe_or_model.steps[:-1])
        return pre, est
    except Exception:
        return None, pipe_or_model


def _feature_names_from_pre(pre: Any, Xt: np.ndarray) -> List[str]:
    """
    Spróbuj pobrać nazwy cech z transformera; jeśli się nie da — f0..fN-1.
    """
    if pre is None:
        return [f"f{i}" for i in range(getattr(Xt, "shape", (0, 0))[1])]
    try:
        names = pre.get_feature_names_out()
        if isinstance(names, (list, tuple, np.ndarray, pd.Index)):
            return list(map(str, list(names)))
    except Exception:
        pass
    return [f"f{i}" for i in range(getattr(Xt, "shape", (0, 0))[1])]


def _is_tree_model(est: Any) -> bool:
    name = est.__class__.__name__.lower()
    return any(k in name for k in (
        "randomforest", "extratrees", "gradientboost", "histgradient",
        "xgb", "lgbm", "catboost", "decisiontree"
    )) or hasattr(est, "feature_importances_")


def _to_dense(X):
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return np.asarray(X)


def _pick_pred_fn(est, is_classification: Optional[bool]):
    """
    Funkcja predykcji dla KernelExplainer (gdy nie działa automatyczne rozpoznanie).
    - Klasyfikacja: preferuj predict_proba → weź klasę 1, a przy multiclass średnią po klasach.
    - Regresja: predict (wektor).
    """
    if is_classification:
        if hasattr(est, "predict_proba"):
            def f(X):
                proba = est.predict_proba(X)
                proba = np.asarray(proba)
                if proba.ndim == 2:
                    return proba[:, 1]  # binarna: klasa 1
                # multiclass: średnia po klasach (skalarny output dla KernelExplainer)
                return proba.mean(axis=1)
            return f
        # fallback na decision_function lub predict → sprowadź do wektora
        if hasattr(est, "decision_function"):
            def f(X):
                out = est.decision_function(X)
                arr = np.asarray(out)
                return arr if arr.ndim == 1 else arr.mean(axis=1)
            return f
        def f(X):
            out = est.predict(X)
            arr = np.asarray(out)
            return arr if arr.ndim == 1 else arr.mean(axis=1)
        return f
    else:
        def f(X):
            return np.asarray(est.predict(X)).ravel()
        return f


def compute_shap(
    pipe_or_model: Any,
    X_sample: pd.DataFrame,
    *,
    max_background: int = 200,
    max_samples: int = 800,
    random_state: int = 42
) -> Optional[pd.DataFrame]:
    """
    Liczy globalną ważność cech SHAP (mean(|SHAP|)).
    Obsługuje Pipeline: najpierw 'pre' (transformer), potem estymator.
    Zwraca DataFrame ['feature','shap_importance'] posortowany malejąco.
    W razie braku SHAP lub błędu — zwraca None.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    if X_sample is None or len(X_sample) == 0:
        return None

    # Przygotuj próbkę (szybciej/stabilniej)
    X_sample = pd.DataFrame(X_sample).copy()
    rng = np.random.RandomState(random_state)
    if len(X_sample) > max_samples:
        idx = rng.choice(len(X_sample), size=max_samples, replace=False)
        Xs = X_sample.iloc[idx].reset_index(drop=True)
    else:
        Xs = X_sample.reset_index(drop=True)

    # Rozpakuj Pipeline
    pre, est = _unwrap_pipeline(pipe_or_model)

    # Transformuj przez 'pre' (jeśli jest)
    if pre is not None:
        try:
            Xt = pre.transform(Xs)
        except Exception:
            # jak się nie uda — spróbuj na oryginalnych X
            Xt = Xs.values
            feat_names = list(map(str, Xs.columns))
        else:
            Xt = _to_dense(Xt)
            feat_names = _feature_names_from_pre(pre, Xt)
    else:
        Xt = Xs.values
        feat_names = list(map(str, Xs.columns))

    # Background (Kernel/Linear)
    bsize = min(max_background, len(Xt))
    if bsize <= 0:
        return None
    bg_idx = rng.choice(len(Xt), size=bsize, replace=False)
    bg = Xt[bg_idx]

    # Klasyfikacja/regresja (heurystyka)
    is_class = None
    try:
        from sklearn.base import ClassifierMixin  # type: ignore
        is_class = isinstance(est, ClassifierMixin)
    except Exception:
        # heurystyka po atrybutach
        is_class = hasattr(est, "predict_proba") or hasattr(est, "decision_function")

    # Wybierz explainer (priorytet: Tree → auto Explainer → Linear → Kernel)
    try:
        if _is_tree_model(est):
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xt)
        else:
            try:
                # Auto – najczęściej trafne i szybsze niż Kernel
                explainer = shap.Explainer(est, bg)
                ex = explainer(Xt)
                sv = ex  # Explanation
            except Exception:
                try:
                    # Modele liniowe (np. LogisticRegression, LinearRegression)
                    explainer = shap.LinearExplainer(est, bg)
                    sv = explainer.shap_values(Xt)
                except Exception:
                    # Ostatecznie Kernel (wolniejszy) z dopasowaną funkcją predykcji
                    f = _pick_pred_fn(est, is_class)
                    explainer = shap.KernelExplainer(f, bg)
                    sv = explainer.shap_values(Xt)
    except Exception:
        return None

    # Unifikacja formatu wyników
    try:
        # Explanation obiekt (nowe SHAP)
        from shap._explanation import Explanation  # type: ignore
        if isinstance(sv, Explanation):
            vals = np.asarray(sv.values)
            if vals.ndim == 3:  # (n, classes, features)
                vals = np.abs(vals).mean(axis=(0, 1))
            else:               # (n, features)
                vals = np.abs(vals).mean(axis=0)
        else:
            # stare API: lista macierzy lub 2D
            if isinstance(sv, list) and len(sv) > 0:
                mats = []
                for v in sv:
                    arr = np.asarray(getattr(v, "values", v))
                    if arr.ndim == 1:
                        mats.append(np.abs(arr))
                    elif arr.ndim == 2:
                        mats.append(np.abs(arr).mean(axis=0))
                    elif arr.ndim == 3:
                        mats.append(np.abs(arr).mean(axis=(0, 1)))
                vals = np.mean(mats, axis=0)
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3:
                    vals = np.abs(arr).mean(axis=(0, 1))
                elif arr.ndim == 2:
                    vals = np.abs(arr).mean(axis=0)
                else:
                    vals = np.abs(arr).ravel()
    except Exception:
        return None

    vals = np.asarray(vals).ravel()

    # Dopasuj długość do liczby cech (awaryjnie przytnij/dopaduj)
    F = len(feat_names)
    if vals.shape[0] != F:
        if vals.shape[0] > F:
            vals = vals[:F]
        else:
            vals = np.concatenate([vals, np.zeros(F - vals.shape[0])], axis=0)

    out = pd.DataFrame(
        {"feature": feat_names, "shap_importance": vals}
    ).sort_values("shap_importance", ascending=False, ignore_index=True)
    return out


__all__ = ["safe_permutation_importance", "compute_shap"]
