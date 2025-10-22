from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd


# =========================
# Helpers
# =========================

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return int(getattr(x, "shape", [0])[0])

def _is_classifier(model) -> bool:
    try:
        from sklearn.base import ClassifierMixin
        return isinstance(model, ClassifierMixin)
    except Exception:
        # fallback: heurystyka po atrybutach
        return hasattr(model, "predict_proba") or hasattr(model, "decision_function")

def _split_pipeline(obj) -> Tuple[Optional[object], object]:
    """
    Zwraca (preprocessor_or_None, estimator).
    Obsługuje popularne nazwy kroków: 'pre'/'transformer' oraz 'model' (ostatni krok fallback).
    """
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(obj, Pipeline):
            pre = obj.named_steps.get("pre") or obj.named_steps.get("transformer")
            est = obj.named_steps.get("model", obj.steps[-1][1])
            if pre is None and len(obj.steps) >= 2:
                # zbuduj pod-pipeline bez estymatora jako preprocesor
                from sklearn.pipeline import Pipeline as _P
                pre = _P(obj.steps[:-1])
            return pre, est
    except Exception:
        pass
    return None, obj

def _transform_X_and_names(pre, X: pd.DataFrame) -> Tuple[object, List[str]]:
    """
    Zwraca (Xt, feature_names) po przejściu przez preprocesor.
    Jeśli brak pre → (X, X.columns) (przyjmuje X jako DataFrame).
    """
    if pre is None:
        # jeśli X nie jest DataFrame (np. ndarray), dopasuj nazwy robocze
        if isinstance(X, pd.DataFrame):
            return X, list(map(str, X.columns))
        n = getattr(X, "shape", (0, 0))[1]
        return X, [f"f{i}" for i in range(n)]
    Xt = pre.transform(X)
    # nazwy po transformacjach
    try:
        names = list(map(str, pre.get_feature_names_out()))
    except Exception:
        # jeżeli Xt ma kolumny (np. DataFrame z ColumnTransformer)
        try:
            names = list(map(str, Xt.columns))  # type: ignore[attr-defined]
        except Exception:
            n = getattr(Xt, "shape", (0, 0))[1]
            names = [f"f{i}" for i in range(n)]
    return Xt, names

def _to_dense(X):
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return np.asarray(X)

def _mean_abs_over_classes(shap_values) -> np.ndarray:
    """
    Normalizuje różne formaty SHAP:
      - list[class] -> średnia z mean(|sv|) po klasach,
      - Explanation -> Explanation.values,
      - ndarray -> mean(|sv|, axis=0)
    """
    try:
        import shap  # noqa: F401
    except Exception:
        pass

    sv = shap_values
    # list: [n_classes * (n_samples, n_features)]
    if isinstance(sv, list) and len(sv) > 0:
        mats = []
        for item in sv:
            arr = getattr(item, "values", item)
            arr = np.asarray(arr)
            # spodziewamy się (n_samples, n_features) lub (n_features,)
            if arr.ndim == 1:
                mats.append(np.abs(arr))
            else:
                mats.append(np.abs(arr).mean(axis=0))
        return np.mean(mats, axis=0)

    # Explanation obiekt
    try:
        from shap._explanation import Explanation  # type: ignore
        if isinstance(sv, Explanation):
            vals = np.asarray(sv.values)
            return np.abs(vals).mean(axis=0)
    except Exception:
        pass

    # ndarray
    arr = np.asarray(sv)
    if arr.ndim == 1:
        return np.abs(arr)
    # (n_samples, n_features) lub (n_classes, n_samples, n_features)
    if arr.ndim == 3:  # (C, N, F)
        return np.abs(arr).mean(axis=(0, 1))
    return np.abs(arr).mean(axis=0)

def _coefs_importance(est) -> Optional[np.ndarray]:
    try:
        coef = getattr(est, "coef_", None)
        if coef is None:
            return None
        arr = np.asarray(coef)
        if arr.ndim == 1:
            return np.abs(arr)
        # multiclass: (C, F) -> średnia z |coef|
        return np.abs(arr).mean(axis=0)
    except Exception:
        return None


# =========================
# SHAP importance
# =========================

def _try_shap_importance(estimator, Xt, feature_names: List[str], *, max_shap_samples: int = 400, random_state: int = 13) -> Optional[pd.DataFrame]:
    """
    Szybka i bezpieczna próba SHAP:
      - drzewiaste: TreeExplainer,
      - fallback: shap.Explainer(model, Xt),
      - sampling do max_shap_samples dla wydajności.
    Zwraca DataFrame(feature, importance) lub None.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    n = _safe_len(Xt)
    if n == 0:
        return None

    # sampling
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=min(max_shap_samples, n), replace=False)
    try:
        Xt_sample = Xt.iloc[idx]  # type: ignore[attr-defined]
    except Exception:
        try:
            Xt_sample = Xt[idx]
        except Exception:
            Xt_sample = Xt

    # SHAP: prefer TreeExplainer dla modeli drzewiastych
    shap_values = None
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(Xt_sample)
    except Exception:
        try:
            # Ogólny explainer (może zbudować odpowiedni masker)
            explainer = shap.Explainer(estimator, Xt_sample)
            sv = explainer(Xt_sample)
            shap_values = sv
        except Exception:
            return None

    vals = _mean_abs_over_classes(shap_values)
    vals = np.asarray(vals).ravel()

    # dopasuj długość do liczby cech (bywają rozjazdy w rzadkich przypadkach)
    F = len(feature_names)
    if vals.shape[0] != F:
        if vals.shape[0] > F:
            vals = vals[:F]
        else:
            pad = np.zeros(F - vals.shape[0], dtype=float)
            vals = np.concatenate([vals, pad], axis=0)

    df = pd.DataFrame({"feature": feature_names, "importance": vals, "source": "shap"})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# =========================
# Built-in importances
# =========================

def _from_feature_importances(estimator, feature_names: List[str]) -> Optional[pd.DataFrame]:
    # (1) standardowe feature_importances_
    vals = getattr(estimator, "feature_importances_", None)
    if vals is not None:
        arr = np.asarray(vals).ravel()
        F = len(feature_names)
        if arr.shape[0] != F:
            # próbuj przyciąć/dopadawać — rzadko potrzebne
            arr = arr[:F] if arr.shape[0] > F else np.pad(arr, (0, F - arr.shape[0]))
        return pd.DataFrame({"feature": feature_names, "importance": arr, "source": "model_importance"}).sort_values("importance", ascending=False).reset_index(drop=True)

    # (2) coef_ dla modeli liniowych
    coefs = _coefs_importance(estimator)
    if coefs is not None:
        arr = np.asarray(coefs).ravel()
        F = len(feature_names)
        if arr.shape[0] != F:
            arr = arr[:F] if arr.shape[0] > F else np.pad(arr, (0, F - arr.shape[0]))
        return pd.DataFrame({"feature": feature_names, "importance": arr, "source": "coef_abs"}).sort_values("importance", ascending=False).reset_index(drop=True)

    # (3) CatBoost / inne: metoda get_feature_importance ?
    try:
        if hasattr(estimator, "get_feature_importance"):
            arr = np.asarray(estimator.get_feature_importance()).ravel()
            F = len(feature_names)
            if arr.shape[0] != F:
                arr = arr[:F] if arr.shape[0] > F else np.pad(arr, (0, F - arr.shape[0]))
            return pd.DataFrame({"feature": feature_names, "importance": arr, "source": "model_importance"}).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    return None


# =========================
# Permutation importance
# =========================

def _choose_scoring(problem: Optional[str], estimator) -> str:
    prob = (problem or "").lower()
    if "reg" in prob:
        return "r2"
    # classification
    # jeśli raczej binarne i jest predict_proba → roc_auc
    try:
        if hasattr(estimator, "predict_proba"):
            return "roc_auc"
    except Exception:
        pass
    return "balanced_accuracy"

def _from_permutation(model_or_pipeline, X, y, *, scoring: Optional[str] = None, n_repeats: int = 5, random_state: int = 13) -> Optional[pd.DataFrame]:
    try:
        from sklearn.inspection import permutation_importance
    except Exception:
        return None

    try:
        r = permutation_importance(
            model_or_pipeline, X, y,
            n_repeats=int(n_repeats),
            random_state=int(random_state),
            n_jobs=-1,
            scoring=scoring
        )
        names = list(getattr(X, "columns", [f"f{i}" for i in range(r.importances_mean.shape[0])]))
        df = pd.DataFrame({
            "feature": names,
            "importance": r.importances_mean,
            "std": r.importances_std,
            "source": "permutation"
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return None


# =========================
# Public API
# =========================

def get_feature_importance(
    results: Dict[str, Any],
    scoring: Optional[str] = None,
    *,
    max_shap_samples: int = 400,
) -> Optional[pd.DataFrame]:
    """
    Zwraca DataFrame z ważnościami cech (kolumny: feature, importance, [std], source).
    Strategia:
      1) SHAP na estymatorze (po preprocesorze),
      2) wbudowane importances (feature_importances_/coef_/get_feature_importance),
      3) permutation_importance (na całym pipeline, jeśli dostępny).
    """
    try:
        pipe = results.get("pipeline")
        model = results.get("model") or results.get("best_model")
        X = results.get("X_test") or results.get("Xte") or results.get("X_valid")
        y = results.get("y_test") or results.get("yte") or results.get("y_valid")
        problem = results.get("problem")  # "classification" / "regression" (mile widziane)

        if X is None:
            return None

        # pipeline → rozdziel preprocesor i estymator
        pre, est = _split_pipeline(pipe if pipe is not None else model if model is not None else None)

        # Xt / feature_names
        Xt, feature_names = _transform_X_and_names(pre, X)

        # SHAP (najpierw spróbuj)
        shap_df = _try_shap_importance(est, Xt, feature_names, max_shap_samples=max_shap_samples)
        if shap_df is not None:
            return shap_df

        # Wbudowane importances z estymatora
        fi_df = _from_feature_importances(est, feature_names)
        if fi_df is not None:
            return fi_df

        # Permutation importance (na całym pipeline jeśli mamy pipe, inaczej na estymatorze + Xt)
        scorer = scoring or _choose_scoring(problem, est)
        if pipe is not None and y is not None:
            perm_df = _from_permutation(pipe, X, y, scoring=scorer)
            if perm_df is not None:
                return perm_df

        if y is not None:
            # brak całego pipeline — spróbuj na estymatorze i Xt
            perm_df = _from_permutation(est, Xt, y, scoring=scorer)
            if perm_df is not None:
                return perm_df

        return None
    except Exception:
        return None
