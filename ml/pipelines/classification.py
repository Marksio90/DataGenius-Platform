# ml/pipelines/classification.py
"""
Classification pipelines (skeleton) – TMIV Advanced ML Platform.

Cel:
- Dostarczyć gotowy zestaw **klasyfikatorów** oraz pomocniczych narzędzi,
  które mogą zostać użyte przez wyższe warstwy (np. `core/services/ml_service.py`)
  lub bezpośrednio w eksperymentach.
- Brak zależności od Streamlit/UI.

Najważniejsze elementy:
- `build_classification_models(n_jobs=-1, random_state=42)` → dict[name -> estimator]
- `make_preprocessor(X)` → ColumnTransformer (imputacja + skalowanie + OneHot)
- `make_pipeline(preprocessor, estimator)` → sklearn.Pipeline
- Stałe z metrykami rekomendowanymi dla klasyfikacji.

Uwaga:
- Modele boostowane (XGBoost/LightGBM/CatBoost) są ładowane **opcjonalnie**.
- To jest świadomie „szkielet produkcyjny” – działa od razu,
  a jednocześnie można go łatwo rozszerzać (grid/tuning/ensembling).

"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --- soft-importy na boostowane modele ---
_xgbc = None
_lgbmc = None
_catc = None
try:  # pragma: no cover
    from xgboost import XGBClassifier as _xgbc  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from lightgbm import LGBMClassifier as _lgbmc  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from catboost import CatBoostClassifier as _catc  # type: ignore
except Exception:
    pass


# =========================
# Preprocessing
# =========================

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Zbuduj przetwarzanie wstępne:
      - numeryczne: imputacja medianą + StandardScaler
      - kategoryczne: imputacja trybem + OneHotEncoder
    """
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

    # low-cardinality ints → kategorie
    for c in X.columns.difference(num_cols + cat_cols):
        s = X[c]
        if pd.api.types.is_integer_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if len(s) and nunique <= 20 and (nunique / len(s)) < 0.2:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            # domyślnie traktuj jako kategorie
            cat_cols.append(c)

    num_tr = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_tr = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    pre = ColumnTransformer(
        transformers=[("num", num_tr, num_cols), ("cat", cat_tr, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def make_pipeline(preprocessor: ColumnTransformer, estimator: BaseEstimator) -> Pipeline:
    """Sklej preprocessor + estymator w jeden Pipeline."""
    return Pipeline(steps=[("pre", preprocessor), ("est", estimator)])


# =========================
# Modele
# =========================

def build_classification_models(*, n_jobs: int = -1, random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Zwróć słownik podstawowych klasyfikatorów.
    - rdzeń: LogisticRegression, RandomForest, GradientBoosting
    - opcjonalnie: XGBoost, LightGBM, CatBoost (jeśli zainstalowane)
    """
    models: Dict[str, BaseEstimator] = {
        "logreg": LogisticRegression(max_iter=1_000, n_jobs=int(n_jobs), class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=int(n_jobs), class_weight="balanced", random_state=int(random_state)),
        "gbc": GradientBoostingClassifier(random_state=int(random_state)),
    }

    if _xgbc is not None:
        models["xgb"] = _xgbc(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=int(n_jobs),
            random_state=int(random_state),
        )
    if _lgbmc is not None:
        models["lgbm"] = _lgbmc(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
        )
    if _catc is not None:
        models["cat"] = _catc(
            iterations=800,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            loss_function="Logloss",
            random_state=int(random_state),
        )

    return models


# =========================
# Metryki – rekomendacje
# =========================

# Metryka główna (domyślna) – dobra na start dla binary/multiclass (OVR)
PRIMARY_METRIC: str = "roc_auc"

# Metryki drugorzędne (używane do porównań / radarów, jeśli dostępne)
SECONDARY_METRICS: List[str] = [
    "accuracy",
    "f1_weighted",
    "f1",
    "logloss",
    "aps",  # average precision
]

# Kierunki – większe lepsze / mniejsze lepsze
BIGGER_IS_BETTER: Sequence[str] = ("accuracy", "f1", "f1_weighted", "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "aps")
LOWER_IS_BETTER: Sequence[str] = ("logloss",)


def is_probabilistic_estimator(est: BaseEstimator) -> bool:
    """
    Czy estymator potrafi zwrócić `predict_proba` lub `decision_function`?
    Przydatne do wyboru wykresów (ROC/PR/kalibracja).
    """
    return hasattr(est, "predict_proba") or hasattr(est, "decision_function")


__all__ = [
    "make_preprocessor",
    "make_pipeline",
    "build_classification_models",
    "PRIMARY_METRIC",
    "SECONDARY_METRICS",
    "BIGGER_IS_BETTER",
    "LOWER_IS_BETTER",
    "is_probabilistic_estimator",
]
