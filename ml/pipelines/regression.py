# ml/pipelines/regression.py
"""
Regression pipelines (skeleton) – TMIV Advanced ML Platform.

Cel:
- Dostarczyć gotowy zestaw **regresorów** i narzędzia do sklejania pipeline’ów,
  możliwe do użycia w `core/services/ml_service.py` lub eksperymentach.
- Brak zależności od Streamlit/UI.

API:
- make_preprocessor(X: pd.DataFrame) -> ColumnTransformer
- make_pipeline(preprocessor, estimator) -> sklearn.Pipeline
- build_regression_models(n_jobs=-1, random_state=42) -> dict[name -> estimator]

Uwagi:
- Modele boostowane (XGBoost/LightGBM/CatBoost) ładowane **opcjonalnie** (soft-import).
- Preprocessing: imputacja, skalowanie (num), OneHot (kat).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- soft-importy na boostowane modele ---
_xgbr = None
_lgbmr = None
_catr = None
try:  # pragma: no cover
    from xgboost import XGBRegressor as _xgbr  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from lightgbm import LGBMRegressor as _lgbmr  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from catboost import CatBoostRegressor as _catr  # type: ignore
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

def build_regression_models(*, n_jobs: int = -1, random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Zwróć słownik podstawowych regresorów.
    - rdzeń: LinearRegression, Ridge, RandomForestRegressor, GradientBoostingRegressor
    - opcjonalnie: XGBoost, LightGBM, CatBoost (jeśli zainstalowane)
    """
    models: Dict[str, BaseEstimator] = {
        "linreg": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=int(random_state)),
        "rfr": RandomForestRegressor(n_estimators=400, n_jobs=int(n_jobs), random_state=int(random_state)),
        "gbr": GradientBoostingRegressor(random_state=int(random_state)),
    }

    if _xgbr is not None:
        models["xgb"] = _xgbr(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
        )
    if _lgbmr is not None:
        models["lgbm"] = _lgbmr(
            n_estimators=700,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
        )
    if _catr is not None:
        models["cat"] = _catr(
            iterations=900,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            loss_function="RMSE",
            random_state=int(random_state),
        )

    return models


# =========================
# Metryki – rekomendacje
# =========================

# Metryka główna (domyślna) – dla regresji
PRIMARY_METRIC: str = "r2"

# Metryki drugorzędne (porównania / radar)
SECONDARY_METRICS: List[str] = [
    "rmse",
    "mae",
]

# Kierunki – większe lepsze / mniejsze lepsze
BIGGER_IS_BETTER: Sequence[str] = ("r2",)
LOWER_IS_BETTER: Sequence[str] = ("rmse", "mae")


__all__ = [
    "make_preprocessor",
    "make_pipeline",
    "build_regression_models",
    "PRIMARY_METRIC",
    "SECONDARY_METRICS",
    "BIGGER_IS_BETTER",
    "LOWER_IS_BETTER",
]
