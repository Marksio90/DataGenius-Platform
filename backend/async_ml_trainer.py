"""
async_ml_trainer.py
Docstring (PL): Asynchroniczny trening wielu modeli z zachowaniem limitów czasu i paralelizmem.
Zapewnia poprawne kodowanie kategorii (OneHotEncoder bez użycia parametru `sparse` – tylko `sparse_output`),
oraz dobór modeli do klasyfikacji/regresji. Zwraca metryki i status per model.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Opcjonalne biblioteki
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

def _build_ohe():
    """
    Docstring (PL): Zwraca OneHotEncoder używający `sparse_output=True` (zgodny z sklearn>=1.2).
    Zakazane jest użycie parametru `sparse`.
    """
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)

def _split(df: pd.DataFrame, target: str, problem_type: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    stratify = y if (problem_type == "classification" and y.nunique() > 1) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=False), num_cols))
    if cat_cols:
        transformers.append(("cat", _build_ohe(), cat_cols))
    if not transformers:
        # w rzadkim przypadku pustej macierzy we/wy
        transformers.append(("passthrough", "passthrough", []))
    return ColumnTransformer(transformers)

def _cls_models(strategy: str) -> List[Tuple[str, Any]]:
    models = [("logreg", LogisticRegression(max_iter=200)) , ("rf_cls", RandomForestClassifier(n_estimators=200, n_jobs=-1))]
    if strategy in ("accurate", "advanced"):
        if HAS_XGB: models.append(("xgb_cls", XGBClassifier(n_estimators=400, max_depth=6, eval_metric="logloss", n_jobs=-1)))
        if HAS_LGBM: models.append(("lgbm_cls", LGBMClassifier(n_estimators=400)))
        if HAS_CAT: models.append(("cat_cls", CatBoostClassifier(verbose=False)))
    return models

def _reg_models(strategy: str) -> List[Tuple[str, Any]]:
    models = [("ridge", Ridge(alpha=1.0)), ("rf_reg", RandomForestRegressor(n_estimators=300, n_jobs=-1))]
    if strategy in ("accurate", "advanced"):
        if HAS_XGB: models.append(("xgb_reg", XGBRegressor(n_estimators=600, max_depth=8, tree_method="hist", n_jobs=-1)))
        if HAS_LGBM: models.append(("lgbm_reg", LGBMRegressor(n_estimators=600)))
        if HAS_CAT: models.append(("cat_reg", CatBoostRegressor(verbose=False)))
    return models

def _evaluate_cls(y_true, y_prob, y_pred) -> Dict[str, float]:
    metrics = {}
    try:
        # obsłuż przypadek 1-klasowy
        if len(np.unique(y_true)) < 2:
            metrics["ROC_AUC"] = float("nan")
            metrics["AveragePrecision"] = float("nan")
        else:
            # jeśli proby są w kształcie (n,2) weź klasę pozytywną 1
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob
            metrics["ROC_AUC"] = roc_auc_score(y_true, prob_pos)
            metrics["AveragePrecision"] = average_precision_score(y_true, prob_pos)
    except Exception:
        metrics["ROC_AUC"] = float("nan")
        metrics["AveragePrecision"] = float("nan")
    metrics["F1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    return metrics

def _evaluate_reg(y_true, y_pred) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Store best model predictions for downstream plots
    best_payload = None
    if best_name and results.get(best_name, {}).get('status')=='ok':
        best_payload = {
            'y_pred': results[best_name].get('y_pred'),
            'y_prob': results[best_name].get('y_prob')
        }
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def train_async(df: pd.DataFrame, target: str, problem_type: str, strategy: str = "balanced",
                max_parallel: int = 2, max_time_sec: int = 180, random_state: int = 42) -> Dict[str, Any]:
    """
    Docstring (PL): Uruchamia równoległe trenowanie zestawu modeli zgodnych z typem problemu.
    Zwraca słownik z metrykami i informacją o najlepszym modelu. Czas liczony per model (timeout).
    """
    X_train, X_test, y_train, y_test = _split(df, target, problem_type, random_state=random_state)
    pre = _preprocessor(X_train)

    if problem_type == "classification":
        candidates = _cls_models(strategy)
    else:
        candidates = _reg_models(strategy)

    results = {}
    start = time.time()

    def fit_one(name, est):
        pipe = Pipeline([("prep", pre), ("model", est)])
        ok = False
        err = None
        y_pred = None
        y_prob = None
        try:
            pipe.fit(X_train, y_train)
            ok = True
            if problem_type == "classification":
                metrics = _evaluate_cls(y_test, y_prob if y_prob is not None else np.zeros((len(y_test), 1)), y_pred)
                results[name] = {"status": "ok", "metrics": metrics, "y_true": y_test.tolist(), "y_pred": y_pred.tolist() if y_pred is not None else None, "y_prob": y_prob.tolist() if y_prob is not None else None}
            else:
                metrics = _evaluate_reg(y_test, y_pred)
                results[name] = {"status": "ok", "metrics": metrics, "y_true": y_test.tolist(), "y_pred": y_pred.tolist() if y_pred is not None else None}
                continue
            if err:
                results[name] = {"status": "error", "error": err}
                continue
            if problem_type == "classification":
                metrics = _evaluate_cls(y_test, y_prob if y_prob is not None else np.zeros((len(y_test), 1)), y_pred)
                results[name] = {"status": "ok", "metrics": metrics, "y_true": y_test.tolist(), "y_pred": y_pred.tolist() if y_pred is not None else None, "y_prob": y_prob.tolist() if y_prob is not None else None}
            else:
                metrics = _evaluate_reg(y_test, y_pred)
                results[name] = {"status": "ok", "metrics": metrics, "y_true": y_test.tolist(), "y_pred": y_pred.tolist() if y_pred is not None else None}
            else:
                metrics = _evaluate_reg(y_test, y_pred)
                results[name] = {"status": "ok", "metrics": metrics, "y_pred": y_pred.tolist() if y_pred is not None else None}

    # Wybór najlepszego modelu (prosty ranking: dla cls ROC_AUC, dla reg odwrotny RMSE)
    best_name = None
    best_score = -np.inf if problem_type == "classification" else np.inf
    for name, info in results.items():
        # attach preds/proba for candidate to help best selection plotting
        if info.get("status") != "ok":
            continue
        m = info["metrics"]
        if problem_type == "classification":
            score = float(m.get("ROC_AUC", float("nan")))
            if np.isnan(score):
                score = float(m.get("Accuracy", 0.0))
            if score > best_score:
                best_score = score
                best_name = name
        else:
            score = float(m.get("RMSE", float("inf")))
            if score < best_score:
                best_score = score
                best_name = name

    elapsed = time.time() - start
    # Store best model predictions for downstream plots
    best_payload = None
    if best_name and results.get(best_name, {}).get('status')=='ok':
        best_payload = {
            'y_pred': results[best_name].get('y_pred'),
            'y_prob': results[best_name].get('y_prob')
        }
    return {
        "status": "OK",
        "problem_type": problem_type,
        "results": results,
        "best_model": best_name,
        "elapsed_sec": round(elapsed, 2),
    }