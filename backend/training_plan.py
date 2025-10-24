"""
training_plan.py
Docstring (PL): Wykrywa kolumnę celu (target) i typ problemu (klasyfikacja/regresja/TS)
na podstawie heurystyk: regex nazw, liczba unikatów, typ danych, sygnał czasowy.
Dobiera też strategię walidacji (KFold/Stratified/TimeSeriesSplit) i metryki.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

TARGET_REGEX = re.compile(r"(target|label|class|y|outcome|result)$", re.IGNORECASE)

def detect_target(df: pd.DataFrame) -> Optional[str]:
    # 1) Regex w nazwach
    for c in df.columns:
        if TARGET_REGEX.search(str(c)):
            return c
    # 2) Heurystyka: ostatnia kolumna nie-datetime o małej krotności
    candidates = [c for c in df.columns if not pd.api.types.is_datetime64_any_dtype(df[c])]
    if not candidates:
        return None
    # mała liczba unikatów → preferuj
    scored = []
    for c in candidates:
        nunique = df[c].nunique(dropna=True)
        scored.append((nunique, c))
    scored.sort()
    return scored[0][1] if scored else None

def detect_problem_type(df: pd.DataFrame, target: str) -> str:
    # Sygnał TS: istnieje kolumna datetime o rosnącym porządku
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        # jeżeli główna kolumna datetime monotoniczna
        for c in dt_cols:
            if df[c].is_monotonic_increasing:
                return "timeseries"
    # klasyfikacja vs regresja
    y = df[target]
    if pd.api.types.is_numeric_dtype(y):
        # mała krotność → klasyfikacja
        if y.nunique(dropna=True) <= 20:
            return "classification"
        return "regression"
    else:
        return "classification"

def plan_validation(df: pd.DataFrame, target: str, problem_type: str, n_splits: int = 5) -> Dict[str, Any]:
    if problem_type == "timeseries":
        splitter = "TimeSeriesSplit"
        cv = {"name": splitter, "n_splits": n_splits}
        metrics = ["RMSE", "MAE", "R2"]
    elif problem_type == "classification":
        splitter = "StratifiedKFold"
        cv = {"name": splitter, "n_splits": n_splits, "shuffle": True, "random_state": 42}
        metrics = ["ROC_AUC", "F1", "Accuracy", "AveragePrecision"]
    else:
        splitter = "KFold"
        cv = {"name": splitter, "n_splits": n_splits, "shuffle": True, "random_state": 42}
        metrics = ["RMSE", "MAE", "R2"]
    return {"cv": cv, "metrics": metrics}

def build_training_plan(df: pd.DataFrame, strategy: str = "balanced") -> Dict[str, Any]:
    target = detect_target(df)
    if not target:
        return {"status": "TMIV-PLAN-001", "message": "Nie wykryto kolumny celu."}
    problem_type = detect_problem_type(df, target)
    val = plan_validation(df, target, problem_type)
    return {
        "status": "OK",
        "target": target,
        "problem_type": problem_type,
        "validation": val,
        "strategy": strategy,
    }