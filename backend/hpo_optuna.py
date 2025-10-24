"""
hpo_optuna.py
Docstring (PL): Prosty HPO z Optuna (gated). Jeśli Optuna nie jest zainstalowana, zwraca komunikat.
Dla klasyfikacji optymalizuje ROC_AUC, a dla regresji minimalizuje RMSE.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = list(X.select_dtypes(include=[np.number]).columns)
    cat = [c for c in X.columns if c not in num]
    trf = []
    if num:
        trf.append(("num", StandardScaler(with_mean=False), num))
    if cat:
        trf.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat))
    return ColumnTransformer(trf) if trf else "passthrough"

def hpo_rf(df: pd.DataFrame, target: str, problem_type: str, n_trials: int = 20, timeout_sec: int = 180, random_state: int = 42) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        return {"status":"TMIV-HPO-000","message":"Optuna nie jest zainstalowana. Włącz USE_OPTUNA po doinstalowaniu pakietu 'optuna'."}
    X = df.drop(columns=[target]); y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if problem_type=="classification" and y.nunique()>1 else None)
    pre = _preprocessor(Xtr)

    def objective(trial):
        if problem_type == "classification":
            n_estimators = trial.suggest_int("n_estimators", 100, 600)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            mdl = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=-1, random_state=random_state
            )
            pipe = Pipeline([("prep", pre), ("model", mdl)])
            pipe.fit(Xtr, ytr)
            # proby
            try:
                p = pipe.predict_proba(Xte)
                prob = p[:,1] if p.ndim==2 else p
            except Exception:
                prob = pipe.predict(Xte)
            if len(np.unique(yte))<2:
                return 0.5
            return roc_auc_score(yte, prob)
        else:
            n_estimators = trial.suggest_int("n_estimators", 100, 800)
            max_depth = trial.suggest_int("max_depth", 3, 24)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            mdl = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=-1, random_state=random_state
            )
            pipe = Pipeline([("prep", pre), ("model", mdl)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            rmse = mean_squared_error(yte, pred, squared=False)
            # Optuna maksymalizuje domyślnie — zwracamy ujemny RMSE aby maksymalizować (lub zmieniamy kierunek w study)
            return -rmse

    direction = "maximize" if problem_type=="classification" else "maximize"  # regresja: maksymalizujemy -RMSE
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)
    best = study.best_trial
    return {
        "status":"OK",
        "best_params": best.params,
        "best_value": best.value if problem_type=="classification" else -best.value,
        "direction": direction,
        "n_trials": len(study.trials),
    }