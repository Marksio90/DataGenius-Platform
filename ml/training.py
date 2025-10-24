# ml/training.py
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.encoders import safe_one_hot_encoder
from ml.utils import detect_target_and_task
from .fi import unify_feature_importance


def build_pipeline(df: pd.DataFrame, target: str, task: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """Buduje pipeline: scaler dla numerycznych + OHE (safe) dla kategorycznych + model RF."""
    numeric = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    transformers = []
    if numeric:
        # with_mean=False by uniknąć konfliktu ze sparse macierzą po OHE
        transformers.append(("num", StandardScaler(with_mean=False), numeric))
    if categorical:
        ohe = safe_one_hot_encoder(handle_unknown="ignore")
        transformers.append(("cat", ohe, categorical))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=42
        )
    else:
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe, {"numeric": numeric, "categorical": categorical}


def train_and_eval(
    df: pd.DataFrame,
    target: str | None = None,
    random_state: int = 42,
    max_train_time_sec: int = 180,
) -> Dict[str, Any]:
    """
    Trenuje model i zwraca metryki + obiekt pipeline.
    - Timeout trenowania: TMIV-ML-TIMEOUT (przerywa elegancko).
    - Dla klasyfikacji zapisuje artifacts/models/preds.json (y_true, y_prob) do wykresów/PDF.
    - Zapisuje CSV z unifikowanym Feature Importance: artifacts/plots/feature_importance.csv
    """
    target, task = detect_target_and_task(df, target)

    X = df.drop(columns=[target])
    y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if task == "classification" else None
    )

    pipe, cols = build_pipeline(df, target, task)

    # --- Timeout-controlled fit ---
    exc_holder: Dict[str, BaseException] = {}

    def _fit() -> None:
        try:
            pipe.fit(Xtr, ytr)
        except BaseException as e:  # szeroko: chcemy wynieść błąd z wątku
            exc_holder["e"] = e

    th = threading.Thread(target=_fit, daemon=True)
    th.start()
    th.join(timeout=max_train_time_sec)
    if th.is_alive():
        # Nie zostawiamy wiszącego wątku – kończy się procesem przy następnym wejściu,
        # ale komunikat dla usera jasno sygnalizuje timeout.
        raise TimeoutError("TMIV-ML-TIMEOUT: przekroczono czas treningu")
    if "e" in exc_holder:
        raise exc_holder["e"]

    # --- Ewaluacja ---
    metrics: Dict[str, Any] = {"task": task}

    if task == "classification":
        # Predykcje klas i proby
        yhat = pipe.predict(Xte)
        metrics["accuracy"] = float(accuracy_score(yte, yhat))
        metrics["f1"] = float(f1_score(yte, yhat, average="weighted"))

        proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(Xte)[:, 1]
            except Exception:
                proba = None

        if proba is not None and len(np.unique(yte)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(yte, proba))
            except Exception:
                pass
            try:
                metrics["brier"] = float(brier_score_loss(yte, proba))
            except Exception:
                pass

            # Zapis preds.json do późniejszych wykresów/PDF
            try:
                os.makedirs("artifacts/models", exist_ok=True)
                with open("artifacts/models/preds.json", "w", encoding="utf-8") as f:
                    json.dump({"y_true": yte.tolist(), "y_prob": proba.tolist()}, f)
            except Exception:
                pass

    else:
        # Regressja
        yhat = pipe.predict(Xte)
        rmse = float(mean_squared_error(yte, yhat, squared=False))
        r2 = float(r2_score(yte, yhat))
        metrics["rmse"] = rmse
        metrics["r2"] = r2

    # --- Feature Importance unify + CSV ---
    try:
        pre = pipe.named_steps.get("pre")
        feature_names_in = getattr(pre, "feature_names_in_", None)
        if feature_names_in is None:
            feature_names_in = list(X.columns)
        importances_df = unify_feature_importance(pipe, list(feature_names_in))
        os.makedirs("artifacts/plots", exist_ok=True)
        importances_df.to_csv("artifacts/plots/feature_importance.csv", index=False)
    except Exception:
        pass

    # Zwracamy pełny wynik (UWAGA: obiekt modelu jest w słowniku – to zamierzone)
    return {"target": target, "columns": cols, "metrics": metrics, "model": pipe}
