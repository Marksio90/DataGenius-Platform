from __future__ import annotations
# -*- coding: utf-8 -*-
"""
TMIV Scoring API — FastAPI
- artefakty: artifacts/last_run (można nadpisać env: TMIV_ARTIFACTS_DIR)
- model: voting/stacking/model .joblib/.pkl (pierwszy istniejący)
- plan: plan.json (używany przez backend.runtime_preprocessor.apply_plan)
- columns.json: {"target": "...", "feature_columns": [...]} do wyrównania kolumn i kolejności

Endpoints:
- GET  /health           -> {"status":"ok"}
- GET  /metadata         -> meta o modelu/kolumnach
- POST /predict          -> {"predictions":[...]}
- POST /predict_proba    -> {"classes":[...], "proba":[[...], ...]} lub binarka {"proba":[...]}
- POST /reload           -> przeładuj model/plan/meta (hot reload)

Uruchomienie lokalne:
    uvicorn scoring_api:app --host 0.0.0.0 --port 8000
Uwaga: Możesz ustawić:
    TMIV_ARTIFACTS_DIR      (domyślnie 'artifacts/last_run')
    TMIV_ALLOWED_ORIGINS    (lista CSV dla CORS, domyślnie '*')
    TMIV_MAX_RECORDS        (limit rekordów w jednym żądaniu, domyślnie 10000)
"""

from backend.safe_utils import truthy_df_safe

import os
import json
import time
import logging
import threading
from typing import List, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# --- joblib/pickle fallback ---
try:
    from joblib import load as joblib_load  # type: ignore
except Exception:  # pragma: no cover
    import pickle

    def joblib_load(p):
        with open(p, "rb") as f:
            return pickle.load(f)

# --- our preprocessor plan ---
from backend.runtime_preprocessor import apply_plan  # type: ignore

# =========================
# Konfiguracja / stałe
# =========================

API_VERSION = "1.1.0"

ARTIFACTS_DIR = os.environ.get("TMIV_ARTIFACTS_DIR", os.path.join("artifacts", "last_run"))
MODEL_PATHS = [
    "voting.joblib",
    "stacking.joblib",
    "model.joblib",
    "voting.pkl",
    "stacking.pkl",
    "model.pkl",
]
PLAN_PATH = os.path.join(ARTIFACTS_DIR, "plan.json")
COLUMNS_META_PATH = os.path.join(ARTIFACTS_DIR, "columns.json")

# Bezpieczeństwo / limity
DEFAULT_MAX_RECORDS = int(os.environ.get("TMIV_MAX_RECORDS", "10000"))

# Logging
logger = logging.getLogger("tmiv.scoring")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# =========================
# Pydantic modele (request/response)
# =========================
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista rekordów (dict) do predykcji")

    @validator("records")
    def _non_empty_records(cls, v):
        if not v:
            raise ValueError("records cannot be empty")
        if not isinstance(v, list):
            raise ValueError("records must be a list of dicts")
        if len(v) > DEFAULT_MAX_RECORDS:
            raise ValueError(f"too many records; max={DEFAULT_MAX_RECORDS}")
        return v


class PredictResponse(BaseModel):
    predictions: List[Any]
    model_path: Optional[str] = None
    problem: Optional[str] = None
    feature_count: Optional[int] = None


class PredictProbaResponse(BaseModel):
    classes: Optional[List[Any]] = None
    proba: List[Any]
    model_path: Optional[str] = None
    feature_count: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    loaded: bool
    problem: str
    version: str


class MetadataResponse(BaseModel):
    status: str
    version: str
    problem: str
    model_path: Optional[str] = None
    loaded_at: Optional[str] = None
    plan_present: bool = False
    columns_meta_present: bool = False
    feature_count: Optional[int] = None
    target: Optional[str] = None


# =========================
# Narzędzia
# =========================
def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _to_py(o: Any) -> Any:
    """Konwersja obiektów numpy/pandas do typów JSON-owych."""
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        # pd.isna obsługuje scalars
        if pd.isna(o):
            return None
    except Exception:
        pass
    return o


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Cannot read JSON: %s (%s)", path, e)
        return None


def _pick_existing(path_list: List[str], base: str) -> Optional[str]:
    for name in path_list:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def _infer_problem(model) -> str:
    # bardzo proste: czy mamy predict_proba/attr classes_
    try:
        if hasattr(model, "predict_proba") or hasattr(model, "classes_"):
            return "classification"
    except Exception:
        pass
    return "regression"


def _prepare_features(
    df_raw: pd.DataFrame, plan: dict, columns_meta: dict
) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """
    1) apply_plan (zgodnie z treningiem),
    2) odrzuć target,
    3) wyrównaj do feature_columns z columns.json: reindex (fill_value=0), drop extra.
    """
    target = plan.get("target") or columns_meta.get("target") or "target"
    df_prep = apply_plan(df_raw, target=target, plan=plan)

    # wyznacz kolumny cech z meta lub z df_prep (bez targetu)
    feature_columns = columns_meta.get("feature_columns", None)
    if not truthy_df_safe(feature_columns):
        feature_columns = [c for c in df_prep.columns if c != target]

    # X = dokładnie feature_columns (kolejność!), brakujące -> 0, nadmiarowe -> drop
    X = df_prep.reindex(columns=feature_columns, fill_value=0)

    # Upewnij się, że typy są akceptowalne dla modelu
    for c in X.columns:
        if str(X[c].dtype).startswith("category"):
            X[c] = X[c].astype("string").fillna("NA")
        if X[c].dtype == "boolean":
            X[c] = X[c].astype("int8")
    # konwersja bool/object na liczby (jeśli zostało)
    obj_like = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    for c in obj_like:
        try:
            X[c] = X[c].astype(float)
        except Exception:
            # stabilny fallback: hash trick w małej przestrzeni
            X[c] = X[c].astype(str).map(lambda v: (hash(v) % 997) / 997.0)

    return X, target, feature_columns


# =========================
# Globalny stan (cache w procesie)
# =========================
class _State:
    model: Any = None
    plan: dict = {}
    columns_meta: dict = {}
    problem: str = "regression"
    model_path: Optional[str] = None
    loaded_at: Optional[str] = None


STATE = _State()
_MODEL_LOCK = threading.RLock()  # chroni stan podczas reload/predict


def _load_model_and_meta() -> Tuple[Any, dict, dict]:
    if not os.path.isdir(ARTIFACTS_DIR):
        raise FileNotFoundError(f"Brak katalogu artefaktów: {ARTIFACTS_DIR}")

    model_file = _pick_existing(MODEL_PATHS, ARTIFACTS_DIR)
    if not model_file:
        raise FileNotFoundError("Nie znaleziono modelu (szukano: " + ", ".join(MODEL_PATHS) + ")")

    logger.info("Loading model from: %s", model_file)
    model = joblib_load(model_file)

    plan = _safe_read_json(PLAN_PATH) or {}
    columns_meta = _safe_read_json(COLUMNS_META_PATH) or {}

    return model, plan, columns_meta


def _reload_state() -> dict:
    with _MODEL_LOCK:
        model, plan, columns_meta = _load_model_and_meta()
        STATE.model = model
        STATE.plan = plan
        STATE.columns_meta = columns_meta
        STATE.problem = _infer_problem(model)
        STATE.model_path = _pick_existing(MODEL_PATHS, ARTIFACTS_DIR)
        STATE.loaded_at = _now_ts()
        logger.info(
            "Reloaded model: path=%s problem=%s features=%s",
            STATE.model_path,
            STATE.problem,
            len(columns_meta.get("feature_columns", [])) if columns_meta else None,
        )
        return {
            "status": "reloaded",
            "problem": STATE.problem,
            "model_path": STATE.model_path,
            "loaded_at": STATE.loaded_at,
            "has_plan": bool(plan),
            "has_columns_meta": bool(columns_meta),
        }


# =========================
# FastAPI app
# =========================
app = FastAPI(title="TMIV Scoring API", version=API_VERSION)

# CORS (domyślnie luźny; dociśnij dla produkcji)
_allowed = os.environ.get("TMIV_ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in _allowed.split(",")] if _allowed else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,  # zawęź w prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy init on import
try:
    _reload_state()
except Exception as e:  # brak artefaktów na starcie – API wstanie, ale zgłosi przy predict
    logger.warning("[TMIV] Initial load warning: %s", e)


# =========================
# Endpoints
# =========================
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        loaded=STATE.model is not None,
        problem=STATE.problem,
        version=API_VERSION,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    return MetadataResponse(
        status="ok" if STATE.model is not None else "uninitialized",
        version=API_VERSION,
        problem=STATE.problem,
        model_path=STATE.model_path,
        loaded_at=STATE.loaded_at,
        plan_present=bool(STATE.plan),
        columns_meta_present=bool(STATE.columns_meta),
        feature_count=len(STATE.columns_meta.get("feature_columns", [])) if STATE.columns_meta else None,
        target=STATE.plan.get("target") or STATE.columns_meta.get("target"),
    )


@app.post("/reload")
def reload_artifacts():
    try:
        info = _reload_state()
        return info
    except Exception as e:
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Skopiuj referencje pod lockiem aby uniknąć race z reloadem
    with _MODEL_LOCK:
        model = STATE.model
        plan = STATE.plan
        columns_meta = STATE.columns_meta
        model_path = STATE.model_path
        feature_count = len(columns_meta.get("feature_columns", [])) if columns_meta else None
        problem = STATE.problem

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model niezaładowany: sprawdź artifacts/last_run lub /reload",
        )

    # surowe -> DataFrame
    try:
        df_raw = pd.DataFrame(req.records)
        if df_raw.empty:
            raise ValueError("Brak rekordów")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowe 'records': {e}") from e

    # przygotowanie cech
    try:
        X, target, feat_cols = _prepare_features(df_raw, plan, columns_meta)
    except Exception as e:
        logger.exception("Preprocessing error")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}") from e

    # predykcja
    try:
        yhat = model.predict(X)
    except Exception as e:
        logger.exception("Predict error")
        raise HTTPException(status_code=500, detail=f"Predict error: {e}") from e

    preds = [_to_py(v) for v in yhat]
    return PredictResponse(predictions=preds, model_path=model_path, problem=problem, feature_count=feature_count)


@app.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(req: PredictRequest):
    with _MODEL_LOCK:
        model = STATE.model
        plan = STATE.plan
        columns_meta = STATE.columns_meta
        model_path = STATE.model_path
        feature_count = len(columns_meta.get("feature_columns", [])) if columns_meta else None

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model niezaładowany: sprawdź artifacts/last_run lub /reload",
        )
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=400, detail="Model nie wspiera predict_proba (nie-klasyfikacja?)")

    try:
        df_raw = pd.DataFrame(req.records)
        if df_raw.empty:
            raise ValueError("Brak rekordów")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowe 'records': {e}") from e

    try:
        X, target, feat_cols = _prepare_features(df_raw, plan, columns_meta)
        proba = model.predict_proba(X)
    except Exception as e:
        logger.exception("Predict_proba error")
        raise HTTPException(status_code=500, detail=f"Predict_proba error: {e}") from e

    # Binarka: zwróć p(klasa=1) jeżeli ma 2 kolumny
    try:
        classes = getattr(model, "classes_", None)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            return PredictProbaResponse(
                classes=[_to_py(int(c)) if isinstance(c, (np.integer,)) else _to_py(c) for c in (classes or [0, 1])],
                proba=[_to_py(p[1]) for p in proba],
                model_path=model_path,
                feature_count=feature_count,
            )
        else:
            # multiclass: pełna macierz
            if classes is None and isinstance(proba, np.ndarray) and proba.ndim == 2:
                classes = list(range(proba.shape[1]))
            return PredictProbaResponse(
                classes=[_to_py(int(c)) if isinstance(c, (np.integer,)) else _to_py(c) for c in (classes or [])]
                if classes is not None
                else None,
                proba=_to_py(proba),
                model_path=model_path,
                feature_count=feature_count,
            )
    except Exception:
        # awaryjnie, bez klas
        return PredictProbaResponse(classes=None, proba=_to_py(proba), model_path=model_path, feature_count=feature_count)


# =========================
# Uruchomienie lokalne
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting TMIV Scoring API v%s on 0.0.0.0:%d", API_VERSION, port)
    uvicorn.run(app, host="0.0.0.0", port=port)
