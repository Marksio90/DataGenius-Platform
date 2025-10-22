from __future__ import annotations
# -*- coding: utf-8 -*-

# backend/optional_engines.py
from typing import Any, Dict, Optional, Tuple
from functools import lru_cache
import importlib

# -------------------------------------------------------------------
# Bezpieczne importy
# -------------------------------------------------------------------

def _safe_import(name: str):
    """Bezpieczny import modułu: zwraca moduł lub None, nigdy nie podnosi wyjątku."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# -------------------------------------------------------------------
# Detekcja dostępności (cache’owana)
# -------------------------------------------------------------------

@lru_cache(maxsize=8)
def has_xgboost() -> bool:
    return _safe_import("xgboost") is not None

@lru_cache(maxsize=8)
def has_lightgbm() -> bool:
    return _safe_import("lightgbm") is not None

@lru_cache(maxsize=8)
def has_catboost() -> bool:
    return _safe_import("catboost") is not None


# -------------------------------------------------------------------
# Wersje pakietów (do diagnostyki / UI)
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"xgboost": None, "lightgbm": None, "catboost": None}
    for pkg in list(out.keys()):
        m = _safe_import(pkg)
        try:
            ver = getattr(m, "__version__", None)
        except Exception:
            ver = None
        out[pkg] = ver
    return out


# -------------------------------------------------------------------
# Klasy estymatorów (leniwe, bezpieczne)
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_regressor_classes() -> Dict[str, Any]:
    """
    Mapuje nazwę → klasę estymatora dla boosterów regresyjnych.
    Importy są leniwe i bezpieczne — brak biblioteki = brak wpisu.
    """
    out: Dict[str, Any] = {}
    if has_xgboost():
        try:
            from xgboost import XGBRegressor
            out["XGBRegressor"] = XGBRegressor
        except Exception:
            pass
    if has_lightgbm():
        try:
            from lightgbm import LGBMRegressor
            out["LGBMRegressor"] = LGBMRegressor
        except Exception:
            pass
    if has_catboost():
        try:
            from catboost import CatBoostRegressor
            out["CatBoostRegressor"] = CatBoostRegressor
        except Exception:
            pass
    return out

@lru_cache(maxsize=1)
def get_classifier_classes() -> Dict[str, Any]:
    """
    Mapuje nazwę → klasę estymatora dla boosterów klasyfikacyjnych.
    """
    out: Dict[str, Any] = {}
    if has_xgboost():
        try:
            from xgboost import XGBClassifier
            out["XGBClassifier"] = XGBClassifier
        except Exception:
            pass
    if has_lightgbm():
        try:
            from lightgbm import LGBMClassifier
            out["LGBMClassifier"] = LGBMClassifier
        except Exception:
            pass
    if has_catboost():
        try:
            from catboost import CatBoostClassifier
            out["CatBoostClassifier"] = CatBoostClassifier
        except Exception:
            pass
    return out


# -------------------------------------------------------------------
# Normalizacja nazw i rozwiązywanie aliasów
# -------------------------------------------------------------------

_ALIASES: Dict[str, str] = {
    # klasyfikacja
    "xgbclassifier": "XGBClassifier",
    "xgb_clf": "XGBClassifier",
    "xgb": "XGBClassifier",
    "lightgbmclassifier": "LGBMClassifier",
    "lgbmclassifier": "LGBMClassifier",
    "lgbm_clf": "LGBMClassifier",
    "lgbm": "LGBMClassifier",
    "catboostclassifier": "CatBoostClassifier",
    "catboost_clf": "CatBoostClassifier",
    "cat": "CatBoostClassifier",
    # regresja
    "xgbregressor": "XGBRegressor",
    "xgb_reg": "XGBRegressor",
    "lgbmregressor": "LGBMRegressor",
    "lightgbmregressor": "LGBMRegressor",
    "lgbm_reg": "LGBMRegressor",
    "catboostregressor": "CatBoostRegressor",
    "catboost_reg": "CatBoostRegressor",
}

def _normalize_name(name: str) -> str:
    n = (name or "").replace("-", "").replace("_", "").lower()
    return _ALIASES.get(n, name)


# -------------------------------------------------------------------
# Listowanie dostępnych estymatorów pod dany problem
# -------------------------------------------------------------------

def list_available_estimators(problem: Optional[str] = None) -> Dict[str, Any]:
    """
    Zwraca słownik {nazwa: klasa} ograniczony do problemu ('classification'/'regression'),
    albo wszystkie dostępne jeśli problem=None.
    """
    p = (problem or "").lower()
    if p.startswith("class"):
        return dict(get_classifier_classes())
    if p.startswith("reg"):
        return dict(get_regressor_classes())
    out = {}
    out.update(get_classifier_classes())
    out.update(get_regressor_classes())
    return out


# -------------------------------------------------------------------
# Rozwiązywanie i tworzenie estymatora
# -------------------------------------------------------------------

def resolve_estimator(name: str, problem: Optional[str] = None) -> Optional[Any]:
    """
    Zwraca klasę estymatora po nazwie/aliasie albo None, jeśli brak biblioteki.
    """
    if not name:
        return None
    canon = _normalize_name(name)
    pool = list_available_estimators(problem)
    # najpierw dokładnie
    if canon in pool:
        return pool[canon]
    # potem „miękko” po case-insensitive
    for k, v in pool.items():
        if k.lower() == canon.lower():
            return v
    return None


def create_estimator(name: str, problem: Optional[str] = None, **kwargs) -> Any:
    """
    Tworzy instancję estymatora (o ile biblioteka jest dostępna).
    Rzuca przyjazny ImportError, gdy pakiet nie jest zainstalowany.
    """
    cls = resolve_estimator(name, problem)
    if cls is None:
        hints = missing_boosters_hints()
        raise ImportError(
            f"Requested estimator '{name}' is not available. "
            f"Install missing packages: {', '.join(hints.values()) or 'xgboost / lightgbm / catboost'}"
        )
    # drobne defaulty przyjazne dla „quiet” treningu
    if "random_state" not in kwargs:
        kwargs["random_state"] = 42
    # CatBoost: domyślne wyciszenie
    try:
        if cls.__name__.startswith("CatBoost") and "verbose" not in kwargs:
            kwargs["verbose"] = False
    except Exception:
        pass
    return cls(**kwargs)


# -------------------------------------------------------------------
# Domyślne przestrzenie HPO (lekkie, sensowne)
# -------------------------------------------------------------------

def default_param_space(name: str) -> Dict[str, Any]:
    """
    Domyślne przestrzenie hiperparametrów dla grid/random-searchy.
    Zwraca pusty słownik dla nierozpoznanych nazw (nie wymusza zależności).
    """
    spaces: Dict[str, Any] = {
        # --- Regresja ---
        "XGBRegressor": {
            "n_estimators": [200, 400, 800, 1200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        "LGBMRegressor": {
            "n_estimators": [300, 600, 1000],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        "CatBoostRegressor": {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1],
            "iterations": [300, 600, 1000],
            "l2_leaf_reg": [1.0, 3.0, 5.0],
        },
        # --- Klasyfikacja ---
        "XGBClassifier": {
            "n_estimators": [200, 400, 800, 1200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        "LGBMClassifier": {
            "n_estimators": [300, 600, 1000],
            "num_leaves": [31, 63],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        "CatBoostClassifier": {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1],
            "iterations": [300, 600, 1000],
            "l2_leaf_reg": [1.0, 3.0, 5.0],
        },
    }
    canon = resolve_estimator(name)  # pozwala przyjąć alias i ustalić prawdziwą nazwę
    key = canon.__name__ if hasattr(canon, "__name__") else _normalize_name(name)
    return spaces.get(key, {})


# -------------------------------------------------------------------
# Diagnoza braków (wygodne pod UI / logi)
# -------------------------------------------------------------------

def missing_boosters_hints() -> Dict[str, str]:
    """
    Zwraca mapę {pakiet: 'pip install ...'} tylko dla brakujących.
    """
    hints: Dict[str, str] = {}
    if not has_xgboost():
        hints["xgboost"] = "pip install xgboost"
    if not has_lightgbm():
        hints["lightgbm"] = "pip install lightgbm"
    if not has_catboost():
        hints["catboost"] = "pip install catboost"
    return hints


# Public API
__all__ = [
    "has_xgboost",
    "has_lightgbm",
    "has_catboost",
    "get_versions",
    "get_regressor_classes",
    "get_classifier_classes",
    "list_available_estimators",
    "resolve_estimator",
    "create_estimator",
    "default_param_space",
    "missing_boosters_hints",
]
