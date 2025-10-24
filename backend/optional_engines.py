
from __future__ import annotations
from typing import Any, Dict
def _safe_import(name: str):
    try: module = __import__(name); return module
    except Exception: return None
def has_xgboost() -> bool: return _safe_import("xgboost") is not None
def has_lightgbm() -> bool: return _safe_import("lightgbm") is not None
def has_catboost() -> bool: return _safe_import("catboost") is not None
def get_regressor_classes() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if has_xgboost(): from xgboost import XGBRegressor; out["XGBRegressor"] = XGBRegressor
    if has_lightgbm(): from lightgbm import LGBMRegressor; out["LGBMRegressor"] = LGBMRegressor
    if has_catboost(): from catboost import CatBoostRegressor; out["CatBoostRegressor"] = CatBoostRegressor
    return out
def get_classifier_classes() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if has_xgboost(): from xgboost import XGBClassifier; out["XGBClassifier"] = XGBClassifier
    if has_lightgbm(): from lightgbm import LGBMClassifier; out["LGBMClassifier"] = LGBMClassifier
    if has_catboost(): from catboost import CatBoostClassifier; out["CatBoostClassifier"] = CatBoostClassifier
    return out
def default_param_space(name: str) -> Dict[str, Any]:
    spaces = {
        "XGBRegressor": {"n_estimators": [200, 400, 800, 1200], "max_depth": [3, 6, 10], "learning_rate": [0.05, 0.1]},
        "LGBMRegressor": {"n_estimators": [300, 600, 1000], "num_leaves": [31, 63, 127], "learning_rate": [0.05, 0.1]},
        "CatBoostRegressor": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "iterations": [300, 600, 1000]},
        "XGBClassifier": {"n_estimators": [200, 400, 800, 1200], "max_depth": [3, 6, 10], "learning_rate": [0.05, 0.1]},
        "LGBMClassifier": {"n_estimators": [300, 600, 1000], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
        "CatBoostClassifier": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "iterations": [300, 600, 1000]},
    }
    return spaces.get(name, {})
