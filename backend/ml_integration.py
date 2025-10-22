from __future__ import annotations
# -*- coding: utf-8 -*-
from backend.metrics_helper import normalize_metric

from backend.safe_utils import truthy_df_safe

# backend/ml_integration.py
from backend.utils_target import resolve_target_column

import warnings
warnings.filterwarnings("ignore")

import time
import hashlib
import json
from typing import Dict, List, Tuple, Any, Optional, Sequence, Callable

import numpy as np
import pandas as pd

# ================================
# Opcjonalne zależności i fallbacki
# ================================
# Streamlit (opcjonalny)
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

# Cache manager (opcjonalny)
try:
    from backend.cache_manager import smart_cache  # type: ignore
except Exception:
    class _DummyCache:
        def cache_decorator(self, ttl: int = 3600):
            def deco(fn):
                def wrapper(*a, **k): return fn(*a, **k)
                return wrapper
            return deco
        def get(self, key: str): return None
        def set(self, key: str, value: Any, ttl: int = 3600): return None
    smart_cache = _DummyCache()  # type: ignore

# Error handling / health monitor (opcjonalny)
try:
    from backend.error_handler import SmartErrorHandler, health_monitor, MLPlatformError, ErrorType  # type: ignore
except Exception:
    class SmartErrorHandler:  # type: ignore
        @staticmethod
        def training_error_handler(fn):
            def wrapper(*a, **k): return fn(*a, **k)
            return wrapper
    def health_monitor(*a, **k): return None  # type: ignore
    class MLPlatformError(Exception):  # type: ignore
        def __init__(self, *args, **kwargs): super().__init__(args[0] if args else "MLPlatformError")
    class ErrorType: USER_INPUT_ERROR = "USER_INPUT_ERROR"  # type: ignore

# Security / rate limiter (opcjonalny)
try:
    from backend.security_manager import rate_limiter  # type: ignore
except Exception:
    class _RateLimiter:  # type: ignore
        def is_allowed(self, key: str): return True
        def get_remaining_requests(self, key: str): return 0
    rate_limiter = _RateLimiter()  # type: ignore

# Asynchroniczny trainer (opcjonalny)
try:
    from backend.async_ml_trainer import AsyncMLTrainer, run_async_in_streamlit, StreamlitAsyncRunner  # type: ignore
except Exception:
    class AsyncMLTrainer:  # type: ignore
        pass
    def run_async_in_streamlit(fn):  # type: ignore
        return fn
    class StreamlitAsyncRunner:  # type: ignore
        pass

# Opcjonalne silniki (rozszerzenia modeli)
try:
    from backend.optional_engines import get_regressor_classes, get_classifier_classes, default_param_space  # type: ignore
except Exception:
    def get_regressor_classes(): return {}
    def get_classifier_classes(): return {}
    def default_param_space(name: str): return {}

# Plotly (wykresy)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sklearn – modelowanie
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    BaggingRegressor, BaggingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, RidgeClassifier, Lasso, ElasticNet,
    BayesianRidge, SGDRegressor, SGDClassifier,
    PassiveAggressiveRegressor, PassiveAggressiveClassifier,
    HuberRegressor, TheilSenRegressor, RANSACRegressor
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC, NuSVR, NuSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, RadiusNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, explained_variance_score, max_error, median_absolute_error
)

# Zaawansowane biblioteki – opcjonalne
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


# =========================================================
# ✅ AUTO DATA CLEANER – automatyczne czyszczenie + log kroków
# =========================================================
class AutoDataCleaner:
    """
    Automatyczne czyszczenie danych z raportowaniem kroków:
      1) duplikaty,
      2) konwersje dat,
      3) imputacja braków,
      4) wstępne kodowanie kategorii (one-hot do 20 uniq, dalej freq-encoding),
      5) podsumowanie kształtu.
    """
    def __init__(self):
        self.eda = None

    def _ensure_analyzer(self):
        try:
            if getattr(self, 'eda', None) is None:
                from backend.eda_integration import EDAAnalyzer
                self.eda = EDAAnalyzer()
        except Exception:
            class _Fallback:
                def detect_problem_type(self, df):
                    # naive heuristic fallback
                    import pandas as _pd
                    if 'target' in df.columns:
                        y = df['target']
                        return 'classification' if y.nunique()<=10 and _pd.api.types.is_integer_dtype(y) else 'regression'
                    return 'regression'
            self.eda = _Fallback()
        return self.eda

# GŁÓWNA KLASA: MLModelTrainer
# =========================================================
class MLModelTrainer:
    """
    Zaawansowana klasa do trenowania modeli:
    - szeroki katalog algorytmów (z boosterami, jeśli dostępne),
    - auto-clean + smart encoding,
    - CV + wybór najlepszego modelu,
    - metryki (reg/cls) + wykresy,
    - tryb asynchroniczny (jeśli backend na to pozwala),
    - cache i limity (fall-back safe).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.async_trainer = AsyncMLTrainer()  # no-op fallback, jeśli brak
        self.trained_models: Dict[str, Dict[str, Any]] = {}
        self.model_results: Dict[str, Any] = {}
        self.hyperparameter_search: bool = False

        self.cache = smart_cache
        self.health_monitor = health_monitor

        # Kontekst do inteligentnego wyboru algorytmów
        self._current_df: Optional[pd.DataFrame] = None
        self._current_target: Optional[str] = None

        # AutoCleaner
        self.auto_cleaner = AutoDataCleaner()

        # Algorytmy
        self.regression_algorithms = self._get_regression_algorithms()
        self.classification_algorithms = self._get_classification_algorithms()

    # -----------------------------
    # Katalog algorytmów
    # -----------------------------
    def _get_regression_algorithms(self) -> Dict[str, Any]:
        alg = {
            # Ensemble
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'Bagging': BaggingRegressor(random_state=42, n_jobs=-1),
            # Linear family
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Bayesian Ridge': BayesianRidge(),
            'Huber Regressor': HuberRegressor(),
            'RANSAC Regressor': RANSACRegressor(random_state=42),
            'Theil Sen': TheilSenRegressor(random_state=42),
            # Trees
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            # Neighbors
            'K-Neighbors': KNeighborsRegressor(),
            'Radius Neighbors': RadiusNeighborsRegressor(),
            # SVM
            'SVR (RBF)': SVR(kernel='rbf'),
            'SVR (Linear)': SVR(kernel='linear'),
            'Nu-SVR': NuSVR(kernel='rbf'),
            # Neural
            'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            # Gaussian Process
            'Gaussian Process': GaussianProcessRegressor(random_state=42),
            # SGD family
            'SGD Regressor': SGDRegressor(random_state=42),
            'Passive Aggressive': PassiveAggressiveRegressor(random_state=42),
        }
        if truthy_df_safe(XGBOOST_AVAILABLE):
            alg['XGBoost'] = xgb.XGBRegressor(n_estimators=200, random_state=42, eval_metric='rmse', verbosity=0)
        if truthy_df_safe(LIGHTGBM_AVAILABLE):
            alg['LightGBM'] = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbosity=-1)
        if truthy_df_safe(CATBOOST_AVAILABLE):
            alg['CatBoost'] = cb.CatBoostRegressor(iterations=200, random_state=42, verbose=False)
        return alg

    def _get_classification_algorithms(self) -> Dict[str, Any]:
        alg = {
            # Ensemble
            'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Bagging': BaggingClassifier(random_state=42, n_jobs=-1),
            # Linear family
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
            'Ridge Classifier': RidgeClassifier(random_state=42),
            'SGD Classifier': SGDClassifier(random_state=42),
            'Passive Aggressive': PassiveAggressiveClassifier(random_state=42),
            # Trees
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            # Neighbors
            'K-Neighbors': KNeighborsClassifier(),
            'Nearest Centroid': NearestCentroid(),
            # SVM (klasyfikacyjne)
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
            'Nu-SVM': NuSVC(random_state=42, probability=True),
            # Neural
            'Neural Network (MLP)': MLPClassifier(
                hidden_layer_sizes=(50, 25), random_state=42, max_iter=200,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
            ),
            # Naive Bayes
            'Gaussian Naive Bayes': GaussianNB(),
            'Bernoulli Naive Bayes': BernoulliNB(),
            # DA
            'Linear Discriminant': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant': QuadraticDiscriminantAnalysis(),
            # Gaussian Process
            'Gaussian Process': GaussianProcessClassifier(random_state=42),
        }
        if truthy_df_safe(XGBOOST_AVAILABLE):
            alg['XGBoost'] = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0)
        if truthy_df_safe(LIGHTGBM_AVAILABLE):
            alg['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1)
        if truthy_df_safe(CATBOOST_AVAILABLE):
            alg['CatBoost'] = cb.CatBoostClassifier(iterations=200, random_state=42, verbose=False)
        return alg

    def get_available_algorithms_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            'regression_count': len(self.regression_algorithms),
            'classification_count': len(self.classification_algorithms),
            'advanced_available': [],
            'advanced_missing': []
        }
        for name, flag in (('XGBoost', XGBOOST_AVAILABLE), ('LightGBM', LIGHTGBM_AVAILABLE), ('CatBoost', CATBOOST_AVAILABLE)):
            (info['advanced_available'] if flag else info['advanced_missing']).append(name)
        return info

    # -----------------------------
    # Inteligentny wybór algorytmów
    # -----------------------------
    def _get_algorithm_type(self, algorithm_name: str) -> str:
        n = algorithm_name.lower()
        if any(w in n for w in ['forest','boosting','adaboost','bagging','xgboost','lightgbm','catboost','extra trees']): return 'ensemble'
        if any(w in n for w in ['linear','logistic','ridge','lasso','elastic','sgd','passive']): return 'linear'
        if 'tree' in n: return 'tree'
        if any(w in n for w in ['mlp','neural','network']): return 'neural'
        if any(w in n for w in ['svm','svr','svc']): return 'svm'
        if 'neighbor' in n: return 'neighbor'
        if 'bayes' in n or 'naive' in n: return 'bayes'
        return 'other'

    def detect_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        y = df[target_column].dropna()
        if not pd.api.types.is_numeric_dtype(y): return "Klasyfikacja"
        un = int(y.nunique()); ratio = un / max(1, len(y))
        if un == 2: return "Klasyfikacja"
        if (2 < un <= 20 and ratio < 0.05) or (un <= 50 and ratio < 0.01): return "Klasyfikacja"
        return "Regresja"

    def suggest_target_column(self, df: pd.DataFrame) -> str:
        scores = []
        keywords = ["target","label","class","outcome","result","prediction","y","price","amount","value",
                    "score","rating","saleprice","response","churn","fraud","default","survived","diagnosis"]
        for col in df.columns:
            s = 0.0; name = col.lower()
            for k in keywords:
                if k in name: s += 3.0
            if col == df.columns[-1]: s += 1.5
            try:
                un = df[col].nunique(dropna=True)
                if un == 2: s += 2.5
                elif 2 < un <= 20: s += 1.5
            except Exception: pass
            try:
                miss = df[col].isna().mean()
                if miss > 0.3: s -= 2.0
                elif miss > 0.1: s -= 1.0
            except Exception: pass
            if np.issubdtype(df[col].dtype, np.datetime64): s -= 3.0
            scores.append((col, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else df.columns[-1]

    def _intelligent_algorithm_selection(self, df: pd.DataFrame, target_column: str, strategy: str) -> Dict[str, Any]:
        n = len(df)
        y = df[target_column]
        problem = self.detect_problem_type(df, target_column)
        imbalance_ratio = 1.0
        if problem == "Klasyfikacja":
            vc = y.value_counts()
            if len(vc) > 1: imbalance_ratio = vc.max() / vc.min()

        # Strategie
        if strategy == "fast":
            return self._get_speed_algorithms(problem, n)
        if strategy == "accurate":
            return self._get_accuracy_algorithms(problem, n, imbalance_ratio)
        if strategy in ("ensemble","advanced"):
            return self._get_accuracy_algorithms(problem, n, imbalance_ratio)
        # "all" – redukcja dla dużych zbiorów
        if n > 10000:
            return self._get_speed_algorithms(problem, n)
        return self.classification_algorithms if problem == "Klasyfikacja" else self.regression_algorithms

    def _get_speed_algorithms(self, problem: str, n: int) -> Dict[str, Any]:
        if problem == "Klasyfikacja":
            base = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
                'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Ridge Classifier': RidgeClassifier(random_state=42)
            }
            if n > 5000:
                base.update({
                    'Random Forest (Fast)': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
                    'Extra Trees (Fast)': ExtraTreesClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
                })
            if truthy_df_safe(LIGHTGBM_AVAILABLE):
                base['LightGBM (Fast)'] = lgb.LGBMClassifier(n_estimators=50, num_leaves=31, random_state=42, verbosity=-1)
            return base
        # Regresja
        base = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        if n > 5000:
            base.update({
                'Random Forest (Fast)': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
                'Extra Trees (Fast)': ExtraTreesRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            })
        if truthy_df_safe(LIGHTGBM_AVAILABLE):
            base['LightGBM (Fast)'] = lgb.LGBMRegressor(n_estimators=50, num_leaves=31, random_state=42, verbosity=-1)
        return base

    def _get_accuracy_algorithms(self, problem: str, n: int, imbalance_ratio: float) -> Dict[str, Any]:
        if problem == "Klasyfikacja":
            accurate = {
                'Random Forest': RandomForestClassifier(n_estimators=200 if n < 10000 else 100, random_state=42, n_jobs=-1),
                'Extra Trees': ExtraTreesClassifier(n_estimators=200 if n < 10000 else 100, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            if truthy_df_safe(XGBOOST_AVAILABLE):
                accurate['XGBoost'] = xgb.XGBClassifier(n_estimators=150 if n < 10000 else 100, random_state=42, eval_metric='logloss', verbosity=0)
            if truthy_df_safe(LIGHTGBM_AVAILABLE):
                accurate['LightGBM'] = lgb.LGBMClassifier(n_estimators=150 if n < 10000 else 100, random_state=42, verbosity=-1)
            if truthy_df_safe(CATBOOST_AVAILABLE):
                accurate['CatBoost'] = cb.CatBoostClassifier(iterations=100 if n < 10000 else 80, random_state=42, verbose=False)
            if imbalance_ratio < 5.0 and n < 5000:
                accurate['SVM (RBF)'] = SVC(kernel='rbf', random_state=42, probability=True, C=1.0)
            return accurate
        # Regresja
        accurate = {
            'Random Forest': RandomForestRegressor(n_estimators=200 if n < 10000 else 100, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(n_estimators=200 if n < 10000 else 100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        if truthy_df_safe(XGBOOST_AVAILABLE):
            accurate['XGBoost'] = xgb.XGBRegressor(n_estimators=150 if n < 10000 else 100, random_state=42, eval_metric='rmse', verbosity=0)
        if truthy_df_safe(LIGHTGBM_AVAILABLE):
            accurate['LightGBM'] = lgb.LGBMRegressor(n_estimators=150 if n < 10000 else 100, random_state=42, verbosity=-1)
        return accurate

    # -----------------------------
    # Przygotowanie danych
    # -----------------------------
    def prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        # Auto-clean
        try:
            df_clean = self.auto_cleaner.clean(df)
            if st is not None:
                st.session_state["auto_clean_report"] = self.auto_cleaner.get_report()
                st.session_state["df_clean"] = df_clean.copy()
        except Exception:
            df_clean = df.copy()

        df_processed = df_clean.copy()
        if target_column not in df_processed.columns:
            suggested = self.suggest_target_column(df_processed)
            if st is not None:
                st.warning(f"Wybrana kolumna celu '{target_column}' nie istnieje. Używam: '{suggested}'.")
            target_column = suggested

        X = df_processed.drop(columns=[target_column], errors="ignore")
        y = df_processed[target_column]

        # Datetime -> znacznik czasu
        dt_cols = X.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]"]).columns.tolist()
        for c in dt_cols:
            try:
                ser = pd.to_datetime(X[c], errors="coerce")
                X[f"{c}__ts"] = ser.view("int64")
            except Exception:
                pass
        X = X.drop(columns=dt_cols, errors="ignore")

        # LabelEncoder (stabilny)
        categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
        for col in categorical_cols:
            try:
                if X[col].dtype.name == 'category': X[col] = X[col].astype('object')
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = X[col].fillna('missing').astype(str)
                    X[col] = self.label_encoders[col].fit_transform(X[col])
                else:
                    X[col] = X[col].fillna('missing').astype(str)
                    known = set(self.label_encoders[col].classes_)
                    fallback = list(self.label_encoders[col].classes_)[0]
                    X[col] = X[col].apply(lambda v: v if v in known else fallback)
                    X[col] = self.label_encoders[col].transform(X[col])
            except Exception:
                X[col] = X[col].astype(str).apply(lambda v: abs(hash(v)) % 100000)

        # Numeryczne -> mediana
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isna().any():
                try:
                    med = float(np.nanmedian(X[col].astype(float)))
                    X[col] = X[col].fillna(med)
                except Exception:
                    X[col] = X[col].fillna(0)

        # Usuń NaN-y w y
        valid = ~y.isnull()
        X, y = X.loc[valid].copy(), y.loc[valid].copy()
        return X, y

    def prepare_data_with_context(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        self._current_df = df.copy()
        self._current_target = target_column
        return self.prepare_data(df, target_column)

    # -----------------------------
    # Główny trening (synchroniczny)
    # -----------------------------
    def _smart_encode_categoricals(self, X: pd.DataFrame, y: Optional[pd.Series], problem_type: str) -> pd.DataFrame:
        import hashlib as _h
        X_enc = X.copy()
        cat_cols = X_enc.select_dtypes(include=['object','category']).columns.tolist()
        # małe – one-hot
        small = [c for c in cat_cols if X_enc[c].nunique(dropna=True) <= 30]
        if truthy_df_safe(small):
            X_enc = pd.get_dummies(X_enc, columns=small, drop_first=True)
        # średnie – freq
        medium = [c for c in cat_cols if c in X_enc.columns and 30 < X_enc[c].nunique(dropna=True) <= 100]
        for c in medium:
            freq = X_enc[c].value_counts(dropna=False)
            X_enc[c] = X_enc[c].map(freq).astype(float)
        # duże – wielohash
        high = [c for c in cat_cols if c in X_enc.columns and X_enc[c].nunique(dropna=False) > 100]
        for c in high:
            for i in range(16):
                X_enc[f"{c}__h{i}"] = X_enc[c].apply(
                    lambda v, ii=i: ((int(_h.md5(str(v).encode('utf-8')).hexdigest(), 16) + ii) % 997) / 997.0
                )
            X_enc = X_enc.drop(columns=[c])
        return X_enc

    def _detect_imbalance(self, y: pd.Series) -> tuple[bool, float]:
        vc = y.value_counts(dropna=False)
        if len(vc) < 2: return (False, 1.0)
        ratio = vc.min() / vc.max()
        return (ratio < 0.2, float(ratio))

    def _balance_classes(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict]:
        info = {"method": "none"}
        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            Xb, yb = ros.fit_resample(X, y)
            info["method"] = "RandomOverSampler"
            return Xb, yb, info
        except Exception:
            vc = y.value_counts()
            maxn = vc.max()
            parts_X, parts_y = [], []
            for cls, n in vc.items():
                Xi = X[y == cls]; yi = y[y == cls]
                reps = max(1, int(maxn // max(1, n)))
                parts_X.append(pd.concat([Xi]*reps, ignore_index=True))
                parts_y.append(pd.concat([yi]*reps, ignore_index=True))
            return pd.concat(parts_X, ignore_index=True), pd.concat(parts_y, ignore_index=True), {"method":"simple_ros"}

    def train_model(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_size: float = 0.8,
        cv_folds: int = 5,
        random_state: int = 42,
        use_full_dataset: bool = False,
        algorithm_selection: str = "all",
        enable_hyperparameter_tuning: bool = False,
        optimization_metric: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        problem_type = self.detect_problem_type(df, target_column)
        X, y = self.prepare_data_with_context(df, target_column)
        X = self._smart_encode_categoricals(X, y, problem_type)

        # Redukcja dla bardzo dużych zbiorów
        if len(X) > 15000 and not use_full_dataset:
            if st is not None:
                st.info(f"Duży zbiór danych ({len(X)}). Używam próbki 8000 dla szybszego treningu…")
            X, _, y, _ = train_test_split(
                X, y, train_size=8000 / len(X), random_state=random_state,
                stratify=y if (problem_type == "Klasyfikacja" and y.nunique() < 50) else None
            )

        # Balans klas
        if problem_type == "Klasyfikacja":
            is_imb, _ratio = self._detect_imbalance(y)
            if truthy_df_safe(is_imb):
                X, y, _ = self._balance_classes(X, y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - train_size, random_state=random_state,
            stratify=y if (problem_type == "Klasyfikacja" and y.nunique() > 1 and y.value_counts().min() >= 2) else None
        )

        # Skalowanie
        self.scaler = StandardScaler()
        pre, cols = _build_preprocessor(df=df, target=target, X=X_train)
        X_train_scaled = pre.fit_transform(X_train) if pre is not None else X_train
        X_test_scaled = self.scaler.transform(X_test)

        # Wybór algorytmów
        algorithms = self._intelligent_algorithm_selection(
            self._current_df if self._current_df is not None else X.assign(_tmp=y),
            self._current_target or (target_column if target_column in df.columns else y.name),
            algorithm_selection
        )

        if st is not None:
            st.info(f"Trenuję {len(algorithms)} algorytmów dla problemu: {problem_type}")
            progress_bar = st.progress(0)
            progress_text = st.empty()
        else:
            progress_bar = None
            progress_text = None

        best_model: Optional[Any] = None
        best_score = -np.inf
        best_model_name = ""
        model_scores: Dict[str, Dict[str, float]] = {}

        slow_algorithms = {'Neural Network (MLP)', 'SVM (RBF)', 'Gaussian Process', 'Nu-SVM'}
        medium_algorithms = {'Random Forest','Extra Trees','XGBoost','LightGBM','CatBoost'}

        for idx, (name, model) in enumerate(algorithms.items()):
            try:
                if progress_bar is not None:
                    progress_bar.progress((idx + 1) / len(algorithms))
                if progress_text is not None:
                    if name in slow_algorithms:
                        progress_text.text(f"Trenuję wolny model: {name} ({idx + 1}/{len(algorithms)})…")
                    elif any(k in name for k in medium_algorithms):
                        progress_text.text(f"Trenuję ensemble: {name} ({idx + 1}/{len(algorithms)})…")
                    else:
                        progress_text.text(f"Trenuję szybki model: {name} ({idx + 1}/{len(algorithms)})")

                start_time = time.time()

                # Szybkie CV
                if problem_type == "Klasyfikacja":
                    min_class_size = int(y_train.value_counts().min()) if y_train.nunique() > 1 else cv_folds
                    if min_class_size < cv_folds:
                        X_vtr, X_vte, y_vtr, y_vte = train_test_split(
                            X_train_scaled, y_train, test_size=0.2, random_state=random_state,
                            stratify=y_train if min_class_size >= 2 else None
                        )
                        model.fit(X_vtr, y_vtr)
                        y_vpred = model.predict(X_vte)
                        mean_cv_score = accuracy_score(y_vte, y_vpred)
                        cv_std = 0.0
                    else:
                        actual_cv = min(cv_folds, min_class_size)
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=actual_cv, scoring='accuracy')
                        mean_cv_score, cv_std = cv_scores.mean(), cv_scores.std()
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                    mean_cv_score, cv_std = cv_scores.mean(), cv_scores.std()

                elapsed_time = time.time() - start_time
                model_scores[name] = {
                    'cv_score': float(mean_cv_score),
                    'cv_std': float(cv_std),
                    'algorithm_type': self._get_algorithm_type(name),
                    'training_time': round(elapsed_time, 2)
                }

                # Fit finalny + predykcja test
                model.fit(X_train_scaled, y_train)
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_model = model
                    best_model_name = name

                if progress_text is not None:
                    if problem_type == "Klasyfikacja":
                        progress_text.text(f"{name}: Acc={mean_cv_score:.3f} (±{cv_std:.3f}) — {elapsed_time:.1f}s")
                    else:
                        progress_text.text(f"{name}: R²={mean_cv_score:.3f} (±{cv_std:.3f}) — {elapsed_time:.1f}s")

                time.sleep(0.2)

            except Exception as e:
                if st is not None:
                    st.warning(f"Błąd podczas trenowania {name}: {str(e)[:100]}…")
                continue

        if progress_bar is not None: progress_bar.empty()
        if progress_text is not None: progress_text.empty()

        if best_model is None:
            raise ValueError("Nie udało się wytrenować żadnego modelu")

        y_pred = best_model.predict(self.scaler.transform(X_test))

        results: Dict[str, Any] = {
            'best_model': best_model_name,
            'model_scores': model_scores,
            'predictions': y_pred,
            'actual': y_test.values,
            'feature_names': list(X.columns),
            'label_encoders': self.label_encoders.copy(),
            'algorithm_info': self.get_available_algorithms_info(),
            'auto_clean_report': (st.session_state.get("auto_clean_report", "") if st is not None else "")
        }

        if problem_type == "Klasyfikacja":
            results.update(self._classification_metrics(y_test, y_pred, best_model, self.scaler.transform(X_test)))
        else:
            results.update(self._regression_metrics(y_test, y_pred))

        fi_df = self._feature_importance(best_model, X.columns)
        if fi_df is not None:
            results['feature_importance'] = fi_df

        self.trained_models[target_column] = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders.copy(),
            'problem_type': problem_type,
            'feature_names': list(X.columns)
        }
        return results

    # -----------------------------
    # Metryki + Feature Importance + Wykresy
    # -----------------------------
    def _classification_metrics(self, y_true, y_pred, model, X_test_scaled):
        out = {}
        try:
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
            out["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            out["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
        except Exception: pass
        # ROC-AUC jeżeli proby
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_scaled)
                if proba.ndim == 1 or proba.shape[1] == 2:
                    proba1 = proba if proba.ndim == 1 else proba[:, 1]
                    out["roc_auc"] = float(roc_auc_score(y_true, proba1))
                else:
                    out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))
            elif hasattr(model, "decision_function"):
                dfv = model.decision_function(X_test_scaled)
                if dfv.ndim == 1:
                    out["roc_auc"] = float(roc_auc_score(y_true, dfv))
                else:
                    out["roc_auc"] = float(roc_auc_score(y_true, dfv, multi_class="ovr"))
        except Exception: pass
        return out

    def _regression_metrics(self, y_true, y_pred):
        out = {}
        try: out["r2"] = float(r2_score(y_true, y_pred))
        except Exception: pass
        try: out["mae"] = float(mean_absolute_error(y_true, y_pred))
        except Exception: pass
        try: out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        except Exception: pass
        try:
            out["mape"] = float((np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))).mean() * 100.0)
        except Exception: pass
        try:
            denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
            out["smape"] = float((np.mean(np.abs(y_true - y_pred) / np.clip(denom, 1e-9, None))) * 100.0)
        except Exception: pass
        try: out["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        except Exception: pass
        try: out["median_ae"] = float(median_absolute_error(y_true, y_pred))
        except Exception: pass
        try: out["max_error"] = float(max_error(y_true, y_pred))
        except Exception: pass
        return out

    def _feature_importance(self, model, feature_names):
        try:
            if hasattr(model, "feature_importances_"):
                fi = np.asarray(model.feature_importances_, dtype=float)
            elif hasattr(model, "coef_"):
                coef = model.coef_
                fi = np.abs(coef.ravel() if hasattr(coef, "ravel") else np.array(coef)).astype(float)
            else:
                return None
            fi = np.nan_to_num(fi)
            order = np.argsort(fi)[::-1]
            feats = np.array(list(feature_names))[order]
            vals = fi[order]
            total = np.clip(vals.sum(), 1e-12, None)
            pct = (vals / total) * 100.0
            cum = np.cumsum(pct)
            return pd.DataFrame({
                "feature": feats,
                "importance": vals,
                "importance_pct": pct,
                "cumulative_importance": cum
            })
        except Exception:
            return None

    def create_advanced_visualizations(self, results: dict, problem_type: str):
        figs = {}
        try:
            ms = results.get("model_scores")
            if isinstance(ms, dict) and len(ms) > 0:
                rows = [{"model": n, "cv_score": v.get("cv_score"), "time": v.get("training_time")} for n, v in ms.items()]
                dfm = pd.DataFrame(rows)
                if not dfm.empty:
                    figs["model_comparison"] = px.bar(
                        dfm.sort_values("cv_score", ascending=False),
                        x="model", y="cv_score", title="Porównanie modeli (CV)", text="cv_score"
                    )
            if "feature_importance" in results and hasattr(results["feature_importance"], "head"):
                df_fi = results["feature_importance"].head(25)
                figs["feature_importance_advanced"] = px.bar(
                    df_fi, x="importance", y="feature", orientation="h", title="Najważniejsze cechy"
                )
            if "predictions" in results and "actual" in results:
                yhat = np.asarray(results["predictions"]); y = np.asarray(results["actual"])
                if problem_type == "Regresja":
                    fig = px.scatter(x=y, y=yhat, labels={"x":"Rzeczywiste","y":"Predykcje"},
                                     title="Rzeczywiste vs Predykcje")
                    fig.add_shape(type="line", x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max(), line=dict(dash="dash"))
                    figs["predictions_plot"] = fig
                else:
                    try:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y, yhat)
                        figs["confusion_matrix"] = px.imshow(cm, text_auto=True, title="Macierz pomyłek")
                    except Exception:
                        pass
        except Exception:
            pass
        return figs

    # -----------------------------
    # Tryb asynchroniczny (jeśli dostępny)
    # -----------------------------
    @run_async_in_streamlit
    @SmartErrorHandler.training_error_handler
    async def train_model_async(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_size: float = 0.8,
        cv_folds: int = 5,
        random_state: int = 42,
        use_full_dataset: bool = False,
        algorithm_selection: str = "all",
        enable_hyperparameter_tuning: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        allowed = rate_limiter.is_allowed('training')
        if not truthy_df_safe(allowed):
            remaining = rate_limiter.get_remaining_requests('training')
            msg = f"Osiągnięto limit operacji. Pozostało: {remaining} w tym oknie czasowym."
            if st is not None: st.error(f"⏳ {msg}")
            raise MLPlatformError(ErrorType.USER_INPUT_ERROR, msg)
        # uruchamiamy synchroniczny wariant wewnątrz async (fallback-safe)
        return self.train_model(
            df=df, target_column=target_column, train_size=train_size, cv_folds=cv_folds,
            random_state=random_state, use_full_dataset=use_full_dataset,
            algorithm_selection=algorithm_selection, enable_hyperparameter_tuning=enable_hyperparameter_tuning,
            **kwargs
        )

    # -----------------------------
    # Szybkie rekomendacje planu
    # -----------------------------
    def suggest_training_plan(self, df: pd.DataFrame, target_column: str) -> dict:
        if target_column not in df.columns:
            suggested = self.suggest_target_column(df)
            if st is not None:
                st.warning(f"Wybrana kolumna celu '{target_column}' nie istnieje. Używam: '{suggested}'.")
            target_column = suggested
        problem_type = self.detect_problem_type(df, target_column)
        n, _ = df.shape
        y = df[target_column]
        X = df.drop(columns=[target_column], errors='ignore')

        rec = {
            "train_size": 0.8 if n < 50000 else 0.9,
            "cv_folds": 5 if n < 10000 else 3,
            "metric": "f1_weighted" if (problem_type == "Klasyfikacja" and y.nunique() > 2) else ("roc_auc" if problem_type == "Klasyfikacja" else "rmse"),
        }
        cat_cols = X.select_dtypes(include=['object','category']).shape[0]
        rec["note"] = "Wykryto zmienne kategoryczne — włączono smart encoding." if cat_cols > 0 else "Zbiór głównie numeryczny — zalecane GB/Forest."
        if problem_type == "Klasyfikacja":
            is_imb, ratio = self._detect_imbalance(y)
            rec["is_imbalanced"] = is_imb
            rec["imbalance_ratio"] = ratio
            if is_imb: rec["class_balance"] = "auto (class_weight/ROS)"
            rec["algorithms"] = ["HistGB", "LightGBM", "XGBoost", "Random Forest", "LogisticRegression"]
        else:
            rec["algorithms"] = ["HistGB", "LightGBM", "XGBoost", "Random Forest", "Ridge"]
        return rec

# === Część 2/2 — dodatki: ensembles, szybka ewaluacja, telemetria, artefakty ===

import os
import csv
import shutil

# joblib (fallback)
try:
    import joblib
except Exception:
    joblib = None  # type: ignore


# ---------------------------------------------------------
# Telemetria: lekki log do CSV (bezpieczny fallback)
# ---------------------------------------------------------
_TELEMETRY_DIR = os.path.join("artifacts", "telemetry")
os.makedirs(_TELEMETRY_DIR, exist_ok=True)
_TELEMETRY_FILE = os.path.join(_TELEMETRY_DIR, "runlog.csv")

def telemetry_log_ml(event: str, detail: str = "", model: str = "", fold: int = 0):
    """
    Bardzo lekki logger CSV: timestamp, event, model, fold, detail.
    Ignoruje błędy zapisu (nie blokuje treningu).
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, event, model, fold, detail]
    try:
        new = not os.path.exists(_TELEMETRY_FILE)
        with open(_TELEMETRY_FILE, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if truthy_df_safe(new):
                w.writerow(["timestamp", "event", "model", "fold", "detail"])
            w.writerow(row)
    except Exception:
        pass


# ---------------------------------------------------------
# Mapowanie nazw → klasy modeli (leniwe, z boosterami)
# ---------------------------------------------------------
def _lazy_model_map() -> Dict[str, Any]:
    mapping: Dict[str, Any] = {
        # klasyfikacja
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
        "SVC": SVC,
        "KNeighborsClassifier": KNeighborsClassifier,

        # regresja
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "LinearRegression": LinearRegression,
        "SVR": SVR,
        "KNeighborsRegressor": KNeighborsRegressor,
    }
    # opcjonalne boostery
    if truthy_df_safe(XGBOOST_AVAILABLE):
        mapping["XGBClassifier"] = xgb.XGBClassifier
        mapping["XGBRegressor"] = xgb.XGBRegressor
    if truthy_df_safe(LIGHTGBM_AVAILABLE):
        mapping["LGBMClassifier"] = lgb.LGBMClassifier
        mapping["LGBMRegressor"] = lgb.LGBMRegressor
    if truthy_df_safe(CATBOOST_AVAILABLE):
        mapping["CatBoostClassifier"] = cb.CatBoostClassifier
        mapping["CatBoostRegressor"] = cb.CatBoostRegressor
    return mapping


# ---------------------------------------------------------
# Funkcje metryk — jedna funkcja generatora
# ---------------------------------------------------------
def _metric_factory(problem: str, metric_name: str) -> Tuple[str, Callable]:
    """
    Zwraca (direction, metric_fn). direction ∈ {"higher","lower","higher_proba"}.
    metric_fn: callable (y_true, y_pred) lub dla _proba: (y_true, y_proba).
    """
    if problem == "classification":
        def f1w(y_true, y_pred): return f1_score(y_true, y_pred, average="weighted")
        def precw(y_true, y_pred): return precision_score(y_true, y_pred, average="weighted", zero_division=0)
        def recw(y_true, y_pred): return recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics = {
            "accuracy": ("higher", accuracy_score),
            "f1_weighted": ("higher", f1w),
            "precision_weighted": ("higher", precw),
            "recall_weighted": ("higher", recw),
        }
        def roc_ovr(y_true, y_proba):
            if y_proba.ndim == 1:
                return roc_auc_score(y_true, y_proba)
            return roc_auc_score(y_true, y_proba, multi_class="ovr")
        metrics["roc_auc_ovr"] = ("higher_proba", roc_ovr)
        return metrics[metric_name]
    else:
        def rmse(y, yhat): return mean_squared_error(y, yhat, squared=False)
        def mape(y, yhat): return (np.abs((y - yhat) / np.clip(np.abs(y), 1e-9, None))).mean() * 100.0
        def smape(y, yhat):
            denom = (np.abs(y) + np.abs(yhat)) / 2.0
            return (np.mean(np.abs(y - yhat) / np.clip(denom, 1e-9, None))) * 100.0
        metrics = {
            "r2": ("higher", r2_score),
            "rmse": ("lower", rmse),
            "mae": ("lower", mean_absolute_error),
            "mape": ("lower", mape),
            "smape": ("lower", smape),
        }
        return metrics[metric_name]


# ---------------------------------------------------------
# Szybkie trenowanie kilku podstawowych modeli (CV)
# ---------------------------------------------------------
def train_multi_models_basic(
    df: pd.DataFrame,
    target: str,
    problem: Optional[str] = None,
    metric: Optional[str] = None,
    cv_folds: int = 5,
    progress_cb: Optional[Callable[[str], None]] = None
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Minimalistyczny, szybki benchmark kilku modeli bazowych.
    Zwraca: (results_df, best_dict)
    """
    assert target in df.columns, "Target column not found"
    X = df.drop(columns=[target])
    y = df[target]

    if problem not in ("classification", "regression"):
        problem = "classification" if (y.dtype == "O" or y.nunique() <= 20) else "regression"

    # Kandydaci
    models: List[Tuple[str, Any, Dict[str, Any]]] = []
    if problem == "classification":
        metric = metric or "accuracy"
        models = [
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42), {}),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42), {}),
            ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42), {}),
            ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=42), {}),
        ]
        cv = __import__("sklearn.model_selection").model_selection.StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=42
        )
    else:
        metric = metric or "r2"
        models = [
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42), {}),
            ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42), {}),
            ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=42), {}),
            ("LinearRegression", LinearRegression(), {}),
        ]
        cv = __import__("sklearn.model_selection").model_selection.KFold(
            n_splits=cv_folds, shuffle=True, random_state=42
        )

    direction, metric_fn = _metric_factory(problem, metric)
    rows = []
    best: Optional[Dict[str, Any]] = None
    is_minimize = (direction == "lower")

    if truthy_df_safe(progress_cb):
        progress_cb(f"🔎 Cross-validation ({cv_folds}-fold), metric={metric}")

    for name, model, params in models:
        telemetry_log_ml("model_start", model=name)
        try:
            scores = []
            for tr, va in cv.split(X, y):
                Xtr, Xva = X.iloc[tr], X.iloc[va]
                ytr, yva = y.iloc[tr], y.iloc[va]
                model.fit(Xtr, ytr)
                if direction == "higher_proba" and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xva)
                    s = metric_fn(yva, proba if proba.ndim > 1 else proba[:, 1])
                else:
                    pred = model.predict(Xva)
                    s = metric_fn(yva, pred)
                scores.append(float(s))
            mean, std = float(np.mean(scores)), float(np.std(scores))
            rows.append({"model": name, "metric": metric, "cv_mean": mean, "cv_std": std, "params": params})
            telemetry_log_ml("model_done", detail=f"{metric}={mean:.5f}", model=name)
            if truthy_df_safe(progress_cb):
                progress_cb(f"• {name}: {metric}={mean:.5f} ± {std:.5f}")

            if best is None:
                best = {"model": name, "params": params, "metric": metric, "cv_mean": mean, "cv_std": std}
            else:
                if (not is_minimize and mean > best["cv_mean"]) or (is_minimize and mean < best["cv_mean"]):
                    best = {"model": name, "params": params, "metric": metric, "cv_mean": mean, "cv_std": std}

        except Exception as e:
            telemetry_log_ml("model_error", detail=str(e)[:200], model=name)
            if truthy_df_safe(progress_cb):
                progress_cb(f"• {name}: pominięty ({e})")

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values("cv_mean", ascending=is_minimize).reset_index(drop=True)
    return results_df, best


# ---------------------------------------------------------
# Budowa prostych ensemble z top modeli
# ---------------------------------------------------------
def fit_ensembles_from_top(
    X: pd.DataFrame,
    y: pd.Series,
    top_rows: List[Dict[str, Any]],
    problem: str = "regression",
    max_members: int = 3
) -> Dict[str, Any]:
    """
    Składa Voting* z 2–3 najlepszych modeli z listy top_rows (jak z train_multi_models_basic).
    Zwraca: {"VotingClassifier" | "VotingRegressor": fitted_model}
    """
    if not truthy_df_safe(top_rows):
        return {}

    mapping = _lazy_model_map()
    usable = []
    for row in sorted(top_rows, key=lambda r: r.get("cv_mean", -np.inf), reverse=True):
        mname = row.get("model")
        if mname in mapping:
            usable.append((mname, mapping[mname], row.get("params", {}) or {}))
        if len(usable) >= max_members:
            break

    if not truthy_df_safe(usable):
        return {}

    members = [(f"m{i+1}", cls(**params)) for i, (name, cls, params) in enumerate(usable)]
    ensembles: Dict[str, Any] = {}
    try:
        if problem == "classification":
            from sklearn.ensemble import VotingClassifier
            vc = VotingClassifier(estimators=members, voting="soft")
            vc.fit(X, y)
            ensembles["VotingClassifier"] = vc
        else:
            from sklearn.ensemble import VotingRegressor
            vr = VotingRegressor(estimators=members)
            vr.fit(X, y)
            ensembles["VotingRegressor"] = vr
    except Exception as e:
        telemetry_log_ml("ensemble_error", detail=str(e)[:200])
    return ensembles


# ---------------------------------------------------------
# 3-fold szybka ewaluacja już złożonych ensemble’ów
# ---------------------------------------------------------
def evaluate_ensembles_cv(
    X: pd.DataFrame,
    y: pd.Series,
    ensembles: Dict[str, Any],
    problem: str = "regression",
    folds: int = 3
) -> pd.DataFrame:
    if not truthy_df_safe(ensembles):
        return pd.DataFrame()

    if problem == "classification":
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scoring = "r2"

    rows = []
    for name, model in ensembles.items():
        try:
            sc = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            rows.append({"model": name, "metric": scoring, "cv_mean": float(np.mean(sc)), "cv_std": float(np.std(sc))})
        except Exception as e:
            rows.append({"model": name, "metric": scoring, "cv_mean": np.nan, "cv_std": np.nan, "error": str(e)[:200]})
    return pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------
# Zapis artefaktów jednego run’u
# ---------------------------------------------------------
def save_artifacts_run(
    best_model: Any,
    df_prepared: pd.DataFrame,
    target: str,
    plan: Optional[Dict[str, Any]] = None,
    recs: Optional[Dict[str, Any]] = None,
    results_df: Optional[pd.DataFrame] = None,
    ensembles: Optional[Dict[str, Any]] = None,
    base_dir: str = "artifacts"
) -> str:
    """
    Tworzy katalog z timestampem i zapisuje:
      - model.joblib (jeśli dostępny joblib),
      - plan.json, recs.json,
      - columns.json (lista kolumn po przygotowaniu),
      - results.csv (jeśli podano),
      - ensembles/*.joblib (jeśli istnieją).
    Zwraca ścieżkę do katalogu run’u.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    # symlink do ostatniego run’u (best-effort)
    last_link = os.path.join(base_dir, "last_run")
    try:
        if os.path.islink(last_link) or os.path.exists(last_link):
            if os.path.islink(last_link):
                os.unlink(last_link)
            else:
                shutil.rmtree(last_link)
        # na Windows symlink może wymagać uprawnień — ignorujemy błędy
        os.symlink(os.path.abspath(run_dir), last_link, target_is_directory=True)
    except Exception:
        pass

    # model
    try:
        if best_model is not None and joblib is not None:
            joblib.dump(best_model, os.path.join(run_dir, "model.joblib"))
    except Exception:
        pass

    # meta
    try:
        with open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan or {}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "recs.json"), "w", encoding="utf-8") as f:
            json.dump(recs or {}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, "columns.json"), "w", encoding="utf-8") as f:
            cols = [c for c in df_prepared.columns if c != target]
            json.dump({"target": target, "feature_columns": cols}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # results
    if results_df is not None:
        try:
            results_df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
        except Exception:
            pass

    # ensembles
    if truthy_df_safe(ensembles) and joblib is not None:
        ens_dir = os.path.join(run_dir, "ensembles")
        os.makedirs(ens_dir, exist_ok=True)
        for name, model in ensembles.items():
            try:
                joblib.dump(model, os.path.join(ens_dir, f"{name}.joblib"))
            except Exception:
                pass

    return run_dir

# === Backward-compat aliases (przywrócenie starych nazw API) ==================
# Jeśli w tej wersji pliku funkcje mają nowe nazwy, mapujemy je na stare.

# save_artifacts → save_artifacts_run
try:
    save_artifacts  # noqa: F401
except NameError:
    def save_artifacts(*args, **kwargs):
        return save_artifacts_run(*args, **kwargs)

# fit_ensembles → fit_ensembles_from_top
try:
    fit_ensembles  # noqa: F401
except NameError:
    def fit_ensembles(*args, **kwargs):
        return fit_ensembles_from_top(*args, **kwargs)

# evaluate_models_quick → evaluate_ensembles_cv
try:
    evaluate_models_quick  # noqa: F401
except NameError:
    def evaluate_models_quick(*args, **kwargs):
        return evaluate_ensembles_cv(*args, **kwargs)

# train_multi_models → train_multi_models_basic
try:
    train_multi_models  # noqa: F401
except NameError:
    def train_multi_models(*args, **kwargs):
        return train_multi_models_basic(*args, **kwargs)

# (opcjonalnie) telemetry_log alias, jeśli gdzieś oczekujesz starej nazwy
try:
    telemetry_log  # noqa: F401
except NameError:
    def telemetry_log(event: str, detail: str = "", model: str = "", fold: int = 0):
        return telemetry_log_ml(event=event, detail=detail, model=model, fold=fold)

# Uporządkuj eksporty
try:
    __all__
except NameError:
    __all__ = []
__all__ += [
    "MLModelTrainer",
    "save_artifacts", "fit_ensembles", "evaluate_models_quick", "train_multi_models",
    # nowe nazwy też zostawiamy eksportowane:
    "save_artifacts_run", "fit_ensembles_from_top", "evaluate_ensembles_cv", "train_multi_models_basic",
]
# ==============================================================================


# --- injected by upgrader: stable feature importance (OneHot + RF) ---
def _infer_task(y: pd.Series) -> str:
    if y.dtype.kind in ("b","O","U") or y.nunique() <= 20:
        return "classification"
    return "regression"

def compute_feature_importance(df: pd.DataFrame, target: str, task: str = "auto") -> pd.DataFrame:
    y = df[target]
    X = df.drop(columns=[target])
    for c in X.columns:
        if pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("string").fillna("NA")
    cat = X.select_dtypes(include=["object","category","bool","string"]).columns.tolist()
    num = X.select_dtypes(exclude=["object","category","bool","string"]).columns.tolist()
    pre = ColumnTransformer([("num","passthrough", num),
                             ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)],
                            remainder="drop", verbose_feature_names_out=False)
    if task == "auto":
        task = _infer_task(y)
    model = RandomForestClassifier(n_estimators=300, random_state=42) if task == "classification"             else RandomForestRegressor(n_estimators=300, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task=="classification" else None)
    pipe.fit(Xtr, ytr)
    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False, ignore_index=True)
    try:
        import streamlit as st
        st.session_state.setdefault("model_results", {})
        st.session_state["model_results"]["feature_importance"] = imp
        st.session_state["model_results"].setdefault("best_model", "RandomForest (baseline)")
        st.session_state["model_trained"] = True
        st.session_state["analysis_complete"] = True
    except Exception:
        pass
    return imp


from typing import List
def _build_preprocessor(df=None, target=None, X=None):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    # Build X based on provided arguments
    if X is None:
        if df is None:
            raise ValueError("_build_preprocessor: need either df or X")
        if target is not None and target in df.columns:
            X = df.drop(columns=[target])
        else:
            X = df.copy()
    # Detect types
    cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=True, with_std=True), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if not transformers:
        return None, X.columns.tolist()
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return ct, num_cols + cat_cols