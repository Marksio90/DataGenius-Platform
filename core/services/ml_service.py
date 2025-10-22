# core/services/ml_service.py
"""
MLService – plan trenowania + wielomodelowy trening i ewaluacja
dla TMIV – Advanced ML Platform.

Zgodne z `IMLService` (core/interfaces.py).

Cechy:
- Plan trenowania: delegacja do backend.training_plan.build_training_plan (lekki heurystyczny plan).
- Trening: uniwersalny pipeline (imputacja, skalowanie, one-hot) + kilka modeli
  (sklearn; XGBoost/LightGBM/CatBoost ładowane opcjonalnie).
- Obsługa klasyfikacji (binary/multi) i regresji; dane TS → split czasowy.
- Leaderboard, metryki, prosta FI, nazwy cech po transformacji.
- Brak twardych zależności na PyCaret.

Uwaga:
- To jest solidny baseline „działa out-of-the-box”. Później można podmienić
  implementację na integrację z `ml/pipelines/*` lub dedykowany backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# --- opcjonalne biblioteki boostowane (soft importy) ---
_xgbc = _xgbr = None
_lgbmc = _lgbmr = None
_catc = _catr = None
try:  # pragma: no cover
    from xgboost import XGBClassifier as _xgbc, XGBRegressor as _xgbr  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from lightgbm import LGBMClassifier as _lgbmc, LGBMRegressor as _lgbmr  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from catboost import CatBoostClassifier as _catc, CatBoostRegressor as _catr  # type: ignore
except Exception:
    pass

# --- heurystyki planu (delegacja do backend.training_plan) ---
try:
    from backend.training_plan import build_training_plan as _build_training_plan  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("backend.training_plan is required for MLService.") from e


# =========================
# Pomocnicze: typ problemu
# =========================

def _detect_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    nunique = int(y.nunique(dropna=True))
    total = int(y.notna().sum())
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
        return "classification"
    if nunique <= 10 and total and (nunique / total) < 0.2:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and nunique <= 20 and total and (nunique / total) < 0.2:
        return "classification"
    return "regression"


# =========================
# Główna klasa serwisu
# =========================

class MLService:
    # --------------------------
    # Plan trenowania
    # --------------------------
    def build_training_plan(
        self,
        df: pd.DataFrame,
        target: str,
        *,
        strategy: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        return _build_training_plan(
            df,
            target,
            strategy=strategy,
            hints=hints,
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )

    # --------------------------
    # Trening i ewaluacja
    # --------------------------
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target: str,
        *,
        plan: Optional[Mapping[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: Optional[int] = 3,
        enable_ensembles: bool = True,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found.")
        if plan is None:
            plan = self.build_training_plan(df, target, random_state=random_state, n_jobs=n_jobs)

        # Ustal typ problemu (TS → użyj split czasowego, ale model jak reg/cls wg celu)
        pt = str(plan.get("problem_type", "") or "").lower()
        is_ts = (pt == "timeseries") or bool(plan.get("timeseries", {}).get("is_ts"))
        time_col = plan.get("timeseries", {}).get("time_col")

        # Wewnętrzny typ modelu:
        y = df[target]
        inner_type = _detect_problem_type(y)

        # Podział na train/valid
        if is_ts and time_col and time_col in df.columns:
            # sort by time and tail-split
            t = pd.to_datetime(df[time_col], errors="coerce")
            order = np.argsort(t.values.astype("datetime64[ns]"))
            n = len(order)
            cut = max(1, int(round(n * (1.0 - float(plan.get("test_size", test_size))))))
            train_idx = order[:cut]
            valid_idx = order[cut:]
        else:
            if inner_type == "classification":
                train_idx, valid_idx = train_test_split(
                    np.arange(len(df)),
                    test_size=float(plan.get("test_size", test_size)),
                    random_state=random_state,
                    stratify=y,
                )
            else:
                train_idx, valid_idx = train_test_split(
                    np.arange(len(df)),
                    test_size=float(plan.get("test_size", test_size)),
                    random_state=random_state,
                )

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_valid = df.iloc[valid_idx].reset_index(drop=True)
        y_train = df_train[target]
        y_valid = df_valid[target]

        X_train = df_train.drop(columns=[target])
        X_valid = df_valid.drop(columns=[target])

        # Kolumny
        num_cols = list(X_train.select_dtypes(include=["number"]).columns)
        cat_cols = list(X_train.select_dtypes(include=["object", "category", "bool"]).columns)
        # low-cardinality ints → kategorie
        for c in X_train.columns.difference(num_cols + cat_cols):
            s = X_train[c]
            if pd.api.types.is_integer_dtype(s):
                nunique = int(s.nunique(dropna=True))
                if len(s) > 0 and nunique <= 20 and (nunique / len(s)) < 0.2:
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

        # Preprocess (imputers + scaler + OHE)
        num_tr = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=True, with_std=True))]
        )
        cat_tr = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]
        )
        pre = ColumnTransformer(
            transformers=[("num", num_tr, num_cols), ("cat", cat_tr, cat_cols)],
            remainder="drop",
            sparse_threshold=0.0,
        )

        # Label encoding (dla klasyfikacji)
        y_encoder = None
        class_names: List[str] | None = None
        if inner_type == "classification":
            from sklearn.preprocessing import LabelEncoder

            y_encoder = LabelEncoder()
            y_train_enc = y_encoder.fit_transform(y_train.astype(str))
            y_valid_enc = y_encoder.transform(y_valid.astype(str))
            class_names = list(map(str, y_encoder.classes_))
        else:
            y_train_enc = y_train.to_numpy()
            y_valid_enc = y_valid.to_numpy()

        # Modele
        models: Dict[str, Any] = {}
        if inner_type == "classification":
            models.update(self._build_cls_models(n_jobs=n_jobs))
        else:
            models.update(self._build_reg_models(n_jobs=n_jobs))

        # Trening + metryki
        metrics_primary = str(plan.get("metrics_primary", "r2")).lower()
        sec = [str(m).lower() for m in (plan.get("metrics_secondary") or [])]

        rows: List[Dict[str, Any]] = []
        metrics_by_model: Dict[str, Dict[str, float]] = {}
        cv_by_model: Dict[str, Dict[str, float]] = {}

        fitted_models: Dict[str, Any] = {}
        feature_names: List[str] = []

        for name, base_est in models.items():
            pipe = Pipeline(steps=[("pre", pre), ("est", base_est)])
            pipe.fit(X_train, y_train_enc)
            fitted_models[name] = pipe

            # Preds
            y_pred = pipe.predict(X_valid)
            y_proba = None
            try:
                y_proba = pipe.predict_proba(X_valid)
            except Exception:
                # decision_function fallback → przeskaluj do (0,1) jeśli binary
                try:
                    dfun = pipe.decision_function(X_valid)  # type: ignore[attr-defined]
                    if dfun is not None:
                        if dfun.ndim == 1:
                            # sigmoid
                            y_proba = np.vstack([1.0 / (1.0 + np.exp(dfun)), 1.0 - 1.0 / (1.0 + np.exp(dfun))]).T
                        else:
                            # softmax
                            e = np.exp(dfun - np.max(dfun, axis=1, keepdims=True))
                            y_proba = e / np.sum(e, axis=1, keepdims=True)
                except Exception:
                    y_proba = None

            # Metryki
            m = self._compute_metrics(
                kind=inner_type,
                y_true=y_valid_enc,
                y_pred=y_pred,
                y_proba=y_proba,
                class_count=(len(class_names) if class_names else None),
            )
            metrics_by_model[name] = m

            # Wiersz leaderboardu (primary + kilka podrzędnych)
            primary_val = m.get(metrics_primary)
            row = {"model": name, "primary_metric": primary_val}
            for k in (set(sec) | set(m.keys())):
                row[k] = m.get(k)
            rows.append(row)

            # nazwy cech (z pierwszego dopasowanego)
            if not feature_names:
                try:
                    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())  # type: ignore[attr-defined]
                except Exception:
                    feature_names = [f"f{i}" for i in range(pipe.named_steps["pre"].transform(X_valid).shape[1])]  # type: ignore[attr-defined]

        leaderboard = pd.DataFrame(rows)
        if not leaderboard.empty:
            # kierunki: większe lepsze (r2, roc_auc, f1, acc, aps), mniejsze lepsze (rmse, mae, logloss)
            minimize = {"rmse", "mae", "logloss"}
            prim = metrics_primary
            if prim in minimize:
                leaderboard["score"] = -leaderboard["primary_metric"].astype(float)
            else:
                leaderboard["score"] = leaderboard["primary_metric"].astype(float)
            leaderboard = leaderboard.sort_values(by="score", ascending=False, na_position="last").reset_index(drop=True)
            leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)

        # Wybór best model
        if not leaderboard.empty:
            best_name = str(leaderboard.iloc[0]["model"])
            best_model = fitted_models[best_name]
        else:
            # fallback (nie powinno się zdarzyć)
            best_name = next(iter(fitted_models.keys()))
            best_model = fitted_models[best_name]

        # Feature Importance dla najlepszego (o ile dostępne)
        fi_df = self._feature_importance(best_model, feature_names)

        # y_valid w tablicy 1D i mapping indeks->etykieta
        y_valid_arr = np.asarray(y_valid_enc)
        y_mapping = {i: cls for i, cls in enumerate(class_names or [])}

        result: Dict[str, Any] = {
            "problem_type": inner_type if not is_ts else "timeseries",
            "target": target,
            "split": {"train_size": int(len(train_idx)), "valid_size": int(len(valid_idx)), "is_timeseries": bool(is_ts)},
            "leaderboard": leaderboard,
            "best_model_name": best_name,
            "models": fitted_models,
            "preprocessor": pre,
            "feature_names": feature_names,
            "feature_importance": fi_df,
            "y_encoder": y_encoder,
            "metrics_by_model": metrics_by_model,
            "cv_by_model": cv_by_model,  # rezerwujemy na przyszłość
            "y_valid": y_valid_arr,
            "y_mapping": y_mapping,
        }
        return result

    # --------------------------
    # Budowa modeli
    # --------------------------
    def _build_cls_models(self, *, n_jobs: int = -1) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "logreg": LogisticRegression(max_iter=1000, n_jobs=int(n_jobs), class_weight="balanced"),
            "rf": RandomForestClassifier(n_estimators=300, n_jobs=int(n_jobs), class_weight="balanced"),
            "gbc": GradientBoostingClassifier(),
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
            )
        if _lgbmc is not None:
            models["lgbm"] = _lgbmc(n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=int(n_jobs))
        if _catc is not None:
            models["cat"] = _catc(iterations=800, depth=6, learning_rate=0.05, verbose=False, loss_function="Logloss")
        return models

    def _build_reg_models(self, *, n_jobs: int = -1) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "linreg": LinearRegression(),
            "rfr": RandomForestRegressor(n_estimators=400, n_jobs=int(n_jobs)),
            "gbr": GradientBoostingRegressor(),
        }
        if _xgbr is not None:
            models["xgb"] = _xgbr(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=int(n_jobs),
            )
        if _lgbmr is not None:
            models["lgbm"] = _lgbmr(n_estimators=700, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=int(n_jobs))
        if _catr is not None:
            models["cat"] = _catr(iterations=900, depth=6, learning_rate=0.05, verbose=False, loss_function="RMSE")
        return models

    # --------------------------
    # Metryki
    # --------------------------
    def _compute_metrics(
        self,
        *,
        kind: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        class_count: Optional[int] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if kind == "classification":
            # Accuracy / F1
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
            average = "binary" if (class_count or 2) <= 2 else "weighted"
            try:
                out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
            except Exception:
                pass
            try:
                out["f1"] = float(f1_score(y_true, y_pred, average=average))
            except Exception:
                pass

            # ROC-AUC
            try:
                if y_proba is not None:
                    if (class_count or 2) <= 2:
                        out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, -1]))
                    else:
                        out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                # logloss
                if y_proba is not None:
                    out["logloss"] = float(log_loss(y_true, y_proba, labels=np.unique(y_true)))
            except Exception:
                pass

            # Average Precision (PR-AUC proxy) – tylko binary lub makro
            try:
                from sklearn.metrics import average_precision_score

                if y_proba is not None:
                    if (class_count or 2) <= 2:
                        out["aps"] = float(average_precision_score(y_true, y_proba[:, -1]))
                    else:
                        # Makro po klasach
                        aps_vals = []
                        for i in range(y_proba.shape[1]):
                            aps_vals.append(average_precision_score((y_true == i).astype(int), y_proba[:, i]))
                        out["aps"] = float(np.mean(aps_vals))
            except Exception:
                pass

        else:  # regression
            out["r2"] = float(np.nan_to_num(self._safe_r2(y_true, y_pred)))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            out["rmse"] = rmse
            out["mae"] = float(mean_absolute_error(y_true, y_pred))

        return out

    @staticmethod
    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import r2_score

            return float(r2_score(y_true, y_pred))
        except Exception:
            return float("nan")

    # --------------------------
    # Feature importance
    # --------------------------
    def _feature_importance(self, pipe: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
        """
        Spróbuj odczytać ważność cech z modelu bazowego:
        - .feature_importances_ (drzewa),
        - |coef_| (modele liniowe) → |coef| i normalizacja.
        Zwraca DataFrame: feature, importance (posortowany malejąco).
        """
        try:
            est = pipe.named_steps["est"]
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"])

        importances: Optional[np.ndarray] = None

        # Tree-based
        for attr in ("feature_importances_",):
            if hasattr(est, attr):
                try:
                    arr = getattr(est, attr)
                    importances = np.asarray(arr, dtype=float).ravel()
                    break
                except Exception:
                    pass

        # Linear
        if importances is None and hasattr(est, "coef_"):
            try:
                coef = np.asarray(est.coef_, dtype=float)
                if coef.ndim == 2:
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)
                importances = coef
            except Exception:
                importances = None

        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])

        # Align lengths
        k = min(len(feature_names), len(importances))
        feats = list(feature_names)[:k]
        imps = np.asarray(importances[:k], dtype=float)
        # normalize to sum=1 for readability
        s = float(np.sum(np.abs(imps))) or 1.0
        imps = imps / s
        df_fi = pd.DataFrame({"feature": feats, "importance": imps})
        df_fi = df_fi.sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_fi


__all__ = ["MLService"]
