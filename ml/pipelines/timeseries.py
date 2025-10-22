# ml/pipelines/timeseries.py
"""
Time Series pipelines (skeleton) – TMIV Advanced ML Platform.

Cel:
- Prosty, skalowalny **feature-based forecasting** (regresja) bez twardych zależności.
- Działa na dowolnym szeregu czasowym (univariate) z opcjonalnymi egzogenicznymi cechami.
- Gotowe narzędzia do:
    * przygotowania cech opóźnień/okien (lags/rolling windows),
    * podziału czasowego train/valid,
    * budowy modeli (sklearn + opcjonalnie XGB/LGBM/CatBoost),
    * prognozowania kroczącego (recursive multi-step).

API (kluczowe):
- assemble_supervised(df, time_col, target, exog=None, lags=(1,7,14), windows=(7,28)) -> pd.DataFrame
- split_time(df, time_col, *, test_size=0.2, horizon=None, min_train=100) -> (train_idx, valid_idx)
- make_preprocessor(X) -> ColumnTransformer
- build_timeseries_models(n_jobs=-1, random_state=42) -> dict[name -> estimator]
- recursive_forecast(model, last_frame, steps, target, time_col, lags, windows) -> np.ndarray
- ts_metrics(y_true, y_pred) -> dict

Uwagi:
- To szkielet produkcyjny: czytelny, działa od ręki; w razie potrzeby można
  podmienić na dedykowane biblioteki TS (Prophet, SARIMAX, Darts) bez naruszania kontraktów.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- soft-importy na boostowane modele (opcjonalnie) ---
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
# Utilities
# =========================

def _ensure_datetime_sorted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values(time_col).reset_index(drop=True)
    return out


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where(b == 0, np.nan, b)
    return a / b


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Symmetric MAPE [%]
    denom = (np.abs(y_true) + np.abs(y_pred))
    val = np.nanmean(2.0 * np.abs(y_pred - y_true) / np.where(denom == 0, np.nan, denom))
    return float(val * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    val = np.nanmean(np.abs(safe_div(y_pred - y_true, y_true)))
    return float(val * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))
    except Exception:
        # ręczny fallback
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        return float(1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan)


def ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Zestaw podstawowych metryk dla prognoz TS."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "r2": r2(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "smape_pct": smape(y_true, y_pred),
    }


# =========================
# Feature engineering
# =========================

def assemble_supervised(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    *,
    exog: Optional[Sequence[str]] = None,
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 28),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Zamień szereg (z opcjonalnymi egzogenicznymi) w nadzorowaną ramkę:
    - cechy: lag_k, rollmean_w, rollstd_w, rollmin_w, rollmax_w
    - target: wartość bieżąca (y_t)
    """
    if time_col not in df.columns or target not in df.columns:
        raise KeyError("Brak kolumny czasu lub targetu.")

    exog = list(exog or [])
    work = _ensure_datetime_sorted(df[[time_col, target] + [c for c in exog if c in df.columns]], time_col)
    work = work.copy()

    # cechy kalendarzowe (podstawowe)
    work["__year"] = work[time_col].dt.year
    work["__month"] = work[time_col].dt.month
    work["__day"] = work[time_col].dt.day
    work["__dow"] = work[time_col].dt.dayofweek
    work["__week"] = work[time_col].dt.isocalendar().week.astype(int)

    # lags
    for k in sorted(set(int(x) for x in lags if int(x) > 0)):
        work[f"lag_{k}"] = work[target].shift(k)

    # rolling stats
    for w in sorted(set(int(x) for x in windows if int(x) > 1)):
        roll = work[target].rolling(window=w, min_periods=max(2, int(w * 0.6)))
        work[f"rollmean_{w}"] = roll.mean()
        work[f"rollstd_{w}"] = roll.std()
        work[f"rollmin_{w}"] = roll.min()
        work[f"rollmax_{w}"] = roll.max()

    # egzogeniczne: dodaj jak są (bez modyfikacji)
    # target docelowy = bieżąca wartość
    out = work
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out


# =========================
# Split time
# =========================

def split_time(
    df: pd.DataFrame,
    time_col: str,
    *,
    test_size: float | None = 0.2,
    horizon: int | None = None,
    min_train: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Podział czasowy: przeszłość → train, ogon → valid.
    - Gdy `horizon` jest podany: użyj dokładnie tylu ostatnich obserwacji jako walidacja.
    - W przeciwnym wypadku użyj ułamka `test_size` (np. 0.2).
    Gwarancja: co najmniej `min_train` obserwacji w treningu, jeśli możliwe.
    """
    s = _ensure_datetime_sorted(df, time_col)
    n = len(s)
    if n < 2:
        raise ValueError("Zbyt mało obserwacji do podziału.")

    if horizon is not None:
        horizon = int(max(1, horizon))
        cut = max(min_train, n - horizon)
    else:
        ts = float(test_size if test_size is not None else 0.2)
        ts = min(max(0.05, ts), 0.5)
        cut = max(min_train, int(round(n * (1.0 - ts))))

    cut = min(max(1, cut), n - 1)
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


# =========================
# Preprocessing (sklearn)
# =========================

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Numeryczne: imputacja medianą + StandardScaler
    Kategoryczne: imputacja trybem + OneHotEncoder
    """
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)

    # low-cardinality ints → kategoria
    for c in X.columns.difference(num_cols + cat_cols):
        s = X[c]
        if pd.api.types.is_integer_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if len(s) and nunique <= 20 and (nunique / len(s)) < 0.2:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
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
    return Pipeline(steps=[("pre", preprocessor), ("est", estimator)])


# =========================
# Models
# =========================

def build_timeseries_models(*, n_jobs: int = -1, random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Zwróć zestaw prostych regresorów do prognozy TS (feature-based).
    """
    models: Dict[str, BaseEstimator] = {
        "ridge": Ridge(alpha=1.0, random_state=int(random_state)),
        "rfr": RandomForestRegressor(n_estimators=400, n_jobs=int(n_jobs), random_state=int(random_state)),
        "gbr": GradientBoostingRegressor(random_state=int(random_state)),
    }
    if _xgbr is not None:
        models["xgb"] = _xgbr(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
        )
    if _lgbmr is not None:
        models["lgbm"] = _lgbmr(
            n_estimators=800,
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
# Forecasting (recursive)
# =========================

def recursive_forecast(
    model: Pipeline | BaseEstimator,
    last_frame: pd.DataFrame,
    steps: int,
    *,
    target: str,
    time_col: str,
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 28),
) -> np.ndarray:
    """
    Prognoza krocząca na `steps` do przodu:
    - `last_frame` to DF zawierający przynajmniej ostatnie max(lags, windows) wierszy szeregu
      oraz wymagane kolumny (time_col, target i ewentualne egzogeniczne).
    - W każdej iteracji dopisujemy nową obserwację z przewidywanym targetem
      i obliczamy cechy dla kolejnego kroku.

    Zwraca wektor y_hat o długości `steps`.
    """
    if steps <= 0:
        return np.array([], dtype=float)

    # Zadbaj o porządek czasu
    hist = _ensure_datetime_sorted(last_frame.copy(), time_col)
    preds: List[float] = []

    # Przyjmujemy równy krok czasowy: próbujemy inferencji z różnicy mediany
    dt = _infer_step(hist[time_col])

    for _ in range(int(steps)):
        # Zbuduj cechy dla następnego punktu czasu:
        next_time = (hist[time_col].iloc[-1] + dt) if dt is not None else (hist[time_col].iloc[-1])
        # skeleton: tworzymy tymczasowy wiersz z NaN targetem
        temp = pd.DataFrame({time_col: [next_time]})
        for c in hist.columns:
            if c not in temp.columns:
                temp[c] = np.nan
        ext = pd.concat([hist, temp], ignore_index=True)

        sup = assemble_supervised(ext, time_col, target, exog=[c for c in hist.columns if c not in {time_col, target}], lags=lags, windows=windows)
        # weź ostatni wiersz (do predykcji)
        row = sup.tail(1).drop(columns=[time_col, target], errors="ignore")
        if isinstance(model, Pipeline):
            y_hat = float(model.predict(row)[0])
        else:
            y_hat = float(model.predict(row)[0])  # typowo również Pipeline
        preds.append(y_hat)

        # uzupełnij przewidywaną wartość w historii i idź dalej
        hist.loc[len(hist)] = hist.iloc[-1]  # powielenie kolumn
        hist.at[len(hist) - 1, time_col] = next_time
        hist.at[len(hist) - 1, target] = y_hat

    return np.asarray(preds, dtype=float)


def _infer_step(s: pd.Series) -> pd.Timedelta | None:
    """Spróbuj oszacować dominujący krok czasu (median delta)."""
    try:
        s = pd.to_datetime(s, errors="coerce")
        d = s.diff().dropna()
        if len(d) == 0:
            return None
        return pd.to_timedelta(d.median())
    except Exception:
        return None


__all__ = [
    "assemble_supervised",
    "split_time",
    "make_preprocessor",
    "make_pipeline",
    "build_timeseries_models",
    "recursive_forecast",
    "ts_metrics",
]
