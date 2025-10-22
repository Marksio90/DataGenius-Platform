from __future__ import annotations

from backend.safe_utils import truthy_df_safe

"""
AutoDataPrep: szybkie, odporne przygotowanie danych z raportem czyszczenia.

Funkcje:
- analyze_data(X, y): analiza typów, braków, kardynalności, podejrzanych ID.
- clean_data(X, y, options): bezpieczne czyszczenie + imputacja + rzadkie kategorie + clipping outlierów + downcast.
- extract_datetime_features(df, cols): inżynieria cech czasowych.
"""
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

_DEF_NUM_IMPUTE = "median"
_DEF_CAT_IMPUTE = "most_frequent"


def _memory_optimize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # int/uint downcast
    int_like = ["int", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"]
    for col in out.select_dtypes(include=int_like).columns:
        try:
            out[col] = pd.to_numeric(out[col], downcast="integer")
        except Exception:
            pass
    # float downcast
    for col in out.select_dtypes(include=["float", "float64", "float32", "float16"]).columns:
        try:
            out[col] = pd.to_numeric(out[col], downcast="float")
        except Exception:
            pass
    return out


def analyze_data(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["n_rows"] = int(len(X))
    info["n_cols"] = int(X.shape[1])
    info["dtypes"] = {c: str(X[c].dtype) for c in X.columns}
    info["missing_ratio"] = {c: float(X[c].isna().mean()) for c in X.columns}
    info["cardinality"] = {c: int(X[c].nunique(dropna=True)) for c in X.columns}

    # podejrzane ID (unikalne dla każdego wiersza)
    suspicious_id = [c for c, k in info["cardinality"].items() if k == len(X) and X[c].dtype.kind in ("i", "u", "O")]
    info["suspected_id_cols"] = suspicious_id

    # kolumny daty/czasu (po nazwie + po dtype)
    dt_cols = [c for c in X.columns if ("date" in c.lower() or "time" in c.lower())]
    try:
        for c in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[c]) and c not in dt_cols:
                dt_cols.append(c)
    except Exception:
        pass
    info["datetime_cols"] = dt_cols

    # kolumny tekstowe (medianowa długość > 20)
    text_cols: List[str] = []
    for c in X.select_dtypes(include=["object", "string"]).columns:
        try:
            s = X[c].astype("string", errors="ignore")
            med_len = s.str.len().fillna(0).median()
            if float(med_len) > 20:
                text_cols.append(c)
        except Exception:
            continue
    info["text_cols"] = text_cols

    if y is not None:
        ys = pd.Series(y)
        info["y_name"] = ys.name
        try:
            info["y_unique"] = int(ys.nunique(dropna=True))
        except Exception:
            info["y_unique"] = int(pd.Series(ys).nunique(dropna=True))

    return info


def extract_datetime_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        try:
            s = pd.to_datetime(out[c], errors="coerce")
            out[c + "__year"] = s.dt.year
            out[c + "__month"] = s.dt.month
            out[c + "__dow"] = s.dt.weekday
            out[c + "__hour"] = s.dt.hour
        except Exception:
            continue
    return out


def clean_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    rare_threshold: float = 0.005,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
    impute_num: str = _DEF_NUM_IMPUTE,
    impute_cat: str = _DEF_CAT_IMPUTE,
    extract_dt: bool = True,
    drop_duplicates: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Zwraca: X_clean, y_clean, report
    report: {
      dropped_cols: list[str],
      imputed: {col: {...}},
      rarelabeled: {col: {n_rare:int}},
      clipped: {col: {low:float, high:float}},
      dtype_casts: {col: "old->new"},
      steps: [str, ...]
    }
    """
    report: Dict[str, Any] = {
        "dropped_cols": [],
        "imputed": {},
        "rarelabeled": {},
        "clipped": {},
        "dtype_casts": {},
        "steps": [],
    }

    Xc = X.copy()
    yc = y.copy() if y is not None else None

    # 1) Opcjonalne usunięcie duplikatów
    if truthy_df_safe(drop_duplicates):
        before = len(Xc)
        Xc = Xc.drop_duplicates()
        if yc is not None:
            yc = yc.loc[Xc.index]
        report["steps"].append(f"drop_duplicates: {before} -> {len(Xc)} rows")

    # 2) Usuń kolumny stałe
    try:
        constant_cols = [c for c in Xc.columns if Xc[c].nunique(dropna=False) <= 1]
    except Exception:
        constant_cols = []
    if truthy_df_safe(constant_cols):
        report["dropped_cols"].extend(constant_cols)
        Xc = Xc.drop(columns=constant_cols, errors="ignore")

    # 3) Wymuszona konwersja typów: bool → category
    for c in Xc.select_dtypes(include=["bool"]).columns:
        try:
            Xc[c] = Xc[c].astype("category")
            report["dtype_casts"][c] = "bool->category"
        except Exception:
            continue

    # 4) Rare label encoding (łącz rzadkie kategorie do 'Other') – wersja wektorowa
    cat_like = Xc.select_dtypes(include=["object", "category", "string"]).columns
    for c in cat_like:
        try:
            s = Xc[c].astype("string").fillna("<NA>")
            vc = s.value_counts(dropna=False, normalize=True)
            rare_vals = vc[vc < rare_threshold].index
            if truthy_df_safe(list(rare_vals)):
                mask = s.isin(rare_vals)
                s = pd.Series(np.where(mask, "Other", s), index=s.index, dtype="string")
                Xc[c] = s
                report["rarelabeled"][c] = {"n_rare": int(len(rare_vals))}
        except Exception:
            continue

    # 5) Imputacja braków
    if impute_num == "median":
        num_cols = Xc.select_dtypes(include=["number"]).columns
        for c in num_cols:
            try:
                if Xc[c].isna().any():
                    med = float(Xc[c].median())
                    Xc[c] = Xc[c].fillna(med)
                    report["imputed"][c] = {"strategy": "median", "value": med}
            except Exception:
                continue

    if impute_cat == "most_frequent":
        cat_cols = Xc.select_dtypes(include=["object", "category", "string"]).columns
        for c in cat_cols:
            try:
                if Xc[c].isna().any():
                    s = Xc[c].astype("string")
                    mf = s.mode(dropna=True)
                    fillv = mf.iloc[0] if len(mf) else "<NA>"
                    Xc[c] = s.fillna(fillv)
                    report["imputed"][c] = {"strategy": "most_frequent", "value": str(fillv)}
            except Exception:
                continue

    # 6) Clipping outlierów (tylko liczby)
    try:
        low, high = float(clip_quantiles[0]), float(clip_quantiles[1])
    except Exception:
        low, high = 0.01, 0.99
    num_cols = Xc.select_dtypes(include=["number"]).columns
    for c in num_cols:
        try:
            l = float(Xc[c].quantile(low))
            h = float(Xc[c].quantile(high))
            Xc[c] = Xc[c].clip(l, h)
            report["clipped"][c] = {"low": l, "high": h}
        except Exception:
            continue

    # 7) Cechy czasowe (opcjonalnie)
    if truthy_df_safe(extract_dt):
        try:
            dt_cols = [c for c in Xc.columns if ("date" in c.lower() or "time" in c.lower())]
            Xc = extract_datetime_features(Xc, dt_cols)
        except Exception:
            pass

    # 8) Downcast dla pamięci
    Xc = _memory_optimize(Xc)
    report["steps"].append("memory_optimize + dtype normalization done")

    return Xc, yc, report
