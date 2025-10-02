
from __future__ import annotations
from typing import Dict
import json
import pandas as pd
import numpy as np
import streamlit as st

def _heuristic_describe_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc or 'time' in lc:
            mapping[c] = 'Data/czas (timestamp)'
        elif 'price' in lc or 'cost' in lc or 'avgprice' in lc:
            mapping[c] = 'Cena / koszt (zmienna ciągła)'
        elif 'type' in lc or 'region' in lc:
            mapping[c] = 'Kategoria (zmienna kategoryczna)'
        elif 'total' in lc or 'volume' in lc or 'qty' in lc or 'quantity' in lc:
            mapping[c] = 'Wielkość / ilość (zmienna ciągła)'
        else:
            mapping[c] = 'Zmienna numeryczna' if pd.api.types.is_numeric_dtype(df[c]) else 'Zmienna kategoryczna/tekstowa'
    return mapping

@st.cache_data(show_spinner=False)
def get_column_descriptions(df: pd.DataFrame, api_key: str | None) -> Dict[str, str]:
    if api_key:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            cols = list(df.columns)
            prompt = (
                "Masz listę nazw kolumn danych. Zwróć CZYSTY obiekt JSON mapujący nazwa_kolumny -> krótki opis po polsku, "
                "bez żadnego tekstu poza JSON. Kolumny: " + ", ".join(cols)
            )
            resp = client.chat.completions.create(
                model=settings.openai_model,  # z configu
                response_format={"type": "json_object"},
                messages=[...]
            )
            txt = resp.choices[0].message.content.strip()
            mapping = json.loads(txt)
            base = _heuristic_describe_columns(df)
            base.update({k: str(v) for k,v in mapping.items() if k in base})
            return base
        except Exception:
            pass
    return _heuristic_describe_columns(df)

def quick_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for c in df.columns:
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser):
            summary.append({
                'column': c, 'dtype': str(ser.dtype),
                'non_null': int(ser.notna().sum()), 'n_unique': int(ser.nunique()),
                'mean': float(ser.mean()), 'std': float(ser.std() or 0.0),
                'min': float(ser.min()), 'max': float(ser.max()),
            })
        else:
            top = ser.value_counts(dropna=True).head(3).to_dict()
            summary.append({
                'column': c, 'dtype': str(ser.dtype),
                'non_null': int(ser.notna().sum()), 'n_unique': int(ser.nunique()),
                'top_values': top
            })
    return pd.DataFrame(summary)

def eda_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        ser = df[c]
        missing = ser.isna().sum()
        nunique = ser.nunique(dropna=True)
        is_constant = (nunique <= 1)
        card = nunique if not pd.api.types.is_numeric_dtype(ser) else None
        rows.append({
            "column": c, "dtype": str(ser.dtype),
            "missing": int(missing), "missing_pct": float(missing / n if n else 0.0),
            "nunique": int(nunique), "constant": bool(is_constant), "cardinality": card
        })
    return pd.DataFrame(rows).sort_values(["missing_pct", "nunique"], ascending=[False, True])

# === AUTOMATYCZNE PRZYGOTOWANIE DANYCH ===
def _is_date_like(s: _pd.Series) -> bool:
    if _pd.api.types.is_datetime64_any_dtype(s): return True
    if s.dtype==object:
        try:
            _pd.to_datetime(s, errors="raise")
            return True
        except Exception:
            return False
    return False

def auto_prepare_data(df: _pd.DataFrame, target: str|None=None) -> tuple[_pd.DataFrame, dict]:
    df = df.copy()
    info: dict = {"dropped_constant": [], "dropped_allnull": [], "parsed_dates": [], "capped_cats": {}, "clipped": []}

    # Drop empty
    for c in list(df.columns):
        if df[c].isna().all():
            info["dropped_allnull"].append(c); df.drop(columns=[c], inplace=True, errors="ignore")

    # Drop constant
    for c in list(df.columns):
        try:
            if df[c].nunique(dropna=True) <= 1:
                info["dropped_constant"].append(c); df.drop(columns=[c], inplace=True, errors="ignore")
        except Exception:
            pass

    # Parse dates
    for c in list(df.columns):
        try:
            if _is_date_like(df[c]):
                dt = _pd.to_datetime(df[c], errors="coerce")
                df[f"{c}__year"] = dt.dt.year
                df[f"{c}__month"] = dt.dt.month
                df[f"{c}__dow"] = dt.dt.dayofweek
                info["parsed_dates"].append(c)
        except Exception:
            pass

    # High cardinality cap: top 30, rest OTHER
    for c in list(df.columns):
        try:
            if _pd.api.types.is_object_dtype(df[c]) or _pd.api.types.is_categorical_dtype(df[c]):
                vc = df[c].value_counts(dropna=False)
                if len(vc) > 30:
                    keep = set(vc.head(30).index.tolist())
                    df[c] = df[c].apply(lambda x: x if x in keep else "OTHER")
                    info["capped_cats"][c] = int(len(vc))
        except Exception:
            pass

    # Light numeric clipping if regression target known
    if target and target in df.columns and _pd.api.types.is_numeric_dtype(df[target]):
        nums = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
        for c in nums:
            try:
                lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
                df[c] = df[c].clip(lo, hi)
                info["clipped"].append(c)
            except Exception:
                pass

    
    # Remove duplicate rows
    try:
        before = len(df)
        df = df.drop_duplicates()
        info["dropped_duplicates"] = int(before - len(df))
    except Exception:
        info["dropped_duplicates"] = 0

    # Impute missing values
    try:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        for c in num_cols:
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
        for c in cat_cols:
            if df[c].isna().any():
                try:
                    mode_val = df[c].mode(dropna=True).iloc[0]
                except Exception:
                    mode_val = "__MISSING__"
                df[c] = df[c].fillna(mode_val)
        info["imputed_num"] = [c for c in num_cols if c in df.columns]
        info["imputed_cat"] = [c for c in cat_cols if c in df.columns]
    except Exception:
        pass

    # Robust winsorization for numerics (0.5% - 99.5%), excluding target
    try:
        num_cols2 = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
        for c in num_cols2:
            ql, qh = df[c].quantile(0.005), df[c].quantile(0.995)
            df[c] = df[c].clip(ql, qh)
        info["winsorized_0_5_99_5"] = num_cols2
    except Exception:
        pass

    return df, info
