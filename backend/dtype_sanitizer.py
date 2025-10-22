# backend/dtype_sanitizer.py
"""
Datatype & schema sanitization for TMIV – Advanced ML Platform.

What it does (idempotent & safe):
- normalizes column names to safe snake_case and ensures uniqueness,
- trims whitespace in string-like columns,
- converts obvious booleans (yes/no, true/false, 0/1) → bool,
- parses numbers stored as strings (handles commas/dots & thousands separators),
- parses datetimes (when column name/value pattern suggests dates),
- converts low-cardinality text to pandas 'category' (optional),
- downcasts numeric dtypes to reduce memory,
- returns a rich REPORT describing all changes.

Public API
----------
sanitize_dataframe(
    df: pd.DataFrame,
    *,
    normalize_names: bool = True,
    coerce_booleans: bool = True,
    parse_numbers: bool = True,
    parse_datetimes: bool = True,
    strip_strings: bool = True,
    category_max_unique: int = 200,
) -> tuple[pd.DataFrame, dict]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Helpers: column naming
# -------------------------

def _to_snake(name: str) -> str:
    orig = str(name)
    s = orig.strip()
    s = s.replace("/", "_").replace("-", "_")
    # Split camelCase -> snake
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Non-alnum -> underscore
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_{2,}", "_", s)
    s = s.strip("_").lower()
    return s or "col"

def _ensure_unique(cols: List[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Ensure unique column names by suffixing _1, _2... Returns (new_cols, mapping, deduped_list)."""
    seen = {}
    out = []
    mapping = {}
    deduped = []
    for old in cols:
        new = old
        if new in seen:
            k = 1
            while f"{new}_{k}" in seen:
                k += 1
            new = f"{new}_{k}"
            deduped.append(new)
        seen[new] = True
        out.append(new)
    mapping = {o: n for o, n in zip(cols, out) if o != n}
    return out, mapping, deduped


# -------------------------
# Heuristics: dtype detection
# -------------------------

_BOOL_TRUE = {"true", "t", "yes", "y", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "0"}

_DATE_NAME_PAT = re.compile(r"(date|time|timestamp|dt|created|updated|day|month|year)\b", re.I)

def _looks_like_bool_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        uniq = pd.unique(s.dropna())
        return set(uniq).issubset({0, 1})
    if pd.api.types.is_object_dtype(s):
        vals = s.dropna().astype(str).str.strip().str.lower()
        uniq = set(vals.unique().tolist())
        uniq = {re.sub(r"\s+", "", v) for v in uniq}
        return uniq.issubset(_BOOL_TRUE | _BOOL_FALSE) and len(uniq) <= 4
    return False

def _coerce_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int8").astype("boolean") if pd.isna(s).any() else s.astype(bool)
    vals = s.astype(str).str.strip().str.lower()
    vals = vals.replace(list(_BOOL_TRUE), True).replace(list(_BOOL_FALSE), False)
    # Pandas replace above may not hit all — map again:
    def _map(v: Any):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        t = str(v).strip().lower().replace(" ", "")
        if t in _BOOL_TRUE:
            return True
        if t in _BOOL_FALSE:
            return False
        return pd.NA
    res = vals.map(_map)
    return res.astype("boolean")

def _ratio_numeric_parseable(series: pd.Series, treat_comma_as_decimal: bool) -> float:
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return 0.0
    ok = 0
    for v in vals.head(1000):
        t = v.strip()
        if treat_comma_as_decimal:
            t = t.replace(" ", "").replace(".", "").replace(",", ".")
        else:
            t = t.replace(" ", "").replace(",", "")
        try:
            float(t)
            ok += 1
        except Exception:
            pass
    return ok / len(vals.head(1000))

def _try_parse_numeric(s: pd.Series) -> Tuple[pd.Series, str | None]:
    """Try parsing object series to numeric, returning (series, reason)."""
    if not pd.api.types.is_object_dtype(s):
        return s, None
    # Decide how to treat comma: decimal or thousands separator
    r_comma_decimal = _ratio_numeric_parseable(s, treat_comma_as_decimal=True)
    r_thousand_sep = _ratio_numeric_parseable(s, treat_comma_as_decimal=False)
    if max(r_comma_decimal, r_thousand_sep) < 0.85:
        return s, None  # too risky
    use_comma_decimal = r_comma_decimal >= r_thousand_sep
    t = s.astype(str)
    if use_comma_decimal:
        t = t.str.replace(" ", "", regex=False).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        reason = "parsed_numeric(comma-as-decimal)"
    else:
        t = t.str.replace(" ", "", regex=False).str.replace(",", "", regex=False)
        reason = "parsed_numeric(thousands-comma)"
    num = pd.to_numeric(t, errors="coerce")
    return num, reason

def _looks_like_datetime_by_name(col: str) -> bool:
    return bool(_DATE_NAME_PAT.search(col))

def _try_parse_datetime(s: pd.Series, colname: str) -> Tuple[pd.Series, str | None]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, None
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)):
        return s, None
    # Heuristic: only attempt if name suggests date/time OR many values parseable
    name_hint = _looks_like_datetime_by_name(colname)
    sample = s.dropna().astype(str).head(500)
    success = 0
    for v in sample:
        try:
            pd.to_datetime(v)
            success += 1
        except Exception:
            pass
    ratio = success / max(1, len(sample))
    if name_hint or ratio >= 0.6:
        try:
            dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
            if dt.notna().mean() >= 0.5:
                return dt, "parsed_datetime"
        except Exception:
            return s, None
    return s, None


# -------------------------
# Main API
# -------------------------

def sanitize_dataframe(
    df: pd.DataFrame,
    *,
    normalize_names: bool = True,
    coerce_booleans: bool = True,
    parse_numbers: bool = True,
    parse_datetimes: bool = True,
    strip_strings: bool = True,
    category_max_unique: int = 200,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean & coerce dtypes in a defensive manner. Returns (df_sanitized, report).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("sanitize_dataframe expects a pandas DataFrame")

    df = df.copy()
    report: Dict[str, Any] = {
        "renamed": {},
        "deduplicated": [],
        "type_changes": [],
        "string_trimmed_cols": [],
        "parsed_datetimes": [],
        "parsed_numbers": [],
        "coerced_booleans": [],
        "as_category": [],
        "downcasted": [],
        "nulls_after": {},
        "warnings": [],
    }

    # --- Normalize & deduplicate column names ---
    if normalize_names:
        new_cols = [_to_snake(c) for c in df.columns]
        if new_cols != list(df.columns):
            report["renamed"] = {str(o): str(n) for o, n in zip(df.columns, new_cols) if str(o) != str(n)}
            df.columns = new_cols
    # Ensure unique
    unique_cols, mapping, deduped = _ensure_unique(list(df.columns))
    if mapping:
        report["renamed"].update(mapping)
        df.columns = unique_cols
    if deduped:
        report["deduplicated"] = deduped

    # --- Trim strings (leading/trailing) ---
    if strip_strings:
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                df[col] = s.astype("string").str.strip()
                report["string_trimmed_cols"].append(col)

    # --- Booleans ---
    if coerce_booleans:
        for col in df.columns:
            s = df[col]
            if _looks_like_bool_series(s):
                before = str(s.dtype)
                df[col] = _coerce_bool(s)
                after = str(df[col].dtype)
                if before != after:
                    report["coerced_booleans"].append(col)
                    report["type_changes"].append({"column": col, "from": before, "to": after, "reason": "boolean_coercion"})

    # --- Numbers from strings ---
    if parse_numbers:
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_object_dtype(s):
                parsed, reason = _try_parse_numeric(s)
                if reason is not None:
                    before = str(s.dtype)
                    df[col] = parsed
                    after = str(df[col].dtype)
                    report["parsed_numbers"].append({"column": col, "reason": reason})
                    report["type_changes"].append({"column": col, "from": before, "to": after, "reason": reason})

    # --- Datetime parsing ---
    if parse_datetimes:
        for col in df.columns:
            s = df[col]
            parsed, reason = _try_parse_datetime(s, col)
            if reason is not None:
                before = str(s.dtype)
                df[col] = parsed
                after = str(df[col].dtype)
                report["parsed_datetimes"].append(col)
                report["type_changes"].append({"column": col, "from": before, "to": after, "reason": reason})

    # --- Low-cardinality text -> category ---
    if category_max_unique is not None and category_max_unique > 0:
        for col in df.columns:
            s = df[col]
            if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.notna().any():
                nunique = s.nunique(dropna=True)
                if nunique <= category_max_unique:
                    before = str(s.dtype)
                    df[col] = s.astype("category")
                    after = str(df[col].dtype)
                    report["as_category"].append({"column": col, "nunique": int(nunique)})
                    report["type_changes"].append({"column": col, "from": before, "to": after, "reason": "low_cardinality"})

    # --- Downcast numeric where safe ---
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            before = str(s.dtype)
            df[col] = pd.to_numeric(s, downcast="integer")
            after = str(df[col].dtype)
            if before != after:
                report["downcasted"].append({"column": col, "from": before, "to": after})
                report["type_changes"].append({"column": col, "from": before, "to": after, "reason": "downcast_int"})
        elif pd.api.types.is_float_dtype(s):
            before = str(s.dtype)
            df[col] = pd.to_numeric(s, downcast="float")
            after = str(df[col].dtype)
            if before != after:
                report["downcasted"].append({"column": col, "from": before, "to": after})
                report["type_changes"].append({"column": col, "from": before, "to": after, "reason": "downcast_float"})

    # --- Final null summary ---
    try:
        report["nulls_after"] = {c: int(df[c].isna().sum()) for c in df.columns}
    except Exception:
        report["warnings"].append("Failed to compute final null counts")

    return df, report
