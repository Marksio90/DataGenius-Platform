from __future__ import annotations
# -*- coding: utf-8 -*-

# NIE-ZMIENNY interfejs publiczny:
# - apply_plan(df, target, plan) -> pd.DataFrame
# - auto_clean_dataset(df) -> tuple[pd.DataFrame, str]

try:
    from backend.safe_utils import truthy_df_safe  # projektowy helper
except Exception:
    # awaryjny fallback (unikamy twardej zależności)
    def truthy_df_safe(x) -> bool:
        try:
            if x is None:
                return False
            # DataFrame/Series
            import pandas as _pd  # noqa
            if isinstance(x, (_pd.DataFrame, _pd.Series)):
                return not x.empty
            # sekwencje/słowniki
            if hasattr(x, "__len__"):
                return len(x) > 0
            return bool(x)
        except Exception:
            return bool(x)

from typing import Dict, Any, Optional, List, Tuple, Sequence

import numpy as np
import pandas as pd


# =========================
# Helpers (bezpieczne, NaN-safe)
# =========================

def _is_numeric(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    except Exception:
        return False

def _is_datetime(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_datetime64_any_dtype(s)
    except Exception:
        return False

def _safe_mode(s: pd.Series):
    try:
        m = s.mode(dropna=True)
        if truthy_df_safe(m):
            return m.iloc[0]
    except Exception:
        pass
    # fallbacki
    if pd.api.types.is_string_dtype(s) or s.dtype == "object":
        return ""
    if pd.api.types.is_bool_dtype(s):
        return False
    return 0

def _iqr_bounds(values: pd.Series) -> Tuple[float, float]:
    """Zwraca (lo, hi) dla clipu IQR. Jeżeli brak rozrzutu – zwraca (min, max)."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.size < 8:
        if v.size == 0:
            return (0.0, 0.0)
        return (float(v.min()), float(v.max()))
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        return (float(v.min()), float(v.max()))
    lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    return lo, hi

def _iqr_cap(series: pd.Series) -> pd.Series:
    """Clip IQR z ochroną na stałe/puste kolumny i nienumeryczne typy."""
    if not _is_numeric(series):
        return series
    lo, hi = _iqr_bounds(series)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return series  # brak sensownego efektu
    try:
        out = series.clip(lo, hi)
    except Exception:
        out = pd.to_numeric(series, errors="coerce").clip(lo, hi)
    return out


# =========================
# Główny runtime preprocessor
# =========================

def apply_plan(df: pd.DataFrame, target: Optional[str], plan: Dict[str, Any]) -> pd.DataFrame:
    """
    Minimalny runtime preprocessor: odtwarza kroki planu (imputacja, outliery, encoding).

    Domyślne zasady:
      - num: NaN -> mediana
      - cat/bool/tekst: NaN -> moda (fallback: ""/False/0)
      - outliers: IQR clip dla numerycznych (bez targetu)
      - one-hot: kolumny kategoryczne (w tym bool) o n_unik <= 15 (drop_first=True)
      - daty nie są one-hotowane (nawet przy niskiej krotności)

    Opcjonalnie respektuje pola w `plan` (wszystko opcjonalne):
      plan = {
        "columns": {"keep": [...], "drop": [...]},
        "impute": {"num": "median" | "mean" | "zero", "cat": "mode" | "missing" | "zero"},
        "clip_outliers": {"enabled": True, "cols": [...]},  # jeśli cols brak → wszystkie numeryczne
        "onehot": {"max_cardinality": 15, "drop_first": True}
      }
    """
    dfp = df.copy()

    # --- wybór kolumn wg planu (jeśli jest)
    cols_cfg = plan.get("columns", {}) if isinstance(plan, dict) else {}
    cols_keep = set(cols_cfg.get("keep", []) or [])
    cols_drop = set(cols_cfg.get("drop", []) or [])

    if truthy_df_safe(cols_keep):
        # zachowaj tylko istniejące + zawsze target (jeśli jest)
        exists = [c for c in cols_keep if c in dfp.columns]
        if target is not None and target in dfp.columns and target not in exists:
            exists.append(target)
        # zachowujemy kolejność występowania w df
        dfp = dfp[[c for c in dfp.columns if c in set(exists)]]

    if truthy_df_safe(cols_drop):
        dfp = dfp.drop(columns=[c for c in cols_drop if c in dfp.columns], errors="ignore")

    # --- podział na typy
    # bool traktujemy jako kategoryczne
    base_cols = [c for c in dfp.columns if c != target]
    numeric_cols = [c for c in base_cols if _is_numeric(dfp[c])]
    datetime_cols = [c for c in base_cols if _is_datetime(dfp[c])]
    categorical_cols = [c for c in base_cols if c not in numeric_cols and c not in datetime_cols]

    # --- imputacja wg planu (albo domyślna)
    imp_cfg = (plan.get("impute") if isinstance(plan, dict) else {}) or {}
    num_imp = str(imp_cfg.get("num", "median")).lower()
    cat_imp = str(imp_cfg.get("cat", "mode")).lower()

    for c in numeric_cols:
        if not dfp[c].isna().any():
            continue
        s_num = pd.to_numeric(dfp[c], errors="coerce")
        if num_imp == "mean":
            val = float(s_num.mean())
        elif num_imp == "zero":
            val = 0.0
        else:
            val = float(s_num.median())
        try:
            dfp[c] = dfp[c].fillna(val)
        except Exception:
            dfp[c] = s_num.fillna(val)

    for c in categorical_cols:
        if not dfp[c].isna().any():
            continue
        if cat_imp == "missing":
            if pd.api.types.is_string_dtype(dfp[c]) or dfp[c].dtype == "object":
                fillv = "missing"
            else:
                fillv = _safe_mode(dfp[c])
        elif cat_imp == "zero":
            fillv = 0
        else:
            fillv = _safe_mode(dfp[c])
        try:
            dfp[c] = dfp[c].fillna(fillv)
        except Exception:
            dfp[c] = dfp[c].astype(object).fillna(fillv)

    # --- outlier clip (IQR), opcjonalnie wg listy kolumn w planie
    clip_cfg = (plan.get("clip_outliers") if isinstance(plan, dict) else {}) or {}
    clip_enabled = bool(clip_cfg.get("enabled", True))
    if clip_enabled:
        clip_cols = clip_cfg.get("cols", None)
        if clip_cols is None:
            cols_to_clip = numeric_cols
        else:
            cols_to_clip = [c for c in clip_cols if c in dfp.columns and _is_numeric(dfp[c])]
        for c in cols_to_clip:
            try:
                dfp[c] = _iqr_cap(dfp[c])
            except Exception:
                pass  # zachowaj stabilność

    # --- one-hot dla niskiej krotności (bez dat)
    oh_cfg = (plan.get("onehot") if isinstance(plan, dict) else {}) or {}
    try:
        max_card = int(oh_cfg.get("max_cardinality", 15))
    except Exception:
        max_card = 15
    drop_first = bool(oh_cfg.get("drop_first", True))

    low_card = [c for c in categorical_cols if dfp[c].nunique(dropna=False) <= max_card]
    if truthy_df_safe(low_card):
        try:
            dummies = pd.get_dummies(dfp[low_card], drop_first=drop_first, dtype="int64")
            dfp = pd.concat([dfp.drop(columns=low_card), dummies], axis=1)
        except Exception:
            # awaryjnie: iteracyjnie
            for c in low_card:
                try:
                    d = pd.get_dummies(dfp[c], prefix=c, drop_first=drop_first, dtype="int64")
                    dfp = pd.concat([dfp.drop(columns=[c]), d], axis=1)
                except Exception:
                    pass  # pomiń problematyczną kolumnę

    return dfp


# =========================
# Minimalne auto-clean (odporne)
# =========================

def auto_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Odporne auto-czyszczenie: nigdy nie zwraca None.
    Minimalne przekształcenia bez ingerencji w dystrybucję:
      - Konwersja kategorii → string (unik metod .cat przy brakach),
      - Zamiana ±Inf → NaN,
      - Usunięcie kolumn całkowicie pustych,
      - Zrzucenie kolumn typu 'Unnamed*' / 'Index*' (artefakty CSV).
    """
    rep: List[str] = []
    try:
        dfx = df.copy()

        # 1) Kolumny techniczne typu 'Unnamed: 0', 'Index'
        drop_cols = [c for c in dfx.columns if str(c).strip().lower().startswith(("unnamed", "index"))]
        if truthy_df_safe(drop_cols):
            dfx.drop(columns=drop_cols, inplace=True, errors="ignore")
            rep.append(f"[clean] dropped cols: {drop_cols}")

        # 2) Category -> string (stabilność UI/EDA)
        for c in list(dfx.columns):
            try:
                if str(dfx[c].dtype).startswith("category"):
                    dfx[c] = dfx[c].astype("string")
                    rep.append(f"[clean] {c}: category→string")
            except Exception:
                pass

        # 3) ±Inf -> NaN (dla numeric)
        try:
            num_vals = dfx.select_dtypes(include=[np.number]).to_numpy(copy=False)
            n_before_inf = int(np.isinf(num_vals).sum())
        except Exception:
            n_before_inf = 0
        dfx.replace([np.inf, -np.inf], np.nan, inplace=True)
        if n_before_inf > 0:
            rep.append(f"[clean] replaced ±Inf→NaN (count≈{n_before_inf})")

        # 4) Usuń kolumny całkowicie puste
        try:
            empty_mask = dfx.isna().all(axis=0)
            empty_cols = [c for c, is_empty in empty_mask.items() if bool(is_empty)]
        except Exception:
            empty_cols = [c for c in dfx.columns if dfx[c].isna().all()]
        if truthy_df_safe(empty_cols):
            dfx.drop(columns=empty_cols, inplace=True, errors="ignore")
            rep.append(f"[clean] dropped empty columns: {empty_cols}")

        if not truthy_df_safe(rep):
            rep.append("no-op")

        return dfx, "\n".join(rep)

    except Exception as e:
        # Hard fallback – nie ryzykujemy utraty danych
        return df.copy(), f"[fallback] auto_clean failed: {e}"
