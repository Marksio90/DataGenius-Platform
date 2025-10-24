# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def _iqr_cap(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return series.clip(lo, hi)

def apply_plan(df: pd.DataFrame, target: Optional[str], plan: Dict[str, Any]) -> pd.DataFrame:
    """Minimalny runtime preprocessor: odtwarza kroki planu (imputacja, outliery, encoding)."""
    dfp = df.copy()
    num_cols = [c for c in dfp.columns if c != target and pd.api.types.is_numeric_dtype(dfp[c])]
    cat_cols = [c for c in dfp.columns if c != target and c not in num_cols]

    for c in num_cols:
        if dfp[c].isna().any():
            med = dfp[c].median()
            dfp[c] = dfp[c].fillna(med)

    for c in cat_cols:
        if dfp[c].isna().any():
            mode = dfp[c].mode().iloc[0] if not dfp[c].mode().empty else ""
            dfp[c] = dfp[c].fillna(mode)

    for c in num_cols:
        dfp[c] = _iqr_cap(dfp[c])

    low_card = [c for c in cat_cols if dfp[c].nunique()<=15]
    if low_card:
        dummies = pd.get_dummies(dfp[low_card], drop_first=True, dtype=int)
        dfp = pd.concat([dfp.drop(columns=low_card), dummies], axis=1)

    return dfp


def auto_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Odporne auto-czyszczenie: nigdy nie zwraca None. Minimalne przekształcenia bez ingerencji w dystrybucję.
    """
    import numpy as _np
    import pandas as _pd

    rep = []
    try:
        dfx = df.copy()

        # Categorical → string (częsty problem z CategoricalDtype)
        for c in dfx.columns:
            try:
                if str(dfx[c].dtype).startswith("category"):
                    dfx[c] = dfx[c].astype("string")
                    rep.append(f"[clean] {c}: category→string")
            except Exception:
                pass

        # zamiana inf na NaN
        dfx.replace([_np.inf, -_np.inf], _np.nan, inplace=True)

        # miękkie wypełnienie bloczkowe – nie wymuszamy, downstream sobie poradzi
        # ale jeżeli wszystkie wartości w kolumnie są NaN, zostawiamy jak jest
        return dfx, ("\n".join(rep) if rep else "no-op")

    except Exception as e:
        return df.copy(), f"[fallback] auto_clean failed: {e}"
