# backend/safe_utils.py
from __future__ import annotations
from typing import Any
import numpy as np

try:
    import pandas as pd
except Exception:  # brak pandas? działamy dalej
    pd = None  # type: ignore


def truthy_df_safe(x: Any) -> bool:
    """
    Bezpieczny 'is truthy?' dla typów używanych w projekcie:
    - None -> False
    - pandas.DataFrame/Series -> True jeśli niepuste (dla Series: ma przynajmniej 1 nie-NaN)
    - numpy.ndarray -> True jeśli size>0 (dla numerycznych: gdy istnieje choć jedna skończona wartość)
    - str -> True jeśli po strip() nie jest puste
    - mapping/sequence -> True jeśli len>0
    - liczby -> standardowy bool(), ale NaN -> False
    - inne -> próba bool(x), w razie błędu -> True (zachowawczo)
    """
    if x is None:
        return False

    # pandas
    if pd is not None:
        if isinstance(x, pd.DataFrame):
            return not x.empty
        if isinstance(x, pd.Series):
            if x.empty:
                return False
            try:
                return x.notna().any()
            except Exception:
                return True

    # numpy arrays
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        # unikamy "ambiguous truth value"
        try:
            if np.issubdtype(x.dtype, np.number):
                # cokolwiek skończonego?
                return np.isfinite(x).any()
            # dla innych dtype spróbuj konwersji do bool
            return np.any(x.astype(bool))
        except Exception:
            return True

    # tekst
    if isinstance(x, str):
        return len(x.strip()) > 0

    # mapping / sequence (bez traktowania stringów jako sequence)
    try:
        from collections.abc import Mapping, Sequence
        if isinstance(x, Mapping):
            return len(x) > 0
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            return len(x) > 0
    except Exception:
        pass

    # liczby – uważaj na NaN
    if isinstance(x, (int, bool)):
        return bool(x)
    if isinstance(x, float):
        try:
            return not np.isnan(x) and bool(x)
        except Exception:
            return False
    # numpy skalary (np.float64 itp.)
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        try:
            return not (np.issubdtype(type(x), np.floating) and np.isnan(x)) and bool(x)
        except Exception:
            return bool(x)

    # fallback
    try:
        return bool(x)
    except Exception:
        # jeśli obiekt nie wspiera bool(), uznaj go za "istniejący"
        return True


__all__ = ["truthy_df_safe"]
