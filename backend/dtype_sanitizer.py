# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd

def sanitize_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix problematic dtypes for sklearn/plotting: string[python]/pyarrow -> object, BooleanDtype -> bool, keep category."""
    df = df.copy()
    from pandas.api.types import is_string_dtype, is_bool_dtype, is_categorical_dtype
    for c in df.columns:
        try:
            if is_string_dtype(df[c].dtype):
                df[c] = df[c].astype(object)
            elif is_bool_dtype(df[c].dtype):
                df[c] = df[c].astype(bool)
            elif is_categorical_dtype(df[c].dtype):
                df[c] = df[c].astype('category')
        except Exception:
            pass
    return df
