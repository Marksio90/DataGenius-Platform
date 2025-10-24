
from __future__ import annotations
import pandas as pd
from typing import Tuple

def detect_target_and_task(df: pd.DataFrame, target: str | None) -> Tuple[str, str]:
    """Prosta heurystyka wyboru targetu i typu zadania."""
    if target is None:
        # wybierz ostatnią kolumnę nie-unikalną jako target
        for col in reversed(df.columns.tolist()):
            if df[col].nunique() < len(df):
                target = col
                break
        if target is None:
            target = df.columns[-1]
    # klasyfikacja gdy <=20 unikalnych wartości i jedna z dtypes: int/bool/object
    task = "regression"
    nunq = df[target].nunique()
    if nunq <= 20 or df[target].dtype == bool or df[target].dtype.name in {"category","object","bool","int64"}:
        task = "classification"
    return target, task
