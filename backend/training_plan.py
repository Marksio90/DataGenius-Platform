# backend/training_plan.py
"""
Training plan heuristics for TMIV – Advanced ML Platform.

Cel modułu
----------
Na podstawie wejściowego DataFrame i kolumny celu generujemy **plan trenowania**,
który zawiera:
- wykrycie typu problemu (klasyfikacja / regresja / time series – best effort),
- dobór metryk (primary + secondary),
- strategię budżetu ("fast_small" | "balanced" | "accurate" | "advanced"),
- ustawienia walidacji (CV folds, rodzaj walidacji, stratification / TSCV),
- decyzje o tuningu i ensemblach (progi zależne od wielkości danych),
- rozmiar walidacyjnego hold-out (test_size), seed, n_jobs,
- notatki/uzasadnienia (for UI).

Moduł nie ma ciężkich zależności; korzysta wyłącznie z numpy/pandas/stdlib.

Public API
----------
build_training_plan(
    df: pd.DataFrame,
    target: str,
    *,
    strategy: str | None = None,          # opcjonalny override
    hints: dict | None = None,            # np. {"timeseries": True, "time_col": "date"}
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict

Zwracany słownik (kluczowe pola)
--------------------------------
{
  "problem_type": "classification" | "regression" | "timeseries",
  "metrics_primary": str,
  "metrics_secondary": list[str],
  "cv": {"kind": "kfold"|"stratified"|"tscv", "folds": int},
  "test_size": float,
  "strategy": "fast_small"|"balanced"|"accurate"|"advanced",
  "tuning": {"enable": bool, "n_trials": int, "time_budget_min": int},
  "ensembles": {"enable": bool, "method": "blend"|"stack", "reason": str},
  "timeseries": {"is_ts": bool, "time_col": str | None, "freq_inferred": str | None},
  "n_jobs": int,
  "random_state": int,
  "notes": list[str]
}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Helpers: detection
# =========================

def _detect_problem_type(y: pd.Series) -> str:
    """Heurystyka typu problemu zgodna z logiką używaną w innych modułach."""
    s = y
    if pd.api.types.is_bool_dtype(s):
        return "classification"
    nunique = int(s.nunique(dropna=True))
    total = int(s.notna().sum())
    if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return "classification"
    if nunique <= 10 and (total > 0) and (nunique / total) < 0.2:
        return "classification"
    if pd.api.types.is_integer_dtype(s) and nunique <= 20 and (total > 0) and (nunique / total) < 0.2:
        return "classification"
    return "regression"


_TS_NAME_HINTS = {"date", "datetime", "timestamp", "ts", "time", "period"}


def _likely_time_column(df: pd.DataFrame, target: str | None = None) -> tuple[Optional[str], Optional[str]]:
    """
    Wykryj kandydat kolumny czasu i spróbuj odczytać częstotliwość (best effort).
    Zwraca: (time_col, inferred_freq_str|None)
    """
    # 1) Kolumny datetime
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    # 2) Nazwowe sugestie
    name_suggest = [c for c in df.columns if c.lower() in _TS_NAME_HINTS or any(h in c.lower() for h in _TS_NAME_HINTS)]
    for c in list(dict.fromkeys(dt_cols + name_suggest)):  # zachowaj kolejność, unikalność
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().mean() > 0.9 and s.nunique(dropna=True) > 0.5 * len(s):
                # wygląda na realny znacznik czasu
                # Spróbuj oszacować częstotliwość na podstawie posortowanego indeksu
                ss = s.sort_values()
                diffs = (ss - ss.shift(1)).dropna()
                if len(diffs) >= 5:
                    q = diffs.quantile(0.5)
                    # przekształć na "X days" / "X hours" itd. (upraszczamy)
                    if q.components.days >= 1:
                        freq = f"{int(round(q.components.days))}D"
                    elif q.components.hours >= 1:
                        freq = f"{int(round(q.components.hours))}H"
                    elif q.components.minutes >= 1:
                        freq = f"{int(round(q.components.minutes))}T"
                    else:
                        freq = None
                else:
                    freq = None
                return c, freq
        except Exception:
            continue
    return None, None


def _detect_timeseries(df: pd.DataFrame, target: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Spróbuj wykryć, czy dane mają charakter sekwencyjno-czasowy.
    Kryteria:
    - indeks DataFrame lub kolumna typu datetime, która rośnie,
    - nazwa kolumny sugerująca time (date/timestamp itp.),
    - brak mieszania prób między foldami (zalecenie TSCV).
    """
    # index as datetime?
    try:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing:
            return True, None, None
    except Exception:
        pass

    time_col, freq = _likely_time_column(df, target)
    if time_col:
        # check monotonicity
        try:
            s = pd.to_datetime(df[time_col], errors="coerce")
            if s.notna().mean() > 0.9:
                monot = s.dropna().is_monotonic_increasing
                if not monot:
                    # nawet jeśli nie jest monotonicznie, nadal możemy potraktować jako TS
                    return True, time_col, freq
                return True, time_col, freq
        except Exception:
            return True, time_col, freq

    return False, None, None


# =========================
# Metrics & CV selection
# =========================

def _select_metrics(problem_type: str, y: pd.Series) -> tuple[str, list[str]]:
    """(primary, secondary). Dla klasyfikacji dostosuj do liczby klas."""
    pt = problem_type
    if pt == "classification":
        n_classes = int(y.nunique(dropna=True))
        if n_classes <= 2:
            primary = "roc_auc"
            secondary = ["f1_weighted", "accuracy", "aps", "logloss"]
        else:
            primary = "f1_weighted"
            secondary = ["accuracy", "roc_auc_ovr", "logloss"]
        return primary, secondary
    # regression
    primary = "r2"
    secondary = ["rmse", "mae"]
    return primary, secondary


def _select_cv(rows: int, problem_type: str, *, is_ts: bool, stratified: bool) -> tuple[str, int]:
    """
    Zwróć (kind, folds).
    - TimeSeries: 'tscv' (5 gdy sporo danych, inaczej 3)
    - Klasyfikacja: 'stratified' (3-5), większe zbiory -> 5
    - Regresja: 'kfold' (3-5)
    """
    if is_ts:
        folds = 5 if rows >= 4000 else 3
        return "tscv", folds
    if problem_type == "classification":
        if rows >= 8000:
            return "stratified", 5
        if rows >= 2000:
            return "stratified", 4
        return "stratified", 3
    # regression
    if rows >= 8000:
        return "kfold", 5
    if rows >= 2000:
        return "kfold", 4
    return "kfold", 3


# =========================
# Strategy / budget / tuning
# =========================

def _choose_strategy(rows: int, cols: int, override: Optional[str]) -> str:
    if override in {"fast_small", "balanced", "accurate", "advanced"}:
        return override
    # heurystyka
    if rows <= 10_000 and cols <= 50:
        return "fast_small"
    if rows <= 80_000 and cols <= 200:
        return "balanced"
    if rows <= 300_000 or cols <= 400:
        return "accurate"
    return "advanced"


def _decide_tuning(strategy: str, rows: int, cols: int) -> tuple[bool, int, int]:
    """
    (enable, n_trials, time_budget_min)
    """
    # Bazowy budżet zależny od strategii
    if strategy == "fast_small":
        return False, 0, 0
    if strategy == "balanced":
        n_trials = 20 if rows < 50_000 else 15
        return True, n_trials, 10
    if strategy == "accurate":
        n_trials = 40 if rows < 200_000 else 25
        return True, n_trials, 30
    # advanced
    n_trials = 80 if rows < 500_000 else 40
    return True, n_trials, 60


def _decide_ensembles(strategy: str, rows: int, cols: int, problem_type: str) -> tuple[bool, str]:
    """
    (enable, method)
    """
    # Na małych zbiorach blending jest tani i zwykle bezpieczny
    if strategy in {"balanced", "accurate", "advanced"}:
        if rows >= 2_000 or cols >= 25:
            return True, "blend"
        return True, "blend"
    # fast_small: ograniczmy koszty, ale zostawmy możliwość blenda
    if rows >= 5_000 or cols >= 40:
        return True, "blend"
    return False, "blend"


# =========================
# Plan dataclass
# =========================

@dataclass
class TrainingPlan:
    problem_type: str
    metrics_primary: str
    metrics_secondary: list[str]
    cv: dict
    test_size: float
    strategy: str
    tuning: dict
    ensembles: dict
    timeseries: dict
    n_jobs: int
    random_state: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# =========================
# Main API
# =========================

def build_training_plan(
    df: pd.DataFrame,
    target: str,
    *,
    strategy: str | None = None,
    hints: Dict[str, Any] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict:
    """
    Zbuduj kompletny plan trenowania na podstawie danych i heurystyk.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    y = df[target]
    # Wykryj problem i (opcjonalnie) TS
    problem_type = _detect_problem_type(y)
    h = hints or {}
    is_ts_hint = bool(h.get("timeseries", False))
    time_col_hint = h.get("time_col")
    is_ts_auto, time_col_auto, freq = _detect_timeseries(df, target)

    is_ts = bool(is_ts_hint or is_ts_auto)
    time_col = time_col_hint or time_col_auto

    # Metryki
    m_primary, m_secondary = _select_metrics(problem_type, y if not is_ts else y)

    # Rozmiary
    rows, cols = int(df.shape[0]), int(df.shape[1] - 1)  # bez targetu
    stratified = problem_type == "classification"

    # Strategia
    chosen_strategy = _choose_strategy(rows, cols, strategy)

    # CV
    cv_kind, cv_folds = _select_cv(rows, problem_type, is_ts=is_ts, stratified=stratified)

    # Test split (większe zbiory mogą oddać mniej na walidację)
    if is_ts:
        test_size = 0.2 if rows < 10_000 else 0.1
    else:
        test_size = 0.2 if rows < 50_000 else 0.15

    # Tuning & Ensembles
    t_enable, t_trials, t_budget = _decide_tuning(chosen_strategy, rows, cols)
    e_enable, e_method = _decide_ensembles(chosen_strategy, rows, cols, problem_type)

    notes: list[str] = []
    if is_ts:
        notes.append("Wykryto sygnały szeregów czasowych – zalecany TimeSeriesSplit.")
    if chosen_strategy in {"accurate", "advanced"}:
        notes.append("Priorytet jakości – większy budżet na tuning i dokładniejsza walidacja.")
    if not t_enable:
        notes.append("Tuning wyłączony w trybie fast_small (szybkie iteracje).")
    if e_enable:
        notes.append("Włączone ensembling (blend) – poprawa stabilności i jakości.")

    plan = TrainingPlan(
        problem_type="timeseries" if is_ts else problem_type,
        metrics_primary=m_primary,
        metrics_secondary=m_secondary,
        cv={"kind": cv_kind, "folds": int(cv_folds), "stratified": bool(stratified and not is_ts)},
        test_size=float(test_size),
        strategy=chosen_strategy,
        tuning={"enable": bool(t_enable), "n_trials": int(t_trials), "time_budget_min": int(t_budget)},
        ensembles={"enable": bool(e_enable), "method": e_method, "reason": "auto"},
        timeseries={"is_ts": bool(is_ts), "time_col": str(time_col) if time_col else None, "freq_inferred": freq},
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        notes=notes,
    )
    return plan.to_dict()


__all__ = ["build_training_plan"]
