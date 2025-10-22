# backend/ai_integration.py
"""
AI-assisted insights for TMIV – The Most Important Variables.

This module provides two core capabilities with SAFE fallbacks:
1) Column descriptions (semantic, concise) for any pandas DataFrame.
2) Recommendations (actions/risks/next-steps) given model metrics & dataset context.

Design goals
------------
- **No hard dependency** on external LLMs. If an API key & package are present, we can enrich
  results via LLM; otherwise robust **heuristics** are used.
- **Deterministic** output without keys. Heuristics rely on schema, statistics and naming rules.
- **Cache-friendly**. Hash-based caching prevents re-computation for the same dataset/inputs.
- **Pure functions**. Side-effect-free; callers handle Streamlit session, secrets, etc.

Public API
----------
- describe_columns(df: pd.DataFrame, *, llm: bool = True, max_per_type: int = 50) -> dict[str, str]
- recommend_actions(context: ModelContext) -> "AIRecommendations"

Typical usage
-------------
    import pandas as pd
    from backend.ai_integration import describe_columns, recommend_actions, ModelContext

    col_desc = describe_columns(df, llm=True)  # falls back to heuristics if no key/pkg
    recs = recommend_actions(ModelContext(
        problem_type="classification",
        metrics={"roc_auc": 0.84, "f1": 0.73, "accuracy": 0.78, "aps": 0.61},
        dataset_info={"rows": len(df), "cols": df.shape[1], "missing_pct": float(df.isna().mean().mean() * 100)},
        feature_importance=[("age", 0.21), ("income", 0.18), ("region", 0.11)],
        notes={"class_imbalance": True, "leakage_risk": False}
    ))

Implementation notes
--------------------
- LLM integration is optional and **best-effort** via OpenAI if available:
  - set env/secret: OPENAI_API_KEY
  - install package: openai>=1.0
- All outputs are short, safe, and business-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from typing import Any, Iterable, Mapping

import math
import os
import re

import pandas as pd


# =========================
# Data models
# =========================


@dataclass(frozen=True)
class ModelContext:
    """Context needed to produce actionable recommendations."""

    problem_type: str  # "classification" | "regression" | "timeseries"
    metrics: Mapping[str, float]  # e.g., {"roc_auc": 0.87, "f1": 0.76, "rmse": 1.25}
    dataset_info: Mapping[str, float]  # {"rows": 12345, "cols": 27, "missing_pct": 1.2}
    feature_importance: Iterable[tuple[str, float]] = field(default_factory=list)
    notes: Mapping[str, Any] = field(default_factory=dict)  # misc flags: imbalance/leakage/etc.


@dataclass(frozen=True)
class AIRecommendations:
    """Structured recommendation payload."""

    summary: str
    quick_wins: list[str]
    data_quality: list[str]
    modeling: list[str]
    risks: list[str]
    next_steps: list[str]


# =========================
# Utilities
# =========================


def _hash_dataframe(df: pd.DataFrame, max_cells: int = 5000) -> str:
    """Fast-ish fingerprint for caching across runs."""
    h = sha256()
    h.update(str(tuple(df.columns)).encode("utf-8"))
    # Sample up to max_cells values to keep hashing cheap
    n = min(max_cells // max(1, len(df.columns)), len(df))
    if n > 0:
        sample = df.head(n)
        # Convert to csv-like string for stable hashing
        h.update(pd.util.hash_pandas_object(sample.reset_index(drop=True), index=False).values.tobytes())
    return h.hexdigest()


def _snake_to_words(name: str) -> str:
    name = name.replace("-", "_")
    parts = re.split(r"[ _]+", name)
    parts2 = []
    for p in parts:
        # split camelCase
        parts2.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|\d+", p) or [p])
    return " ".join(p.lower() for p in parts2 if p)


def _guess_semantic(col: str) -> str:
    """Very small ontology based on common business terms."""
    n = col.lower()
    rules = [
        (r"\bage\b|\byears?\b|\bage_years?\b", "Wiek osoby / czasu trwania."),
        (r"\bprice\b|\bcost\b|\brevenue\b|\bsales?\b|\bamount\b", "Wartość finansowa (cena, koszt lub sprzedaż)."),
        (r"\bqty\b|\bquantity\b|\bvolume\b|\bunits?\b", "Ilość / wolumen."),
        (r"\bdate\b|\b_dt\b|\btime\b|\btimestamp\b", "Data/Czas (UTC?)."),
        (r"\bregion\b|\bstate\b|\bcountry\b|\bcity\b|\blocation\b", "Atrybut lokalizacji / geografia."),
        (r"\buser\b|\bcustomer\b|\bclient\b|\baccount\b", "Identyfikator klienta/konta."),
        (r"\bcategory\b|\bsegment\b|\bclass\b|\btype\b", "Kategoria/segment."),
        (r"\btarget\b|\blabel\b|\by\b|\bchurn\b|\bfraud\b", "Zmienna docelowa / wynik."),
        (r"\bproduct\b|\bsku\b|\bitem\b", "Identyfikator/rodzaj produktu."),
        (r"\bflag\b|\bbool\b|\bis_\w+", "Flaga logiczna (0/1, True/False)."),
    ]
    for pat, desc in rules:
        if re.search(pat, n):
            return desc
    return "Opis kolumny do ustalenia na podstawie kontekstu biznesowego."


def _dtype_hint(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "Typ logiczny (boolean)."
    if pd.api.types.is_integer_dtype(s):
        return "Liczba całkowita."
    if pd.api.types.is_float_dtype(s):
        return "Liczba zmiennoprzecinkowa."
    if pd.api.types.is_datetime64_any_dtype(s):
        return "Data/Czas."
    if pd.api.types.is_categorical_dtype(s) or s.dtype == "object":
        nunique = s.nunique(dropna=True)
        return f"Wartość kategoryczna/tekstowa (~{nunique} unikatów)."
    return f"Typ: {s.dtype}"


def _brief_stats(s: pd.Series) -> str:
    try:
        miss = s.isna().mean() * 100
        if pd.api.types.is_numeric_dtype(s):
            q = s.dropna().quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict()
            return f"Braki: {miss:.1f}%. Zakres: [{q[0.0]:.3g}, {q[1.0]:.3g}], mediana: {q[0.5]:.3g}."
        else:
            nunique = s.nunique(dropna=True)
            top = s.value_counts(dropna=True).head(3).to_dict()
            return f"Braki: {miss:.1f}%. Unikatów: {nunique}. Top: {list(top.items())[:3]}."
    except Exception:
        return "Statystyki niedostępne."


# =========================
# Column descriptions
# =========================


def _heuristic_column_descriptions(df: pd.DataFrame, max_per_type: int = 50) -> dict[str, str]:
    out: dict[str, str] = {}
    cols = list(df.columns)
    # Build short, useful descriptions
    for c in cols[: max_per_type if max_per_type > 0 else len(cols)]:
        s = df[c]
        words = _snake_to_words(str(c))
        sem = _guess_semantic(str(c))
        hint = _dtype_hint(s)
        stats = _brief_stats(s)
        out[str(c)] = f"{words} — {sem} {hint} {stats}"
    # For remaining, keep minimal info to avoid huge payloads
    if max_per_type and len(cols) > max_per_type:
        for c in cols[max_per_type:]:
            s = df[c]
            out[str(c)] = f"{_snake_to_words(str(c))} — {_dtype_hint(s)}"
    return out


def _try_openai_generate(prompt: str, *, model: str | None = None, max_tokens: int = 300) -> str | None:
    """Best-effort OpenAI call; returns None if API not configured/available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        # OpenAI >= 1.x
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analysis assistant. Respond concisely. "
                        "Return plain text; no Markdown tables."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content if resp.choices else None
    except Exception:
        return None


def _llm_column_descriptions(df: pd.DataFrame, *, sample_rows: int = 20, max_chars: int = 4000) -> dict[str, str] | None:
    """LLM-augmented descriptions if possible. Falls back by returning None."""
    # Construct compact schema preview
    schema_lines = []
    for c in df.columns[:100]:
        s = df[c]
        schema_lines.append(f"- {c}: dtype={str(s.dtype)}, missing={s.isna().mean()*100:.1f}%")

    # Include a brief data sample (truncated)
    try:
        sample = df.head(sample_rows).to_dict(orient="records")
    except Exception:
        sample = []

    content = (
        "Given the following pandas DataFrame schema and a small data sample, "
        "write a one-sentence, business-friendly description for each column. "
        "Return as 'column: description' per line.\n\n"
        f"SCHEMA:\n" + "\n".join(schema_lines) + "\n\n"
        f"SAMPLE (up to {sample_rows} rows):\n{str(sample)[:max_chars]}"
    )
    text = _try_openai_generate(content)
    if not text:
        return None

    # Parse simple "col: desc" lines back to dict; tolerate missing lines
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if key:
                out[key] = val
    if not out:
        return None
    # Ensure all columns have some description
    for c in df.columns:
        if c not in out:
            s = df[c]
            out[c] = f"{_snake_to_words(c)} — {_dtype_hint(s)}"
    return out


@lru_cache(maxsize=64)
def _describe_columns_cached(df_hash: str, *, use_llm: bool, max_per_type: int) -> dict[str, str]:
    # This cached layer expects a hash; the actual dataframe processing happens in wrappers.
    raise RuntimeError("This function should not be called directly without wrapper.")


def describe_columns(
    df: pd.DataFrame,
    *,
    llm: bool = True,
    max_per_type: int = 50,
) -> dict[str, str]:
    """
    Produce semantic descriptions for dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    llm : bool
        If True, attempts LLM enrichment (when configured). Otherwise uses heuristics.
    max_per_type : int
        Upper bound of fully-detailed columns; remaining columns get brief descriptions.

    Returns
    -------
    dict[str, str] : mapping column_name -> description
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    df_hash = _hash_dataframe(df)
    # Manual caching, because lru_cache can't hash DataFrames
    cache_key = f"{df_hash}:{int(llm)}:{max_per_type}"
    _GLOBAL_CACHE.setdefault("col_desc", {})
    if cache_key in _GLOBAL_CACHE["col_desc"]:
        return _GLOBAL_CACHE["col_desc"][cache_key]

    # Try LLM first if requested
    if llm:
        llm_out = _llm_column_descriptions(df)
        if llm_out:
            _GLOBAL_CACHE["col_desc"][cache_key] = llm_out
            return llm_out

    # Heuristic fallback
    heur = _heuristic_column_descriptions(df, max_per_type=max_per_type)
    _GLOBAL_CACHE["col_desc"][cache_key] = heur
    return heur


# =========================
# Recommendations
# =========================


def _score_band(value: float | None, good: float, ok: float, higher_is_better: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    v = float(value)
    if higher_is_better:
        if v >= good:
            return "good"
        if v >= ok:
            return "ok"
        return "poor"
    else:
        if v <= good:
            return "good"
        if v <= ok:
            return "ok"
        return "poor"


def recommend_actions(ctx: ModelContext) -> AIRecommendations:
    """Generate actionable, business-friendly recommendations from metrics & context."""
    pt = (ctx.problem_type or "").lower().strip()

    # Extract common metrics
    auc = ctx.metrics.get("roc_auc")
    f1 = ctx.metrics.get("f1")
    acc = ctx.metrics.get("accuracy")
    aps = ctx.metrics.get("aps") or ctx.metrics.get("average_precision")

    rmse = ctx.metrics.get("rmse")
    mae = ctx.metrics.get("mae")
    r2 = ctx.metrics.get("r2")

    rows = int(ctx.dataset_info.get("rows", 0) or 0)
    cols = int(ctx.dataset_info.get("cols", 0) or 0)
    missing_pct = float(ctx.dataset_info.get("missing_pct", 0.0) or 0.0)

    imbalance = bool(ctx.notes.get("class_imbalance", False))
    leakage = bool(ctx.notes.get("leakage_risk", False))
    drift_risk = bool(ctx.notes.get("drift_risk", False))

    # Score bands
    bands = []
    if pt == "classification":
        bands += [
            ("roc_auc", _score_band(auc, good=0.85, ok=0.75, higher_is_better=True)),
            ("f1", _score_band(f1, good=0.80, ok=0.70, higher_is_better=True)),
            ("accuracy", _score_band(acc, good=0.90, ok=0.80, higher_is_better=True)),
            ("aps", _score_band(aps, good=0.50, ok=0.30, higher_is_better=True)),
        ]
    elif pt == "regression":
        bands += [
            ("rmse", _score_band(rmse, good=0.0, ok=0.0, higher_is_better=False)),  # contextual
            ("mae", _score_band(mae, good=0.0, ok=0.0, higher_is_better=False)),
            ("r2", _score_band(r2, good=0.75, ok=0.50, higher_is_better=True)),
        ]
    elif pt == "timeseries":
        bands += [
            ("rmse", _score_band(rmse, good=0.0, ok=0.0, higher_is_better=False)),
            ("mae", _score_band(mae, good=0.0, ok=0.0, higher_is_better=False)),
            ("r2", _score_band(r2, good=0.60, ok=0.35, higher_is_better=True)),
        ]

    # Feature importance summary
    top_fi = list(ctx.feature_importance)[:5]
    top_names = [f for f, _ in top_fi]

    quick_wins: list[str] = []
    data_quality: list[str] = []
    modeling: list[str] = []
    risks: list[str] = []
    next_steps: list[str] = []

    # Data Quality
    if missing_pct > 5:
        data_quality.append(
            f"Uzupełnij/brakujące wartości (średnio {missing_pct:.1f}%). Rozważ imputację per-cecha i 'missing indicators'."
        )
    if rows < 1000 and pt in {"classification", "regression"}:
        data_quality.append("Zwiększ próbkę danych (rows < 1k). Rozważ augmentację / łączenie źródeł.")
    if cols > 200:
        data_quality.append("Redukcja wymiaru (PCA/UMAP) lub selekcja cech — aktualnie wiele kolumn.")

    # Classification-specific
    if pt == "classification":
        if bands and any(name == "f1" and b == "poor" for name, b in bands):
            quick_wins.append("Zbalansuj klasy (class weights/SMOTE) i dostrój threshold pod F1.")
        if imbalance:
            modeling.append("Użyj stratified CV i wagi klas w modelach (np. XGBoost scale_pos_weight).")
            risks.append("Silny imbalance może zaniżać F1/Recall mimo wysokiej AUC.")
        modeling.append("Zweryfikuj kalibrację prawdopodobieństw (Platt/Isotonic).")
        if top_names:
            next_steps.append(f"Zbadaj SHAP dla kluczowych cech: {', '.join(top_names)}.")

    # Regression-specific
    if pt == "regression":
        if r2 is not None and r2 < 0.5:
            quick_wins.append("Rozszerz featury (interakcje, transformaty), sprawdź nieliniowe modele (GBDT).")
        modeling.append("Sprawdź rozkład błędów i zakres outlierów; rozważ robust loss / transformację targetu.")

    # Time series
    if pt == "timeseries":
        modeling.append("Zadbaj o poprawny podział czasowy (TimeSeriesSplit) i walidację na najnowszym oknie.")
        next_steps.append("Dodaj święta/sezonowości/lag features; przetestuj LightGBM/XGBoost vs. klasyczne modele TS.")

    # Generic modeling
    modeling.append("Przetestuj ensembling (stacking/blending) i lekki tuning hiperparametrów.")
    if cols > 30:
        modeling.append("Wypróbuj selekcję cech (mutual_info, permutation importance) dla uproszczenia modelu.")

    # Risks
    if leakage:
        risks.append("Wykryto ryzyko leakage — sprawdź kolumny 'id/time/target leak' oraz czasowe zależności.")
    if drift_risk:
        risks.append("Ryzyko driftu — skonfiguruj monitoring rozkładów (PSI/JSD) i okresową rewalidację.")

    # Next steps
    next_steps.extend(
        [
            "Zapamiętaj champion model oraz parametry CV/metryki do bazy (reproducibility).",
            "Wygeneruj raport PDF Explainability dla biznesu i ZIP z artefaktami.",
        ]
    )

    # Compose summary
    quality_tag = "dobrze" if all(b in {"good", "ok", "unknown"} for _, b in bands) else "wymaga poprawy"
    main_metric = None
    if pt == "classification":
        main_metric = f"AUC={auc:.3f}" if auc is not None else None
    elif pt in {"regression", "timeseries"}:
        main_metric = f"R²={r2:.3f}" if r2 is not None else None

    summary_parts = [f"Problem: {pt}, obserwacje={rows:,}, cechy={cols:,}."]
    if main_metric:
        summary_parts.append(f"Główna metryka: {main_metric}.")
    summary_parts.append(f"Jakość danych: {quality_tag}.")
    if top_names:
        summary_parts.append(f"Najważniejsze cechy: {', '.join(top_names)}.")
    summary = " ".join(summary_parts)

    return AIRecommendations(
        summary=summary,
        quick_wins=sorted(set(quick_wins)),
        data_quality=sorted(set(data_quality)),
        modeling=sorted(set(modeling)),
        risks=sorted(set(risks)),
        next_steps=sorted(set(next_steps)),
    )


# =========================
# Global in-memory cache
# =========================

_GLOBAL_CACHE: dict[str, dict[str, Any]] = {}
