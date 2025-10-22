# core/services/insights_service.py
"""
InsightsService – AI-owe (lub heurystyczne) opisy kolumn i rekomendacje
dla TMIV – Advanced ML Platform.

Implementuje `IInsightsService` (zob. core/interfaces.py).

Cechy:
- Jeśli dostępny jest moduł `backend.ai_integration` oraz są klucze,
  to deleguje generowanie treści do LLM (z bezpiecznym fallbackiem).
- Gdy LLM niedostępny: szybkie, deterministyczne heurystyki (PL).
- Zero twardych zależności UI/Streamlit.

Public API:
- describe_columns(df) -> dict[column -> opis]
- generate_recommendations(context) -> list[str]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---- Opcjonalna warstwa AI (łagodny import) ----
_AI = None
try:  # pragma: no cover
    import backend.ai_integration as _AI  # type: ignore
except Exception:  # pragma: no cover
    _AI = None  # type: ignore

# ---- Pomoc: bezpieczeństwo / hashowanie danych (cache) ----
try:  # pragma: no cover
    from backend.cache_manager import df_fingerprint
except Exception:  # pragma: no cover
    def df_fingerprint(df: pd.DataFrame) -> str:
        sample = df.head(1000)
        payload = {
            "cols": list(map(str, sample.columns)),
            "dtypes": [str(t) for t in sample.dtypes],
            "shape": list(df.shape),
        }
        import hashlib, json

        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12].upper()


@dataclass
class _ColMeta:
    name: str
    kind: str          # "numeric" | "categorical" | "datetime" | "boolean" | "text" | "unknown"
    missing_pct: float
    unique: int
    unique_ratio: float


class InsightsService:
    # ========================================
    # Public API
    # ========================================

    def describe_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Zwraca krótkie, zrozumiałe opisy kolumn (PL).
        Priorytet:
        1) backend.ai_integration.describe_columns(df) (jeśli dostępne),
        2) heurystyka lokalna.
        """
        if df is None or df.empty:
            return {}

        # Spróbuj LLM (bez wywracania się, jeśli brak)
        if _AI is not None:
            for attr in ("describe_columns", "describe_columns_ai"):
                if hasattr(_AI, attr):
                    try:
                        result = getattr(_AI, attr)(df)  # type: ignore[misc]
                        if isinstance(result, dict) and result:
                            return {str(k): str(v) for k, v in result.items()}
                    except Exception:
                        pass  # fallback do heurystyki

        # Heurystyka
        metas = self._infer_meta(df)
        out: Dict[str, str] = {}
        for m in metas:
            out[m.name] = self._build_column_description(df[m.name], m)
        return out

    def generate_recommendations(self, context: Dict[str, Any]) -> Sequence[str]:
        """
        Zwraca listę rekomendacji (PL) na podstawie kontekstu trenowania/wyników.

        Oczekiwane (nieobowiązkowe) pola w `context`:
          - problem_type: "classification" | "regression" | "timeseries"
          - metrics: dict (np. {"roc_auc": 0.91, "f1_weighted": 0.84})
          - cv: dict (np. {"kind":"stratified","folds":5})
          - feature_importance: pd.DataFrame (kolumny: feature, importance)
          - class_balance: dict[label->count] lub {'minority_ratio': 0.1}
          - rows, cols: int
          - calibration_needed: bool
        """
        recs: List[str] = []

        # 1) Spróbuj LLM
        if _AI is not None:
            for attr in ("generate_recommendations", "generate_recommendations_ai"):
                if hasattr(_AI, attr):
                    try:
                        ai_recs = getattr(_AI, attr)(context)  # type: ignore[misc]
                        if isinstance(ai_recs, (list, tuple)) and ai_recs:
                            recs.extend([str(x) for x in ai_recs if str(x).strip()])
                            break
                    except Exception:
                        pass  # przejdź do heurystyk

        # 2) Heurystyki (zawsze można dodać; potem deduplikujemy)
        pt = str(context.get("problem_type", "") or "").lower()
        metrics = context.get("metrics") or {}
        rows = int(context.get("rows", 0) or 0)
        cols = int(context.get("cols", 0) or 0)
        cv = context.get("cv") or {}
        fi = context.get("feature_importance")
        class_balance = context.get("class_balance") or {}
        needs_cal = bool(context.get("calibration_needed", False))

        # — Ogólne
        if rows and cols:
            if rows < 2000 and cols > 50:
                recs.append("Rozważ selekcję cech lub regularizację – wysoka liczba cech przy małej próbce może prowadzić do przeuczenia.")
            if rows > 50_000 and cols > 5:
                recs.append("Włącz batchowe przetwarzanie i sprawdź algorytmy gradientowe (LightGBM/XGBoost) dla skalowalności.")
        if cv:
            if cv.get("kind") == "stratified":
                recs.append("Stratyfikowana walidacja utrzyma proporcje klas – to właściwy wybór dla klasyfikacji.")
            if cv.get("kind") == "tscv":
                recs.append("Dla danych czasowych używaj TimeSeriesSplit, aby uniknąć przecieku czasowego.")

        # — Klasyfikacja
        if pt == "classification":
            roc_auc = _get(metrics, ["roc_auc", "roc_auc_ovr", "roc_auc_ovo"])
            f1w = _get(metrics, ["f1_weighted", "f1"])
            acc = _get(metrics, ["accuracy"])

            if roc_auc is not None and roc_auc < 0.7:
                recs.append("AUC < 0.7 – spróbuj wzmocnić sygnał: inżynieria cech, inne algorytmy, kalibracja progów.")
            if f1w is not None and acc is not None and (acc - f1w) > 0.1:
                recs.append("Duża różnica ACC vs F1 sugeruje niezbalansowane klasy – rozważ wagi klas, oversampling (SMOTE) lub metryki robustowe.")
            if needs_cal:
                recs.append("Skalibruj prawdopodobieństwa (Platt/Isotonic), aby poprawić jakość decyzji przy progach biznesowych.")

            # Nierównowaga klas
            minority_ratio = None
            if isinstance(class_balance, dict):
                if "minority_ratio" in class_balance:
                    minority_ratio = float(class_balance["minority_ratio"])
                elif class_balance:
                    total = sum(int(v) for v in class_balance.values())
                    if total > 0:
                        minority_ratio = min(int(v) for v in class_balance.values()) / total
            if minority_ratio is not None and minority_ratio < 0.1:
                recs.append("Bardzo niezbalansowane klasy (<10%) – zastosuj class_weight, threshold tuning i odpowiednie metryki (PR-AUC, F1).")

        # — Regresja
        if pt == "regression":
            r2 = _get(metrics, ["r2"])
            rmse = _get(metrics, ["rmse"])
            if r2 is not None and r2 < 0.3:
                recs.append("Niskie R² – rozważ transformację celu (log/yeo-johnson), dodanie nieliniowości lub cech interakcyjnych.")
            if rmse is not None and rows and rmse > 0:
                recs.append("Zweryfikuj jednostki RMSE oraz rozkład błędów; w razie skośności użyj RobustScaler lub HuberLoss.")

        # — Feature importance
        if isinstance(fi, pd.DataFrame) and not fi.empty:
            topk = (fi.sort_values(by=fi.columns[1] if fi.shape[1] > 1 else fi.columns[0], ascending=False).head(5))
            top_feats = [str(topk.iloc[i, 0]) for i in range(min(5, len(topk)))]
            if top_feats:
                recs.append(f"Największy wpływ mają: {', '.join(top_feats)} — rozważ dogłębne sprawdzenie ich stabilności i driftu.")

        # — Porządkowe
        recs.append("Zadbaj o replikowalność: stały `random_state`, logowanie wersji danych i parametrów.")
        recs.append("Po wyborze modelu końcowego przeprowadź walidację na odłożonym zbiorze (hold-out) i monitoring produkcyjny (drift, kalibracja).")

        # Deduplikacja i porządek
        return _dedup(recs)

    # ========================================
    # Heurystyki – kolumny
    # ========================================

    def _infer_meta(self, df: pd.DataFrame) -> List[_ColMeta]:
        rows = int(df.shape[0]) or 1
        metas: List[_ColMeta] = []
        for c in df.columns:
            s = df[c]
            kind = _infer_kind(s)
            miss = float(s.isna().mean() * 100.0)
            uniq = int(s.nunique(dropna=True))
            ur = float(uniq / rows)
            metas.append(_ColMeta(name=str(c), kind=kind, missing_pct=miss, unique=uniq, unique_ratio=ur))
        return metas

    def _build_column_description(self, s: pd.Series, m: _ColMeta) -> str:
        name = m.name
        miss = f"{m.missing_pct:.1f}% braków"
        # ID-like?
        id_like = m.unique_ratio > 0.9 and m.unique > 50
        parts: List[str] = []
        if id_like:
            parts.append(f"'{name}' wygląda na identyfikator (unikalność {m.unique_ratio:.0%}).")
        if m.kind == "numeric":
            desc = _num_desc(s)
            parts.append(f"Zm. numeryczna; {miss}. {desc}")
        elif m.kind == "categorical":
            top = _top_cat(s)
            parts.append(f"Zm. kategoryczna; {miss}. {top}")
        elif m.kind == "boolean":
            ratio = _bool_ratio(s)
            parts.append(f"Flaga logiczna; {miss}. Proporcja True≈{ratio}")
        elif m.kind == "datetime":
            rng = _dt_range(s)
            parts.append(f"Data/Czas; {miss}. {rng}")
        elif m.kind == "text":
            ln = _text_len(s)
            parts.append(f"Tekst; {miss}. {ln}")
        else:
            parts.append(f"Typ nieokreślony; {miss}.")
        return " ".join(parts).strip()


# ========================================
# Helpers: typy i mini-opisy
# ========================================

def _infer_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        # odróżnij „tekst swobodny” od kategorii po kardynalności
        nunique = int(s.nunique(dropna=True))
        n = int(s.notna().sum())
        if n and (nunique / n) > 0.5:
            return "text"
        return "categorical"
    return "unknown"


def _num_desc(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce")
    arr = x.to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "Brak wartości numerycznych."
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    # skośność (bez twardej zależności na scipy)
    try:
        skew = float(pd.Series(arr).skew())
    except Exception:
        skew = 0.0
    skew_txt = "dodatnio skośny" if skew > 0.75 else ("ujemnie skośny" if skew < -0.75 else "w miarę symetryczny")
    return f"Śr≈{mean:.3g}, SD≈{std:.3g}, mediana≈{med:.3g}, IQR≈{(q75 - q25):.3g}, rozkład {skew_txt}."


def _top_cat(s: pd.Series, k: int = 5) -> str:
    vc = s.astype("string").value_counts(dropna=False).head(k)
    items = []
    for val, cnt in vc.items():
        label = "" if pd.isna(val) else str(val)
        items.append(f"{label} ({cnt})")
    return "Top: " + ", ".join(items) + "." if items else "Rozkład kategorii niedostępny."


def _bool_ratio(s: pd.Series) -> str:
    try:
        true = int((s == True).sum())  # noqa: E712
        total = int(s.notna().sum())
        r = true / total if total else 0.0
        return f"{r:.0%} z nie-NA"
    except Exception:
        return "brak danych"


def _dt_range(s: pd.Series) -> str:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() == 0:
        return "brak poprawnych dat."
    return f"zakres {str(dt.min())[:19]} → {str(dt.max())[:19]}."


def _text_len(s: pd.Series) -> str:
    try:
        ln = s.astype("string").str.len()
        return f"średnia długość ≈ {float(ln.mean()):.1f} znaków."
    except Exception:
        return "brak estymacji długości."


def _get(d: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def _dedup(recs: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for r in recs:
        rr = " ".join(r.split()).strip().rstrip(".")
        if not rr:
            continue
        key = rr.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(rr + ".")
    return out


__all__ = ["InsightsService"]
