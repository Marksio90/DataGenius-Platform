from __future__ import annotations

from backend.safe_utils import truthy_df_safe

"""
EDAAnalyzer: eksploracyjna analiza danych z fallbackami (AI/SHAP/perm, SciPy),
auto-clean, raporty i wizualizacje Plotly, oraz AI data-prep helper.
"""

import hashlib
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Any, Optional
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Plotly (wizualizacje – bez twardej zależności od Streamlit)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SciPy (bezpieczny fallback) ---
try:
    from scipy import stats  # type: ignore
except Exception:
    class _StatsFallback:
        @staticmethod
        def skew(s):
            try:
                return float(pd.Series(s).skew())
            except Exception:
                return np.nan

        @staticmethod
        def kurtosis(s):
            try:
                # pandas.Series.kurt: Fisher’s definition (excess) – spójne z scipy.kurtosis default fisher=True
                return float(pd.Series(s).kurt())
            except Exception:
                return np.nan

        @staticmethod
        def zscore(s, nan_policy="omit"):
            s = pd.Series(s, dtype="float64")
            mu = s.mean()
            sd = s.std()
            if sd == 0 or not np.isfinite(sd):
                return np.zeros(len(s), dtype="float64")
            z = (s - mu) / sd
            return z.to_numpy()

        @staticmethod
        def entropy(values):
            v = np.asarray(values, dtype=float)
            tot = v.sum()
            if tot <= 0:
                return 0.0
            p = v / tot
            p = p[p > 0]
            return float(-(p * np.log(p)).sum())

        @staticmethod
        def shapiro(sample):
            # brak dokładnego testu – zwracamy neutralny p-value None
            return (None, None)

    stats = _StatsFallback()  # type: ignore

# --- Fuzzy dopasowywanie nazw ---
from difflib import SequenceMatcher

# --- HELPER: twarde wyrównanie do df.columns + fallback dla braków ---
def _normalize_key(x: str) -> str:
    import re
    return re.sub(r"[\W_]+", "", str(x or "").strip().lower())


# ======================================================================
#  Fallbacki i importy opcjonalne
# ======================================================================

# smart_cache (fallback, jeżeli brak backend.cache_manager)
try:
    from backend.cache_manager import smart_cache  # type: ignore
except Exception:
    class _DummyCache:
        def cache_decorator(self, ttl: int = 3600):
            def deco(fn):
                def wrapper(*a, **k):
                    return fn(*a, **k)
                return wrapper
            return deco
    smart_cache = _DummyCache()  # type: ignore

# AI/LLM opisy (opcjonalnie)
try:
    from backend.ai_integration import AIDescriptionGenerator  # type: ignore
except Exception:
    AIDescriptionGenerator = None  # type: ignore

# SHAP / Permutation importance (opcjonalnie)
try:
    from backend.eda_shap_utils import (  # type: ignore
        compute_permutation_importance,
        compute_shap_importance,
    )
except Exception:
    def compute_permutation_importance(*args, **kwargs):
        return None
    def compute_shap_importance(*args, **kwargs):
        return None

# Streamlit – używamy defensywnie, tylko gdy jest dostępny
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore


# ======================================================================
#  Pomocnicze: bezpieczna lokalizacja katalogu cache (żeby nie triggerować reloadów)
# ======================================================================

def _get_cache_base_dir() -> Path:
    """Zwraca bazowy katalog cache:
    1) jeśli smart_cache ma cache_dir → użyj go,
    2) TMIV_CACHE_DIR (env),
    3) gdy TMIV_CACHE_TO_TMP=1 → /tmp/tmiv_cache,
    4) fallback: 'cache' (w projekcie).
    """
    try:
        cd = getattr(smart_cache, "cache_dir", None)
        if cd:
            return Path(cd)
    except Exception:
        pass
    env_dir = os.getenv("TMIV_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    if os.getenv("TMIV_CACHE_TO_TMP") == "1":
        return Path(tempfile.gettempdir()) / "tmiv_cache"
    return Path("cache")


# ======================================================================
#  Klasa EDAAnalyzer
# ======================================================================

class EDAAnalyzer:
    """
    Eksploracyjna Analiza Danych (EDA) + auto-clean + raporty + wizualizacje.
    Wszystkie metody bezpieczne względem braków opcjonalnych zależności.
    """

    def __init__(self):
        self.analysis_cache: Dict[str, Any] = {}
        self.cache = smart_cache

    # ====================== AI / OPISY KOLUMN ======================

    def _align_descriptions(self, raw_desc: Dict[str, str], df: pd.DataFrame) -> Dict[str, str]:
        """
        Zwraca słownik {exact_column_name: opis} dla KAŻDEJ kolumny.
        1) próba dopasowania 1:1 (dokładna nazwa),
        2) dopasowanie po normalizacji (usunięcie znaków niedozwolonych / case),
        3) fuzzy match (SequenceMatcher) – ostrożny próg,
        4) brakujące kolumny uzupełnione symulacją.
        """
        if not isinstance(raw_desc, dict):
            raw_desc = {}

        # Spłaszcz ewentualny blok {"descriptions": {...}}
        if "descriptions" in raw_desc and isinstance(raw_desc["descriptions"], dict):
            raw_desc = raw_desc["descriptions"]

        # indeksy pomocnicze
        norm_to_text = {}
        for k, v in raw_desc.items():
            norm_to_text[_normalize_key(k)] = str(v)

        # opis symulacyjny dla braków
        sim_all = self._generate_descriptions_fallback(df)

        out: Dict[str, str] = {}
        for col in df.columns:
            key_exact = str(col)
            key_norm = _normalize_key(col)

            # 1) exact
            if key_exact in raw_desc:
                out[key_exact] = str(raw_desc[key_exact])
                continue
            # 2) normalized
            if key_norm in norm_to_text:
                out[key_exact] = norm_to_text[key_norm]
                continue

            # 3) fuzzy (gdy AI mocno zmieniło nazwę; próg konserwatywny)
            best = None; best_score = 0.0
            for k_raw, v in raw_desc.items():
                score = SequenceMatcher(None, key_norm, _normalize_key(k_raw)).ratio()
                if score > best_score:
                    best, best_score = v, score
            if best is not None and best_score >= 0.86:
                out[key_exact] = str(best)
                continue

            # 4) fallback – zawsze coś zwróć
            out[key_exact] = sim_all.get(key_exact, "Kolumna – opis niedostępny (fallback).")

        return out

    @smart_cache.cache_decorator(ttl=3600)
    def generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generuje inteligentne opisy kolumn przy użyciu AI (jeśli dostępne)
        lub fallbacku statystycznego. Zawsze zwraca deskrypcje dla KAŻDEJ kolumny
        dopasowane dokładnie do nazw w df.columns.
        """
        raw = None
        if AIDescriptionGenerator is not None:
            try:
                ai_gen = AIDescriptionGenerator()
                raw = ai_gen.generate_column_descriptions(df)
            except Exception:
                raw = None
        if raw is None:
            raw = self._generate_descriptions_fallback(df)
        return self._align_descriptions(raw, df)

    def generate_column_descriptions_cached(self, df: pd.DataFrame, *, force: bool = False) -> Dict[str, str]:
        """
        Non-destructive cache (dodatkowo dysk + session_state) niezależny od SmartCache:
        - kluczowany sygnaturą kolumn/dtypów i próbki,
        - gdy `force=True`, odświeża opisy,
        - ZAPIS DO KATALOGU CACHE SPOZA REPO, jeśli skonfigurowano (żeby nie triggerować reloadów Streamlit).
        """
        try:
            import json

            sig_payload = json.dumps(
                {
                    "cols": list(map(str, df.columns)),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "sample": df.head(50).to_json(orient="split", index=False)
                },
                sort_keys=True, ensure_ascii=False
            )
            sig = hashlib.md5(sig_payload.encode("utf-8")).hexdigest()

            base_dir = _get_cache_base_dir()
            cache_dir = base_dir / "column_descriptions"
            cache_dir.mkdir(parents=True, exist_ok=True)
            fp = cache_dir / f"{sig}.json"

            # Session cache (jeśli Streamlit dostępny)
            ss_key = f"col_desc::{sig}"
            if not force and st is not None and hasattr(st, "session_state") and ss_key in st.session_state:
                return self._align_descriptions(st.session_state[ss_key], df)

            # Disk cache
            if not force and fp.exists():
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if st is not None and hasattr(st, "session_state"):
                        st.session_state[ss_key] = data
                    return self._align_descriptions(data, df)
                except Exception:
                    pass

            # Fallback do właściwej metody
            data = self.generate_column_descriptions(df)
            if isinstance(data, dict):
                try:
                    tmp = fp.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp.replace(fp)  # zapis atomowy
                except Exception:
                    pass
                if st is not None and hasattr(st, "session_state"):
                    st.session_state[ss_key] = data
            return self._align_descriptions(data or {}, df)
        except Exception:
            try:
                return self.generate_column_descriptions(df)
            except Exception:
                return {}

    def _generate_descriptions_fallback(self, df: pd.DataFrame) -> Dict[str, str]:
        """Symulacja opisów kolumn bez AI – na bazie statystyki + heurystyk."""
        descriptions: Dict[str, str] = {}

        for column in df.columns:
            col_data = df[column]
            dtype = col_data.dtype
            null_count = col_data.isnull().sum()
            null_pct = float((null_count / len(col_data)) * 100) if len(col_data) else 0.0
            unique_count = int(col_data.nunique(dropna=True))
            unique_pct = float((unique_count / len(col_data)) * 100) if len(col_data) else 0.0

            parts: List[str] = []

            if pd.api.types.is_numeric_dtype(col_data):
                # statystyki
                cc = col_data.dropna()
                if len(cc) > 0:
                    mean_val = float(cc.mean())
                    median_val = float(cc.median())
                    std_val = float(cc.std())
                    min_val = float(cc.min())
                    max_val = float(cc.max())
                else:
                    mean_val = median_val = std_val = min_val = max_val = np.nan

                if unique_count <= 10:
                    parts.append(f"Zmienna kategoryczna numeryczna ({unique_count} unikalnych)")
                elif self._is_id_column(column, col_data):
                    parts.append("Prawdopodobnie kolumna identyfikacyjna (ID)")
                elif self._is_year_like(column, col_data):
                    parts.append("Kolumna reprezentująca rok/daty")
                elif self._is_percentage(col_data):
                    parts.append("Wartości najpewniej procentowe")
                elif self._is_binary_numeric(col_data):
                    parts.append("Zmienna binarna (0/1)")
                else:
                    parts.append("Zmienna numeryczna ciągła")

                if np.isfinite(min_val) and np.isfinite(max_val):
                    parts.append(f"Zakres: {min_val:.2f}–{max_val:.2f}")
                if np.isfinite(mean_val) and np.isfinite(median_val):
                    parts.append(f"Średnia: {mean_val:.2f}, Mediana: {median_val:.2f}")

                if np.isfinite(std_val) and np.isfinite(mean_val) and mean_val != 0 and std_val > abs(mean_val):
                    parts.append("Wysoka zmienność")

            else:
                # tekst/kategoria/datetime-like
                most_common = col_data.value_counts(dropna=True).head(3)
                if self._is_name_column(column):
                    parts.append("Prawdopodobnie nazwy/etykiety tekstowe")
                elif unique_pct > 90:
                    parts.append("Kolumna o bardzo wysokiej unikalności (ID tekstowe?)")
                elif self._is_datetime_series(col_data):
                    parts.append("Dane czasowe (daty/czas)")
                elif unique_count <= 10:
                    parts.append(f"Zmienna kategoryczna ({unique_count} kategorii)")
                else:
                    parts.append(f"Zmienna tekstowa ({unique_count} unikalnych)")

                if len(most_common) > 0:
                    parts.append("Najczęstsze: " + ", ".join(map(str, most_common.index[:3])))

            if null_count > 0:
                parts.append(f"⚠️ {null_pct:.1f}% braków")

            if unique_count == 1:
                parts.append("⚠️ Wszystkie wartości identyczne (bezużyteczna kolumna)")
            elif unique_pct < 1 and pd.api.types.is_numeric_dtype(col_data):
                parts.append("⚠️ Bardzo mała różnorodność wartości")

            descriptions[column] = ". ".join(parts)

        return descriptions

    # ====================== AUTO-CLEAN ======================

    def auto_clean(
        self,
        df: pd.DataFrame,
        progress_cb: Optional[Any] = None,
        log_cb: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Pipeline czyszczenia:
        - daty (parsowanie),
        - duplikaty,
        - kolumny z >60% NaN,
        - kolumny stałe,
        - imputacja (num=mediana, kat=moda),
        - winsoryzacja IQR.
        """
        def p(step: float, msg: str):
            if truthy_df_safe(progress_cb):
                try: progress_cb(step)
                except Exception: pass
            if truthy_df_safe(log_cb):
                try: log_cb(msg)
                except Exception: pass

        p(0.02, "🚀 Start auto-clean")
        dfc = df.copy()

        # 1) Daty
        date_like = [c for c in dfc.columns if any(k in c.lower() for k in ["date","czas","time","data","timestamp"])]
        for c in date_like:
            try:
                dfc[c] = pd.to_datetime(dfc[c], errors="ignore", utc=False, dayfirst=True)
            except Exception:
                pass
        p(0.10, f"📅 Daty przetworzone: {len(date_like)}")

        # 2) Duplikaty
        before = len(dfc)
        dfc = dfc.drop_duplicates()
        p(0.15, f"🗑️ Usunięte duplikaty: {before - len(dfc)}")

        # 3) Kolumny z >60% NaN
        try:
            miss_ratio = dfc.isna().mean(numeric_only=False)
        except Exception:
            miss_ratio = dfc.isna().mean()
        drop_cols = [c for c, r in miss_ratio.items() if r >= 0.6]
        if truthy_df_safe(drop_cols):
            dfc = dfc.drop(columns=drop_cols)
        p(0.25, f"❌ Kolumny >60% NaN: {len(drop_cols)}")

        # 4) Kolumny stałe
        nunique = dfc.nunique(dropna=False)
        const_cols = [c for c, u in nunique.items() if u <= 1]
        if truthy_df_safe(const_cols):
            dfc = dfc.drop(columns=const_cols)
        p(0.33, f"🔒 Stałe kolumny usunięte: {len(const_cols)}")

        # 5) Imputacja
        num_cols = dfc.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in dfc.columns if c not in num_cols]

        for c in num_cols:
            if dfc[c].isna().any():
                dfc[c] = dfc[c].fillna(dfc[c].median())

        for c in cat_cols:
            if dfc[c].isna().any():
                try:
                    mode_val = dfc[c].mode(dropna=True).iloc[0]
                except Exception:
                    mode_val = "UNKNOWN"
                dfc[c] = dfc[c].fillna(mode_val)
        p(0.55, f"🔧 Imputacja wykonana (num={len(num_cols)}, cat={len(cat_cols)})")

        # 6) Winsoryzacja IQR
        for c in num_cols:
            s = dfc[c]
            try:
                if s.notna().sum() > 20:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    if pd.notna(iqr) and iqr > 0:
                        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        dfc[c] = s.clip(lower=lo, upper=hi)
            except Exception:
                pass
        p(0.85, f"📊 Winsoryzacja IQR zakończona")

        p(0.96, "✅ Auto-clean gotowe")
        return dfc

    # ====================== RAPORT KOMPLEKSOWY ======================

    @smart_cache.cache_decorator(ttl=7200)
    def generate_comprehensive_eda_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Kompleksowy raport EDA (cache 2h)."""
        return {
            "basic_info": self._get_basic_info(df),
            "missing_data_analysis": self._analyze_missing_data(df),
            "numerical_analysis": self._analyze_numerical_columns(df),
            "categorical_analysis": self._analyze_categorical_columns(df),
            "correlation_analysis": self._analyze_correlations(df),
            "outlier_analysis": self._detect_outliers(df),
            "data_quality_issues": self._detect_data_quality_issues(df),
            "recommendations": self._generate_recommendations(df),
        }

    # --- Podmetody raportu ---

    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        return {
            "shape": tuple(df.shape),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
            "column_types": {
                "numeric": int(len(numeric_cols)),
                "categorical": int(len(categorical_cols)),
                "datetime": int(len(datetime_cols)),
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_values": int(df.isnull().sum().sum()),
        }

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100 if len(df) else 0.0
        missing_df = pd.DataFrame({
            "missing_count": missing_data,
            "missing_percentage": missing_pct
        }).sort_values("missing_percentage", ascending=False)

        patterns = self._find_missing_patterns(df)
        return {
            "missing_by_column": missing_df[missing_df["missing_count"] > 0].to_dict(),
            "columns_with_missing": missing_df[missing_df["missing_count"] > 0].index.tolist(),
            "missing_patterns": patterns,
            "total_complete_rows": int(len(df.dropna())),
            "rows_with_any_missing": int(len(df) - len(df.dropna())),
        }

    def _find_missing_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        if df.empty:
            return patterns
        miss = df.isnull()

        # Ochrona przed O(n^2) przy bardzo wielu kolumnach – limit do 80 kolumn o największych brakach
        cols = miss.columns
        if len(cols) > 80:
            miss_counts = miss.sum().sort_values(ascending=False)
            cols = miss_counts.head(80).index

        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                try:
                    corr = float(miss[c1].corr(miss[c2]))
                except Exception:
                    corr = np.nan
                if pd.notna(corr) and corr > 0.5:
                    patterns.append({
                        "columns": [c1, c2],
                        "correlation": corr,
                        "pattern_type": "correlated_missing",
                    })
        return patterns

    def _analyze_numerical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        out: Dict[str, Any] = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            try:
                d: Dict[str, Any] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "range": float(s.max() - s.min()),
                    "skewness": float(stats.skew(s)),
                    "kurtosis": float(stats.kurtosis(s)),
                    "coefficient_of_variation": float(s.std() / s.mean()) if s.mean() != 0 else float("inf"),
                }
            except Exception:
                d = {k: None for k in [
                    "mean","median","std","min","max","range","skewness","kurtosis","coefficient_of_variation"
                ]}
            if len(s) > 8:
                try:
                    res = stats.shapiro(s.sample(min(5000, len(s)), random_state=42))
                    pval = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else None
                    d["normality_test_p_value"] = float(pval) if pval is not None else None
                    d["is_normal"] = bool(pval > 0.05) if pval is not None else None
                except Exception:
                    d["normality_test_p_value"] = None
                    d["is_normal"] = None
            out[col] = d
        return out

    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        out: Dict[str, Any] = {}
        for col in cat_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            vc = s.value_counts()
            d: Dict[str, Any] = {
                "unique_count": int(s.nunique()),
                "most_frequent": vc.index[0] if len(vc) > 0 else None,
                "most_frequent_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "least_frequent": vc.index[-1] if len(vc) > 0 else None,
                "least_frequent_count": int(vc.iloc[-1]) if len(vc) > 0 else 0,
                "entropy": float(stats.entropy(vc.values)) if vc.sum() > 0 else 0.0,
                "top_10_categories": {str(k): int(v) for k, v in vc.head(10).to_dict().items()},
            }
            if len(vc) > 1 and vc.iloc[-1] != 0:
                d["imbalance_ratio"] = float(vc.iloc[0] / vc.iloc[-1])
                d["is_highly_imbalanced"] = bool(d["imbalance_ratio"] > 10)
            else:
                d["imbalance_ratio"] = None
                d["is_highly_imbalanced"] = None
            out[col] = d
        return out

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}
        corr = df[numeric_cols].corr()
        strong: List[Dict[str, Any]] = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                v = float(corr.iloc[i, j])
                if pd.notna(v) and abs(v) > 0.7:
                    strong.append({
                        "var1": corr.columns[i],
                        "var2": corr.columns[j],
                        "correlation": v,
                        "strength": "very_strong" if abs(v) > 0.9 else "strong",
                    })
        return {
            "correlation_matrix": {c: {r: float(v) for r, v in corr[c].items()} for c in corr.columns},
            "strong_correlations": strong,
            "multicollinearity_warning": len(strong) > 0,
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if s.empty:
                continue
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            iqr_mask = (s < low) | (s > high)
            try:
                z = np.abs(stats.zscore(s, nan_policy="omit"))
                z_mask = pd.Series(z, index=s.index) > 3
            except Exception:
                z_mask = pd.Series(False, index=s.index)
            out[col] = {
                "iqr_outliers_count": int(iqr_mask.sum()),
                "iqr_outliers_percentage": float((iqr_mask.sum() / len(s)) * 100),
                "zscore_outliers_count": int(z_mask.sum()),
                "zscore_outliers_percentage": float((z_mask.sum() / len(s)) * 100),
                "outlier_bounds": {"lower": float(low), "upper": float(high)},
            }
        return out

    def _detect_data_quality_issues(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        dup = int(df.duplicated().sum())
        if dup > 0:
            issues.append({
                "type": "duplicates",
                "severity": "medium",
                "description": f"Znaleziono {dup} zduplikowanych wierszy",
                "recommendation": "Rozważ usunięcie duplikatów lub ich uzasadnienie",
            })
        # stałe kolumny
        for col in df.columns:
            try:
                if df[col].nunique(dropna=True) == 1:
                    issues.append({
                        "type": "constant_column",
                        "severity": "high",
                        "description": f"Kolumna '{col}' ma tylko jedną wartość",
                        "recommendation": "Usuń – kolumna nie wnosi informacji",
                    })
            except Exception:
                pass
        # dużo braków
        if len(df) > 0:
            miss_pct = (df.isnull().sum() / len(df)) * 100
            for col in miss_pct[miss_pct > 50].index.tolist():
                issues.append({
                    "type": "high_missing_data",
                    "severity": "high",
                    "description": f"Kolumna '{col}' ma {miss_pct[col]:.1f}% braków",
                    "recommendation": "Imputuj, uzupełnij u źródła lub usuń kolumnę",
                })
        # problemy z encoding
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                if df[col].astype(str).str.contains("�", na=False).any():
                    issues.append({
                        "type": "encoding_issue",
                        "severity": "medium",
                        "description": f"Kolumna '{col}' może mieć problem z kodowaniem",
                        "recommendation": "Sprawdź i wymuś poprawne kodowanie pliku (UTF-8/CP1250 itd.)",
                    })
            except Exception:
                pass
        return issues

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        recs: List[str] = []
        if len(df) > 0:
            miss_pct = (df.isnull().sum() / len(df)) * 100
            cols_with_missing = miss_pct[miss_pct > 0].index.tolist()
            if truthy_df_safe(cols_with_missing):
                recs.append("🔍 Zbadaj brakujące dane: " + ", ".join(cols_with_missing[:3]))

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            rngs = df[num_cols].max() - df[num_cols].min()
            try:
                ratio = (rngs.replace(0, np.nan).max() / rngs.replace(0, np.nan).min())
                if np.isfinite(ratio) and ratio > 100:
                    recs.append("📊 Rozważ standaryzację/normalizację – bardzo różne zakresy cech")
            except Exception:
                pass

        outliers = self._detect_outliers(df)
        high_out = [c for c, d in outliers.items() if d["iqr_outliers_percentage"] > 5]
        if truthy_df_safe(high_out):
            recs.append("⚠️ Sprawdź outliers w kolumnach: " + ", ".join(high_out[:3]))

        corr = self._analyze_correlations(df)
        if corr.get("multicollinearity_warning"):
            recs.append("🔗 Silne korelacje – rozważ redukcję wymiarowości / usunięcie redundantnych cech")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            recs.append("🛠️ Użyj encodowania zmiennych kategorycznych (one-hot, target/label encoding)")

        return recs

    # ====================== WIZUALIZACJE ======================

    def create_eda_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tworzy wykresy EDA – bezpośrednio do użycia w Streamlit (st.plotly_chart)."""
        visuals: Dict[str, Any] = {}

        # Braki
        missing = df.isnull().sum()
        if int(missing.sum()) > 0:
            m = missing[missing > 0]
            fig_missing = px.bar(
                x=m.index,
                y=m.values,
                title="🔍 Brakujące dane w kolumnach",
                labels={"x": "Kolumny", "y": "Liczba braków"},
            )
            fig_missing.update_layout(height=400, margin=dict(l=50, r=50, t=80, b=50))
            visuals["missing_data"] = fig_missing

        # Korelacje
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="🔥 Macierz korelacji",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
            )
            fig_corr.update_layout(
                height=min(600, max(400, len(num_cols) * 40)),
                margin=dict(l=100, r=100, t=100, b=50),
                xaxis={"side": "bottom"},
            )
            visuals["correlation_matrix"] = fig_corr

        # Rozkłady
        if len(num_cols) > 0:
            cols_to_plot = num_cols[:6]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

            fig_dist = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[f"Rozkład: {c}" for c in cols_to_plot],
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
            )
            for i, c in enumerate(cols_to_plot):
                r = i // n_cols + 1
                cpos = i % n_cols + 1
                fig_dist.add_trace(
                    go.Histogram(x=df[c], name=c, showlegend=False, nbinsx=30),
                    row=r, col=cpos
                )
            fig_dist.update_layout(
                title_text="📊 Rozkłady zmiennych numerycznych",
                height=max(400, 300 * n_rows),
                margin=dict(l=50, r=50, t=100, b=50),
                showlegend=False,
            )
            visuals["distributions"] = fig_dist

        return visuals

    # ====================== FEATURE IMPORTANCE ======================

    def get_feature_importance(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        problem: str
    ) -> Dict[str, Any]:
        """Zwraca {'permutation': df?, 'shap': df?} jeżeli możliwe."""
        out: Dict[str, Any] = {}
        try:
            perm = compute_permutation_importance(estimator, X, y, problem=problem)
            if perm is not None:
                out["permutation"] = perm
        except Exception:
            pass
        try:
            shap_imp = compute_shap_importance(estimator, X)
            if shap_imp is not None:
                out["shap"] = shap_imp
        except Exception:
            pass
        return out

    # ====================== Heurystyki pomocnicze ======================

    def _is_id_column(self, name: str, s: pd.Series) -> bool:
        keys = ["id", "index", "key", "pk", "primary"]
        n = name.lower()
        try:
            numeric_like = str(s.dtype).startswith("int") or str(s.dtype).startswith("uint")
            return (
                any(k in n for k in keys)
                or s.nunique(dropna=True) == len(s.dropna())
                or (numeric_like and s.min() >= 0 and s.nunique(dropna=True) > len(s.dropna()) * 0.9)
            )
        except Exception:
            return any(k in n for k in keys)

    def _is_year_like(self, name: str, s: pd.Series) -> bool:
        keys = ["year", "rok", "date", "time"]
        n = name.lower()
        try:
            numeric_like = str(s.dtype).startswith("int")
            s_nn = s.dropna()
            in_range = (
                numeric_like and len(s_nn) > 0
                and 1900 <= s_nn.min() <= 2035
                and 1900 <= s_nn.max() <= 2035
            )
            return any(k in n for k in keys) or in_range
        except Exception:
            return any(k in n for k in keys)

    def _is_percentage(self, s: pd.Series) -> bool:
        s = pd.to_numeric(s, errors="coerce")
        s = s.dropna()
        if s.empty:
            return False
        try:
            return (s.min() >= 0 and s.max() <= 1) or (s.min() >= 0 and s.max() <= 100)
        except Exception:
            return False

    def _is_binary_numeric(self, s: pd.Series) -> bool:
        u = set(pd.Series(s).dropna().unique().tolist())
        return u.issubset({0, 1}) or u.issubset({0.0, 1.0})

    def _is_name_column(self, name: str) -> bool:
        return any(k in name.lower() for k in ["name", "nazwa", "title", "tytuł", "label"])

    def _is_datetime_series(self, s: pd.Series) -> bool:
        try:
            sample = s.dropna().head(10)
            if sample.empty:
                return False
            pd.to_datetime(sample, errors="raise")
            return True
        except Exception:
            return False


# ======================================================================
#  Dodatkowe narzędzia (jeśli korzystasz z AI data-prep poza klasą)
# ======================================================================

def apply_ai_dataprep(df: pd.DataFrame, target: str, plan: dict) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    AI data-prep:
      - harmonizacja typów,
      - parsowanie i featuryzacja dat (rok/mies/dzień/dow/kwartał itd.),
      - outliers IQR clip (bez targetu),
      - imputacja: num=mediana, kat='missing',
      - usunięcie duplikatów i kolumn stałych.
    Zwraca: (df_prepared, log_steps).
    """
    log: List[Dict[str, Any]] = []
    X = df.copy()
    log.append({"name": "Wczytanie", "detail": f"kształt={X.shape}"})

    # category -> object
    cats = [c for c in X.columns if str(X[c].dtype).startswith("category")]
    if truthy_df_safe(cats):
        for c in cats:
            X[c] = X[c].astype("object")
        log.append({"name": "Konwersja dtype", "detail": f"category→object: {cats}"})

    # Parsowanie dat z obiektów
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    parsed: List[str] = []
    for c in obj_cols:
        ser = pd.to_datetime(X[c], errors="coerce", utc=False, infer_datetime_format=True)
        ok = ser.notna().mean() if len(ser) else 0.0
        if ok >= 0.8:
            X[c] = ser
            parsed.append(c)
    if truthy_df_safe(parsed):
        log.append({"name": "Parsowanie dat", "detail": f"kolumny={parsed}"})

    # Ekstrakcja cech z datetime i drop oryginałów
    dt_cols = X.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    made: List[str] = []
    for c in dt_cols:
        ser = pd.to_datetime(X[c], errors="coerce")
        X[f"{c}__year"] = ser.dt.year.astype("Int64")
        X[f"{c}__month"] = ser.dt.month.astype("Int64")
        X[f"{c}__day"] = ser.dt.day.astype("Int64")
        X[f"{c}__dayofweek"] = ser.dt.dayofweek.astype("Int64")
        X[f"{c}__quarter"] = ser.dt.quarter.astype("Int64")
        X[f"{c}__is_month_start"] = ser.dt.is_month_start.astype("Int64")
        X[f"{c}__is_month_end"] = ser.dt.is_month_end.astype("Int64")
        made.append(c)
    if truthy_df_safe(made):
        X.drop(columns=made, inplace=True, errors="ignore")
        log.append({"name": "Ekstrakcja cech z dat", "detail": f"usunięto oryginały: {made}"})

    # Duplikaty
    before = len(X)
    X = X.drop_duplicates()
    removed = before - len(X)
    if removed > 0:
        log.append({"name": "Usunięto duplikaty wierszy", "detail": f"-{removed} wierszy"})

    # Stałe kolumny
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if truthy_df_safe(const_cols):
        X.drop(columns=const_cols, inplace=True, errors="ignore")
        log.append({"name": "Usunięto kolumny stałe", "detail": f"kolumny={const_cols}"})

    # Outliers – IQR clip (bez targetu)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    num_for_outliers = [c for c in num_cols if c != target]
    clipped: List[str] = []
    for c in num_for_outliers:
        s = X[c]
        if s.isna().all():
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X[c] = s.clip(lower=low, upper=high)
        clipped.append(c)
    if truthy_df_safe(clipped):
        log.append({"name": "Outliers IQR-clip", "detail": f"kolumny={clipped[:10]}{'...' if len(clipped)>10 else ''}"})

    # Imputacja
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in X.select_dtypes(include=["object", "category"]).columns:
        if X[c].isna().any():
            X[c] = X[c].fillna("missing")
    log.append({"name": "Imputacja braków", "detail": "num=mediana, kat='missing'"})

    log.append({"name": "Zakończono data-prep", "detail": f"kształt={X.shape}"})
    return X, log
