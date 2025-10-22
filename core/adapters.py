import asyncio
import inspect
import hashlib
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import pandas as pd

try:
    import streamlit as st  # type: ignore
except Exception:  # fallback jeśli streamlit niedostępny w danym środowisku
    st = None  # type: ignore

from core.interfaces import (
    MLTrainerInterface,
    AIInsightsInterface,
    CacheServiceInterface,
    DataRepositoryInterface,
    TrainingData,
    TrainingResults,
)

# ---------------------------------------------------------------------
# Pomocnicze: bezpieczne wywołania sync/async i offload do wątku
# ---------------------------------------------------------------------


async def _maybe_await(obj: Union[Awaitable, Any]) -> Any:
    """Jeśli 'obj' jest awaitable → await, inaczej zwróć jak jest."""
    if inspect.isawaitable(obj):
        return await obj  # type: ignore[func-returns-value]
    return obj


async def _call_maybe_async(fn: Callable, *args, **kwargs) -> Any:
    """
    Wywołaj funkcję, która może być sync lub async.
    - Jeśli to coroutine function → await bezpośrednio.
    - W przeciwnym razie uruchom w wątku (nie blokuj event loopa).
    Uwaga: NIE wywołujemy `fn` przed decyzją, by uniknąć podwójnego wykonania.
    """
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)  # type: ignore[misc]
    # sync → do wątku
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)  # py>=3.9
    except AttributeError:  # py<3.9
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def _infer_problem_from_metrics(metrics: Dict[str, float]) -> str:
    """Prosta heurystyka rozpoznania problemu po metrykach."""
    # Regresja: klasyczne regresyjne metryki
    if any(
        k in metrics
        for k in (
            "r2",
            "rmse",
            "mae",
            "mape",
            "smape",
            "explained_variance",
            "median_ae",
            "max_error",
        )
    ):
        return "Regression"
    # Klasyfikacja
    if any(k in metrics for k in ("accuracy", "f1_weighted", "precision", "recall", "roc_auc")):
        return "Classification"
    # Fallback
    return "Classification"


# =====================================================================
# MLTrainerAdapter
# =====================================================================


class MLTrainerAdapter:
    """Adapter łączący stary MLModelTrainer z nowym interface."""

# AIInsightsAdapter
# =====================================================================


class AIInsightsAdapter:
    """Adapter dla AI Insights Generator."""

    def __init__(self, generator: Any):
        self.generator = generator

    async def generate_insights(self, results: TrainingResults) -> Dict[str, Any]:
        """
        Generuj insights na podstawie wyników ML.
        Obsługa obu światów: metody sync/async i różne kształty generatora.
        """
        # Zbuduj „stary” format dla kompatybilności:
        results_dict: Dict[str, Any] = {
            "best_model": results.best_model,
            "model_scores": results.model_scores,
            "feature_importance": results.feature_importance,
            **(results.metrics or {}),
        }

        # Detekcja problemu na podstawie metryk:
        problem = _infer_problem_from_metrics(results.metrics or {})
        target_placeholder = "target"

        # 1) Preferowana metoda: generate_recommendations
        if hasattr(self.generator, "generate_recommendations"):
            try:
                return await _call_maybe_async(
                    self.generator.generate_recommendations, results_dict, target_placeholder, problem
                )
            except Exception:
                # miękki fallback niżej
                pass

        # 2) Fallback: prosty zestaw rekomendacji
        return {
            "key_insights": "Automatyczne podsumowanie: analiza ML zakończona.",
            "action_items": ["Monitoruj metryki produkcyjne", "Rozważ wzbogacenie danych"],
            "limitations": "Wyniki oparte na aktualnym zbiorze danych",
            "next_steps": ["Wdrożenie modelu", "Konfiguracja monitoringu i alertów"],
        }

    async def generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generuj opisy kolumn (sync/async-safe)."""
        if hasattr(self.generator, "generate_column_descriptions"):
            return await _call_maybe_async(self.generator.generate_column_descriptions, df)
        # Fallback: proste opisy
        return {str(c): f"Column '{c}' (dtype={df[c].dtype})" for c in df.columns}

    async def generate_training_recommendations(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Generuj rekomendacje treningu (sync/async-safe)."""
        if hasattr(self.generator, "generate_training_recommendations"):
            return await _call_maybe_async(self.generator.generate_training_recommendations, df, target)
        # Fallback
        return {
            "cv_folds": 5,
            "train_size": 0.8,
            "notes": "Brak generatora – zastosowano domyślne rekomendacje.",
        }


# =====================================================================
# CacheServiceAdapter
# =====================================================================


class CacheServiceAdapter:
    """Adapter dla Cache Service."""

    def __init__(self, cache: Any):
        self.cache = cache

    async def get(self, key: str) -> Any:
        """Async get z cache."""
        try:
            return self.cache.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Async set do cache."""
        try:
            if hasattr(self.cache, "set"):
                self.cache.set(key, value, ttl)
        except Exception:
            pass

    async def invalidate(self, pattern: str) -> None:
        """
        Invalidate prostym dopasowaniem fragmentu klucza.
        Jeżeli backend posiada atrybuty pamięci, wyczyść je bezpiecznie.
        """
        frag = pattern.replace("*", "")
        try:
            # API cache'a (jeśli istnieje)
            if hasattr(self.cache, "invalidate"):
                self.cache.invalidate(pattern)  # type: ignore[attr-defined]
                return
        except Exception:
            pass

        # Wewnętrzne słowniki (np. smart_cache.memory_cache / memory_ttl)
        try:
            mem = getattr(self.cache, "memory_cache", None)
            ttl = getattr(self.cache, "memory_ttl", None)
            if isinstance(mem, dict):
                to_del = [k for k in list(mem.keys()) if frag in str(k)]
                for k in to_del:
                    mem.pop(k, None)
                    if isinstance(ttl, dict):
                        ttl.pop(k, None)
        except Exception:
            pass


# =====================================================================
# DataRepositoryAdapter
# =====================================================================


class DataRepositoryAdapter:
    """Adapter dla Data Repository (pamięciowy, z fallbackiem poza Streamlit)."""

    def __init__(self):
        self._use_st = st is not None
        if self._use_st:
            try:
                if "models_storage" not in st.session_state:
                    st.session_state.models_storage = {}
            except Exception:
                # jeżeli session_state nie działa – fallback do własnej pamięci
                self._use_st = False
        if not self._use_st:
            self._store: Dict[str, Dict[str, Any]] = {}

    def _get_store(self) -> Dict[str, Dict[str, Any]]:
        if self._use_st:
            return st.session_state.models_storage  # type: ignore[attr-defined]
        return self._store

    async def save_model(self, model_data: Any, metadata: Dict[str, Any]) -> str:
        """Zapisz model do storage (pamięciowo)."""
        # Stabilny, krótki id: hash(user_id + timestamp + skrót metadanych)
        user = str(metadata.get("user_id", "default"))
        ts = datetime.now(timezone.utc).isoformat()
        meta_digest = hashlib.sha1(repr(sorted(metadata.items())).encode("utf-8")).hexdigest()[:8]
        raw = f"{user}|{ts}|{meta_digest}".encode("utf-8")
        model_id = hashlib.md5(raw).hexdigest()[:12]

        store = self._get_store()
        store[model_id] = {
            "model_data": model_data,
            "metadata": metadata,
            "created_at": ts,
        }
        return model_id

    async def load_model(self, model_id: str) -> Any:
        """Wczytaj model ze storage."""
        store = self._get_store()
        rec = store.get(model_id)
        return None if rec is None else rec.get("model_data")

    async def list_user_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Zwróć listę modeli użytkownika (posortowaną malejąco po dacie)."""
        store = self._get_store()
        out: List[Dict[str, Any]] = []
        for mid, data in store.items():
            if data.get("metadata", {}).get("user_id") == user_id:
                md = data.get("model_data")
                # Obsłuż różne kształty (TrainingResults / dict / inne)
                if hasattr(md, "best_model"):
                    best = getattr(md, "best_model", "Unknown")
                elif isinstance(md, dict):
                    best = md.get("best_model", "Unknown")
                else:
                    best = "Unknown"
                out.append(
                    {
                        "model_id": mid,
                        "created_at": data.get("created_at"),
                        "target_column": data.get("metadata", {}).get("target_column"),
                        "best_model": best,
                    }
                )
        out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        return out