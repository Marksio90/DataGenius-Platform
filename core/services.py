from __future__ import annotations

from backend.safe_utils import truthy_df_safe

import asyncio
import hashlib
import json
from typing import Dict, Any, Optional, List
from contextlib import nullcontext
from datetime import datetime

import pandas as pd

# Streamlit jest opcjonalny – jeśli nie ma, komunikaty UI są pomijane.
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

from core.interfaces import (
    MLTrainerInterface,
    DataRepositoryInterface,
    AIInsightsInterface,
    CacheServiceInterface,
    TrainingData,
    TrainingResults,
    AnalysisResults,
)


_CACHE_VER = "v2"  # podbij gdy zmieniasz format cache/kluczy


class MLPlatformService:
    """Główny serwis orkiestrujący całą platformę ML."""

    def __init__(
        self,
        ml_trainer: MLTrainerInterface,
        data_repo: DataRepositoryInterface,
        ai_insights: AIInsightsInterface,
        cache_service: CacheServiceInterface,
    ):
        self.ml_trainer = ml_trainer
        self.data_repo = data_repo
        self.ai_insights = ai_insights
        self.cache = cache_service

    # ------------------------- Public API -------------------------

    async def run_complete_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        user_id: str = "default",
    ) -> AnalysisResults:
        """
        End-to-end analiza ML z inteligentnym cachingiem, zapisem modelu
        i generowaniem AI-insightów.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("run_complete_analysis: przekazano pusty lub niepoprawny DataFrame.")

        target_column = self._ensure_target(df, target_column)

        # 1) Klucz cache per-user + fingerprint danych
        key = f"{_CACHE_VER}:complete_analysis:{user_id}:{self._generate_cache_key(df, target_column)}"

        # 2) Cache hit?
        cached_result = await self.cache.get(key)
        if truthy_df_safe(cached_result):
            self._st_info("🚀 Używam gotowej analizy z cache")
            if isinstance(cached_result, AnalysisResults):
                return cached_result
            # rehydratacja z lekkiego payloadu
            return self._rehydrate_analysis_results(cached_result)

        # 3) Dane treningowe
        training_data = TrainingData(df, target_column)

        # 4) Równoległe zadania
        with (st.spinner("🔄 Uruchamiam równoległe analizy...") if st else nullcontext()):
            # Trenowanie modeli
            self._st_info("🤖 Trenowanie modeli...")
            ml_task = asyncio.create_task(self.ml_trainer.train_models(training_data), name="ml_training")

            # Wstępne opisy kolumn (niezależne od treningu)
            insights_task = asyncio.create_task(
                self._safe_generate_column_descriptions(df), name="ai_insights"
            )

            # Czekamy na ML, bo dalsze insighty ich potrzebują
            ml_results: TrainingResults = await ml_task

            # Zapis modelu + zaawansowane insighty równolegle
            save_task = asyncio.create_task(
                self.data_repo.save_model(
                    ml_results,
                    {
                        "user_id": user_id,
                        "target_column": target_column,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ),
                name="model_saving",
            )
            advanced_insights_task = asyncio.create_task(
                self._safe_generate_insights(ml_results), name="advanced_insights"
            )

            self._st_info("💡 Generowanie AI insights...")
            column_descriptions, model_id, advanced_insights = await asyncio.gather(
                insights_task, save_task, advanced_insights_task
            )

        # 5) Złożenie wyników
        final_insights: Dict[str, Any] = {
            "column_descriptions": column_descriptions,
            "ml_insights": advanced_insights,
            "recommendations": (advanced_insights or {}).get("recommendations", []),
        }

        final_results = AnalysisResults(ml_results=ml_results, insights=final_insights, model_id=model_id)

        # 6) Cache (2h) – zapisuj w formie lekkiego słownika (JSON-owalnego)
        try:
            payload = self._serialize_analysis_results(final_results)
            # walidacja JSON
            json.dumps(payload, ensure_ascii=False)
            await self.cache.set(key, payload, ttl=7200)
        except Exception:
            # cichy fallback – brak cache nie blokuje zwrotu
            pass

        self._st_success("✅ Kompletna analiza zakończona!")
        return final_results

    async def quick_model_training(
        self,
        df: pd.DataFrame,
        target_column: str,
        strategy: str = "fast",
        user_id: str = "default",
    ) -> TrainingResults:
        """
        Szybki trening modelu bez dodatkowych analiz (również cache per-user).
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("quick_model_training: przekazano pusty lub niepoprawny DataFrame.")

        target_column = self._ensure_target(df, target_column)

        key = f"{_CACHE_VER}:quick_training:{user_id}:{self._generate_cache_key(df, target_column, extra=strategy)}"

        cached = await self.cache.get(key)
        if truthy_df_safe(cached):
            self._st_info("🚀 Model już wytrenowany — używam cache")
            if isinstance(cached, TrainingResults):
                return cached
            return self._rehydrate_training_results(cached)

        training_data = TrainingData(df, target_column)
        results = await self.ml_trainer.train_models(training_data)

        try:
            await self.cache.set(key, self._serialize_training_results(results), ttl=3600)
        except Exception:
            pass
        return results

    async def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Pobierz historię analiz użytkownika (z repozytorium modeli)."""
        return await self.data_repo.list_user_models(user_id)

    async def invalidate_user_cache(self, user_id: str) -> None:
        """
        Wyczyść cache dla użytkownika.
        Wzorce są zgodne z kluczami używanymi wyżej: prefix:user_id:fingerprint.
        """
        await self.cache.invalidate(f"{_CACHE_VER}:complete_analysis:{user_id}:*")
        await self.cache.invalidate(f"{_CACHE_VER}:quick_training:{user_id}:*")

    # ------------------------- Helpers -------------------------

    def _ensure_target(self, df: pd.DataFrame, target_column: str) -> str:
        """Ustal poprawny target – preferuj podany, w razie braku użyj sugestii/fallback."""
        if truthy_df_safe(target_column) and target_column in df.columns:
            return str(target_column)
        try:
            suggested = self.ml_trainer.suggest_target_column(df)
            if truthy_df_safe(suggested) and suggested in df.columns:
                return str(suggested)
        except Exception:
            pass
        # Fallback: ostatnia kolumna
        return str(df.columns[-1])

    def _generate_cache_key(self, df: pd.DataFrame, target_column: str, *, extra: str = "") -> str:
        """
        Stabilny fingerprint danych:
          - kształt, kolumny, dtypes,
          - próbka (do 1000 wierszy) w JSON (orient='split'),
        następnie SHA-256 i skrót do 16 znaków.
        """
        try:
            sample_json = df.head(1000).to_json(orient="split", date_format="iso", index=False)
        except Exception:
            # awaryjnie CSV — dalej deterministycznie
            sample_json = df.head(1000).to_csv(index=False)

        payload = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": [str(c) for c in df.columns],
            "dtypes": [str(t) for t in df.dtypes.tolist()],
            "target": str(target_column),
            "extra": str(extra),
            "sample": sample_json,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    async def _safe_generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Bezpieczne generowanie opisów kolumn – zawsze zwróci słownik."""
        try:
            out = await self.ai_insights.generate_column_descriptions(df)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
        # fallback
        return {str(c): f"Column '{c}' (dtype={df[c].dtype})" for c in df.columns}

    async def _safe_generate_insights(self, ml_results: TrainingResults) -> Dict[str, Any]:
        """Bezpieczne generowanie insightów – zawsze zwróci dict."""
        try:
            out = await self.ai_insights.generate_insights(ml_results)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
        return {
            "key_insights": "Nie udało się wygenerować insightów AI.",
            "action_items": ["Sprawdź konfigurację generatora AI lub logi błędów."],
        }

    @staticmethod
    def _serialize_training_results(res: TrainingResults) -> Dict[str, Any]:
        """Lekka, JSON-owalna reprezentacja TrainingResults."""
        out: Dict[str, Any] = {
            "best_model": getattr(res, "best_model", None),
            "metrics": getattr(res, "metrics", {}) or {},
            "model_scores": getattr(res, "model_scores", {}) or {},
        }
        fi = getattr(res, "feature_importance", None)
        if fi is not None:
            try:
                out["feature_importance"] = fi.to_dict(orient="list")
            except Exception:
                out["feature_importance"] = None
        else:
            out["feature_importance"] = None
        return out

    @staticmethod
    def _rehydrate_training_results(payload: Optional[Dict[str, Any]]) -> TrainingResults:
        """Odtwórz TrainingResults z lekkiego słownika (np. z cache)."""
        payload = payload or {}
        fi = payload.get("feature_importance")
        fi_df = None
        if isinstance(fi, dict):
            try:
                fi_df = pd.DataFrame(fi)
            except Exception:
                fi_df = None
        return TrainingResults(
            best_model=payload.get("best_model", "") or "",
            metrics=payload.get("metrics", {}) or {},
            model_scores=payload.get("model_scores", {}) or {},
            feature_importance=fi_df,
        )

    def _serialize_analysis_results(self, res: AnalysisResults) -> Dict[str, Any]:
        """Lekka, JSON-owalna reprezentacja AnalysisResults (zawiera zserializowane TrainingResults)."""
        return {
            "ml_results": self._serialize_training_results(res.ml_results),
            "insights": res.insights or {},
            "model_id": res.model_id,
        }

    def _rehydrate_analysis_results(self, payload: Dict[str, Any]) -> AnalysisResults:
        """Odtwórz AnalysisResults z lekkiego słownika (np. z cache)."""
        ml_payload = (payload or {}).get("ml_results", {})
        return AnalysisResults(
            ml_results=self._rehydrate_training_results(ml_payload),
            insights=(payload or {}).get("insights", {}) or {},
            model_id=(payload or {}).get("model_id"),
        )

    @staticmethod
    def _st_info(msg: str) -> None:
        if st is not None:
            try:
                st.info(msg)
            except Exception:
                pass

    @staticmethod
    def _st_success(msg: str) -> None:
        if st is not None:
            try:
                st.success(msg)
            except Exception:
                pass
