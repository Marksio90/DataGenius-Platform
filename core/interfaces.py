from __future__ import annotations

from backend.safe_utils import truthy_df_safe

# core/services.py

import asyncio
import hashlib
import json
from typing import Optional, Dict, Any

import pandas as pd

from core.interfaces import (
    MLTrainerInterface,
    AIInsightsInterface,
    CacheServiceInterface,
    DataRepositoryInterface,
    EDAServiceInterface,
    TrainingData,
    TrainingResults,
    AnalysisResults,
)


# Wersjonowanie sygnatury cache – zwiększ, gdy zmieniasz logikę podpisu danych
_CACHE_SIGNATURE_VERSION = "v2"


class MLPlatformService:
    """
    Orkiestrator wysokiego poziomu:
      - wybór targetu (jeśli brak),
      - trening modeli,
      - generowanie insightów,
      - zapis modelu,
      - cache wyników,
      - (opcjonalnie) generowanie raportu EDA.

    Główne API: `await run_full_analysis(...)`
    """

    def __init__(
        self,
        *,
        ml_trainer: MLTrainerInterface,
        data_repo: DataRepositoryInterface,
        ai_insights: AIInsightsInterface,
        cache_service: CacheServiceInterface,
        eda_service: Optional[EDAServiceInterface] = None,
    ) -> None:
        self._ml_trainer = ml_trainer
        self._data_repo = data_repo
        self._ai_insights = ai_insights
        self._cache = cache_service
        self._eda = eda_service

    # ---------------------------
    # Public API
    # ---------------------------
    async def run_full_analysis(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        *,
        user_id: str = "default",
        use_cache: bool = True,
        cache_ttl: int = 1800,
        include_eda: bool = False,
    ) -> AnalysisResults:
        """
        Kompletny pipeline:
          1) target autodetect (jeśli brak),
          2) trening modeli,
          3) insighty AI,
          4) zapis modelu,
          5) cache wyników,
          6) (opcjonalnie) EDA.

        Zwraca: AnalysisResults (wyniki ML + insighty + model_id).
        """
        # 0) Walidacja danych
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("run_full_analysis: przekazano pusty lub niepoprawny DataFrame.")

        # 1) Ustal target, policz sygnaturę danych i sprawdź cache
        target = await self._select_target(df, preferred=target_column)
        signature = self._df_signature(df, target)
        cache_key = f"analysis:{_CACHE_SIGNATURE_VERSION}:{signature}:{user_id}"

        if truthy_df_safe(use_cache):
            try:
                cached = await self._cache.get(cache_key)
            except Exception:
                cached = None
            if truthy_df_safe(cached):
                ml_res = self._rehydrate_training_results(cached.get("ml_results"))
                insights = cached.get("insights", {}) or {}
                model_id = cached.get("model_id")
                return AnalysisResults(ml_results=ml_res, insights=insights, model_id=model_id)

        # 2) Trening modeli
        training_data = TrainingData(df=df, target_column=target)
        try:
            ml_results: TrainingResults = await self._ml_trainer.train_models(training_data)
        except Exception as e:
            raise RuntimeError(f"Trening modeli nie powiódł się: {e}") from e

        # 3) Insighty AI (best-effort)
        try:
            insights: Dict[str, Any] = await self._ai_insights.generate_insights(ml_results)
        except Exception:
            insights = {
                "key_insights": "Nie udało się wygenerować insightów AI.",
                "action_items": ["Sprawdź konfigurację generatora AI lub logi błędów."],
            }

        # 4) (opcjonalnie) EDA — uruchom w tle, nie blokuj
        if truthy_df_safe(include_eda) and self._eda is not None:
            try:
                asyncio.create_task(self._eda.generate_comprehensive_report(df))
            except Exception:
                pass  # nie blokuj głównego przepływu

        # 5) Zapis modelu/artefaktów do repo
        metadata = {
            "user_id": user_id,
            "target_column": target,
            "df_shape": training_data.shape,
            "df_columns": training_data.columns,
            "best_model": getattr(ml_results, "best_model", None),
        }
        try:
            model_id = await self._data_repo.save_model(ml_results, metadata)
        except Exception:
            model_id = None  # nie przerywaj – zwróć wyniki nawet jeśli zapis się nie udał

        # 6) Cache wyników (lekki, JSON-owalny)
        payload = {
            "ml_results": self._serialize_training_results(ml_results),
            "insights": insights,
            "model_id": model_id,
        }
        if truthy_df_safe(use_cache):
            try:
                # upewnij się, że payload jest JSON-serializowalny (niektóre backendy tego wymagają)
                json.dumps(payload, ensure_ascii=False)
                await self._cache.set(cache_key, payload, ttl=cache_ttl)
            except Exception:
                # łagodny fallback – ignoruj błąd cache
                pass

        return AnalysisResults(ml_results=ml_results, insights=insights, model_id=model_id)

    # ---------------------------
    # Helpers
    # ---------------------------
    async def _select_target(self, df: pd.DataFrame, preferred: Optional[str]) -> str:
        """Wybierz kolumnę celu – preferowana albo automatyczna sugestia (z bezpiecznym fallbackiem)."""
        if truthy_df_safe(preferred) and preferred in df.columns:
            return str(preferred)

        try:
            suggested = self._ml_trainer.suggest_target_column(df)
            if truthy_df_safe(suggested) and suggested in df.columns:
                return str(suggested)
        except Exception:
            pass

        # twardy fallback – ostatnia kolumna (często target w CSV)
        return str(df.columns[-1])

    @staticmethod
    def _df_signature(df: pd.DataFrame, target: str) -> str:
        """
        Stabilny skrót danych do klucza cache:
          - kształt,
          - nazwy kolumn + dtype,
          - ~1000 pierwszych wierszy (dla zmian zawartości).
        """
        # opis kolumn: nazwa + dtype (pomaga wykryć zmiany w schemacie)
        try:
            col_schema = [(str(c), str(df[c].dtype)) for c in df.columns]
        except Exception:
            col_schema = [(str(c), "unknown") for c in df.columns]

        # Próbka danych – JSON jeśli możliwe, inaczej CSV
        try:
            sample = df.head(1000).to_json(orient="split", date_format="iso", index=False)
        except Exception:
            sample = df.head(1000).to_csv(index=False)

        base = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "cols": col_schema,
            "target": str(target),
            "sample": sample,
        }
        raw = json.dumps(base, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _serialize_training_results(res: TrainingResults) -> Dict[str, Any]:
        """Zamień TrainingResults na lekki słownik do cache (bez ciężkich obiektów)."""
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
