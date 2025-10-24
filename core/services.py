from __future__ import annotations
import asyncio
import hashlib
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from datetime import datetime

from core.interfaces import (
    MLTrainerInterface, DataRepositoryInterface, AIInsightsInterface,
    CacheServiceInterface, TrainingData, TrainingResults, AnalysisResults
)

class MLPlatformService:
    """Główny serwis orchestrujący całą platformę ML"""
    
    def __init__(
        self,
        ml_trainer: MLTrainerInterface,
        data_repo: DataRepositoryInterface,
        ai_insights: AIInsightsInterface,
        cache_service: CacheServiceInterface
    ):
        self.ml_trainer = ml_trainer
        self.data_repo = data_repo
        self.ai_insights = ai_insights
        self.cache = cache_service
    
    async def run_complete_analysis(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        user_id: str = "default"
    ) -> AnalysisResults:
        """End-to-end analiza ML z inteligentnym cachingiem"""
        
        # 1. Generate cache key
        cache_key = self._generate_cache_key(df, target_column)
        
        # 2. Check cache first
        cached_result = await self.cache.get(f"complete_analysis_{cache_key}")
        if cached_result:
            st.info("🚀 Używam gotowej analizy z cache")
            return cached_result
        
        # 3. Create training data
        training_data = TrainingData(df, target_column)
        
        # 4. Start parallel tasks
        with st.spinner("🔄 Uruchamiam równoległe analizy..."):
            
            # Start ML training
            ml_task = asyncio.create_task(
                self.ml_trainer.train_models(training_data),
                name="ml_training"
            )
            
            # Start AI insights generation (can run in parallel with early data)
            insights_task = asyncio.create_task(
                self.ai_insights.generate_column_descriptions(df),
                name="ai_insights"
            )
            
            # Wait for ML training to complete first
            st.info("🤖 Trenowanie modeli...")
            ml_results = await ml_task
            
            # Start model saving (parallel with AI insights)
            save_task = asyncio.create_task(
                self.data_repo.save_model(ml_results, {
                    'user_id': user_id,
                    'target_column': target_column,
                    'timestamp': datetime.now().isoformat()
                }),
                name="model_saving"
            )
            
            # Generate advanced AI insights based on ML results
            advanced_insights_task = asyncio.create_task(
                self.ai_insights.generate_insights(ml_results),
                name="advanced_insights"
            )
            
            # Wait for all remaining tasks
            st.info("💡 Generowanie AI insights...")
            column_descriptions, model_id, advanced_insights = await asyncio.gather(
                insights_task,
                save_task, 
                advanced_insights_task
            )
        
        # 5. Combine all results
        final_insights = {
            'column_descriptions': column_descriptions,
            'ml_insights': advanced_insights,
            'recommendations': advanced_insights.get('recommendations', [])
        }
        
        final_results = AnalysisResults(
            ml_results=ml_results,
            insights=final_insights,
            model_id=model_id
        )
        
        # 6. Cache for future use
        await self.cache.set(f"complete_analysis_{cache_key}", final_results, ttl=7200)  # 2h
        
        st.success("✅ Kompletna analiza zakończona!")
        
        return final_results
    
    async def quick_model_training(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        strategy: str = "fast"
    ) -> TrainingResults:
        """Szybki trening modelu bez dodatkowych analiz"""
        
        cache_key = self._generate_cache_key(df, target_column, strategy)
        
        # Check cache
        cached_result = await self.cache.get(f"quick_training_{cache_key}")
        if cached_result:
            st.info("🚀 Model już wytrenowany - używam cache")
            return cached_result
        
        # Train model
        training_data = TrainingData(df, target_column)
        results = await self.ml_trainer.train_models(training_data)
        
        # Cache results
        await self.cache.set(f"quick_training_{cache_key}", results, ttl=3600)
        
        return results
    
    def _generate_cache_key(self, df: pd.DataFrame, target_column: str, *args) -> str:
        """Generuj unikalny klucz cache dla analizy"""
        # Hash DataFrame content
        df_hash = hashlib.md5(df.to_string().encode()).hexdigest()[:12]
        
        # Combine with other parameters
        params = f"{df_hash}_{target_column}_{args}"
        return hashlib.sha256(params.encode()).hexdigest()[:16]
    
    async def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Pobierz historię analiz użytkownika"""
        return await self.data_repo.list_user_models(user_id)
    
    async def invalidate_user_cache(self, user_id: str):
        """Wyczyść cache dla użytkownika"""
        await self.cache.invalidate(f"*_{user_id}_*")