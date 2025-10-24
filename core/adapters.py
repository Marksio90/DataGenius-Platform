import asyncio
from typing import Dict, Any, List
import pandas as pd
import streamlit as st
import pickle
import hashlib
from datetime import datetime

from core.interfaces import (
    MLTrainerInterface, AIInsightsInterface, 
    CacheServiceInterface, DataRepositoryInterface,
    TrainingData, TrainingResults
)

class MLTrainerAdapter:
    """Adapter łączący stary MLModelTrainer z nowym interface"""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    async def train_models(self, data: TrainingData) -> TrainingResults:
        """Adaptuje nowy interface do starej implementacji"""
        
        # Wywołaj oryginalną metodę (może być sync lub async)
        if hasattr(self.trainer, 'train_model_async'):
            results_dict = await self.trainer.train_model_async(
                data.df, 
                data.target_column
            )
        else:
            # Fallback to sync method
            results_dict = self.trainer.train_model(
                data.df,
                data.target_column
            )
        
        # Konwertuj do nowego formatu
        return TrainingResults(
            best_model=results_dict.get('best_model', ''),
            metrics=self._extract_metrics(results_dict),
            model_scores=results_dict.get('model_scores', {}),
            feature_importance=results_dict.get('feature_importance')
        )
    
    def _extract_metrics(self, results_dict: Dict) -> Dict[str, float]:
        """Wyciągnij metryki z oryginalnego formatu"""
        metrics = {}
        metric_keys = ['r2', 'mae', 'mse', 'rmse', 'mape', 'accuracy', 'precision', 'recall', 'f1']
        
        for key in metric_keys:
            if key in results_dict:
                metrics[key] = float(results_dict[key])
        
        return metrics
    
    def suggest_target_column(self, df: pd.DataFrame) -> str:
        """Delegate to original implementation"""
        return self.trainer.suggest_target_column(df)
    
    def detect_problem_type(self, df: pd.DataFrame, target: str) -> str:
        """Delegate to original implementation"""
        return self.trainer.detect_problem_type(df, target)

class AIInsightsAdapter:
    """Adapter dla AI Insights Generator"""
    
    def __init__(self, generator):
        self.generator = generator
    
    async def generate_insights(self, results: TrainingResults) -> Dict[str, Any]:
        """Generuj insights na podstawie wyników ML"""
        
        # Konwertuj z powrotem do starego formatu dla compatibility
        results_dict = {
            'best_model': results.best_model,
            'model_scores': results.model_scores,
            'feature_importance': results.feature_importance,
            **results.metrics
        }
        
        # Wywołaj oryginalną metodę
        if hasattr(self.generator, 'generate_recommendations'):
            # Async version
            loop = asyncio.get_event_loop()
            recommendations = await loop.run_in_executor(
                None,
                self.generator.generate_recommendations,
                results_dict,
                "target",  # placeholder
                "Classification" if 'accuracy' in results.metrics else "Regression"
            )
        else:
            # Fallback
            recommendations = {
                'key_insights': 'Advanced ML analysis completed successfully',
                'action_items': ['Monitor model performance', 'Collect more data if needed'],
                'limitations': 'Model based on current dataset',
                'next_steps': ['Deploy to production', 'Set up monitoring']
            }
        
        return recommendations
    
    async def generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generuj opisy kolumn"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generator.generate_column_descriptions,
            df
        )
    
    async def generate_training_recommendations(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Generuj rekomendacje treningu"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generator.generate_training_recommendations,
            df, target
        )

class CacheServiceAdapter:
    """Adapter dla Cache Service"""
    
    def __init__(self, cache):
        self.cache = cache
    
    async def get(self, key: str) -> Any:
        """Async get z cache"""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Async set do cache"""
        self.cache.set(key, value, ttl)
    
    async def invalidate(self, pattern: str) -> None:
        """Invalidate cache pattern"""
        # Simple implementation - w produkcji użyj Redis SCAN
        if hasattr(self.cache, 'memory_cache'):
            keys_to_delete = [k for k in self.cache.memory_cache.keys() if pattern.replace('*', '') in k]
            for key in keys_to_delete:
                if key in self.cache.memory_cache:
                    del self.cache.memory_cache[key]
                if key in self.cache.memory_ttl:
                    del self.cache.memory_ttl[key]

class DataRepositoryAdapter:
    """Adapter dla Data Repository (memory-based)"""
    
    def __init__(self):
        # W pamięci dla demo - w produkcji użyj prawdziwej bazy
        if 'models_storage' not in st.session_state:
            st.session_state.models_storage = {}
    
    async def save_model(self, model_data: Any, metadata: Dict[str, Any]) -> str:
        """Zapisz model do storage"""
        
        # Generate unique ID
        model_id = hashlib.md5(
            f"{metadata.get('user_id', 'default')}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Store model data
        st.session_state.models_storage[model_id] = {
            'model_data': model_data,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        return model_id
    
    async def load_model(self, model_id: str) -> Any:
        """Wczytaj model z storage"""
        if model_id in st.session_state.models_storage:
            return st.session_state.models_storage[model_id]['model_data']
        return None
    
    async def list_user_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Lista modeli użytkownika"""
        user_models = []
        
        for model_id, data in st.session_state.models_storage.items():
            if data['metadata'].get('user_id') == user_id:
                user_models.append({
                    'model_id': model_id,
                    'created_at': data['created_at'],
                    'target_column': data['metadata'].get('target_column'),
                    'best_model': data['model_data'].best_model if hasattr(data['model_data'], 'best_model') else 'Unknown'
                })
        
        return sorted(user_models, key=lambda x: x['created_at'], reverse=True)