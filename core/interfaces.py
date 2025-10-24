from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List
import pandas as pd
import asyncio

# Domain Models
class TrainingData:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.shape = df.shape
        self.columns = list(df.columns)

class TrainingResults:
    def __init__(self, best_model: str, metrics: Dict[str, float], 
                 model_scores: Dict[str, Any], feature_importance: pd.DataFrame = None):
        self.best_model = best_model
        self.metrics = metrics
        self.model_scores = model_scores
        self.feature_importance = feature_importance

class AnalysisResults:
    def __init__(self, ml_results: TrainingResults, insights: Dict[str, Any], model_id: str = None):
        self.ml_results = ml_results
        self.insights = insights
        self.model_id = model_id

# Service Interfaces
class MLTrainerInterface(Protocol):
    """Interface dla ML Trainer"""
    async def train_models(self, data: TrainingData) -> TrainingResults: ...
    def suggest_target_column(self, df: pd.DataFrame) -> str: ...
    def detect_problem_type(self, df: pd.DataFrame, target: str) -> str: ...

class DataRepositoryInterface(Protocol):
    """Interface dla Data Repository"""
    async def save_model(self, model_data: Any, metadata: Dict[str, Any]) -> str: ...
    async def load_model(self, model_id: str) -> Any: ...
    async def list_user_models(self, user_id: str) -> List[Dict[str, Any]]: ...

class AIInsightsInterface(Protocol):
    """Interface dla AI Insights Generator"""
    async def generate_insights(self, results: TrainingResults) -> Dict[str, Any]: ...
    async def generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]: ...
    async def generate_training_recommendations(self, df: pd.DataFrame, target: str) -> Dict[str, Any]: ...

class CacheServiceInterface(Protocol):
    """Interface dla Cache Service"""
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None: ...
    async def invalidate(self, pattern: str) -> None: ...

class EDAServiceInterface(Protocol):
    """Interface dla EDA Service"""
    async def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]: ...
    async def create_visualizations(self, df: pd.DataFrame, target: str = None) -> Dict[str, Any]: ...