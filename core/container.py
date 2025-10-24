from typing import Optional
import streamlit as st

# Import implementations
from backend.ml_integration import MLModelTrainer
from backend.ai_integration import AIDescriptionGenerator
from backend.cache_manager import SmartCache
from core.services import MLPlatformService
from core.adapters import (
    MLTrainerAdapter, 
    AIInsightsAdapter,
    CacheServiceAdapter,
    DataRepositoryAdapter
)

class DependencyContainer:
    """Dependency Injection Container - Singleton pattern"""
    
    _instance: Optional['DependencyContainer'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_dependencies()
            DependencyContainer._initialized = True
    
    def _setup_dependencies(self):
        """Inicjalizuj wszystkie dependencje"""
        
        # Core services - adaptery łączące stary kod z nową architekturą
        self._cache_service = CacheServiceAdapter(SmartCache())
        self._ml_trainer = MLTrainerAdapter(MLModelTrainer())
        self._ai_insights = AIInsightsAdapter(AIDescriptionGenerator())
        self._data_repo = DataRepositoryAdapter()
        
        # Main orchestrator service
        self._platform_service = MLPlatformService(
            ml_trainer=self._ml_trainer,
            data_repo=self._data_repo,
            ai_insights=self._ai_insights,
            cache_service=self._cache_service
        )
    
    @property
    def platform_service(self) -> MLPlatformService:
        """Główny serwis platformy"""
        return self._platform_service
    
    @property
    def ml_trainer(self):
        """ML Trainer (backward compatibility)"""
        return self._ml_trainer.trainer
    
    @property
    def ai_insights(self):
        """AI Insights (backward compatibility)"""  
        return self._ai_insights.generator
    
    @property
    def cache_service(self):
        """Cache Service"""
        return self._cache_service

# Global container instance
def get_container() -> DependencyContainer:
    """Pobierz globalny kontener dependencji"""
    return DependencyContainer()