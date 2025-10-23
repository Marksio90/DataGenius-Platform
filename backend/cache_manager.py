"""
Zarządzanie cache aplikacji.

Funkcjonalności:
- Cache DataFrame (hash-based)
- Cache opisów kolumn z LLM
- Cache wyników EDA
- Czyszczenie cache
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from backend.utils import hash_dataframe
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheManager:
    """
    Manager cache dla aplikacji.
    
    Wykorzystuje st.cache_data i lokalne pliki dla trwałości.
    """
    
    def __init__(self):
        """Inicjalizacja managera cache."""
        self.cache_dir = settings.root_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.enabled = settings.enable_cache
    
    def get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Zwraca hash DataFrame dla celów cache.
        
        Args:
            df: DataFrame do zahashowania
        
        Returns:
            str: Hash SHA256
        """
        return hash_dataframe(df)
    
    def cache_column_descriptions(
        self,
        df_hash: str,
        descriptions: dict
    ) -> None:
        """
        Cachuje opisy kolumn z LLM.
        
        Args:
            df_hash: Hash DataFrame
            descriptions: Słownik opisów kolumn
        """
        if not self.enabled:
            return
        
        try:
            cache_file = self.cache_dir / f"col_desc_{df_hash[:16]}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(descriptions, f)
            logger.info(f"Zapisano opisy kolumn do cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Nie udało się zapisać cache opisów: {e}")
    
    def get_column_descriptions(self, df_hash: str) -> Optional[dict]:
        """
        Pobiera opisy kolumn z cache.
        
        Args:
            df_hash: Hash DataFrame
        
        Returns:
            Optional[dict]: Opisy kolumn lub None
        """
        if not self.enabled:
            return None
        
        try:
            cache_file = self.cache_dir / f"col_desc_{df_hash[:16]}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    descriptions = pickle.load(f)
                logger.info(f"Wczytano opisy kolumn z cache: {cache_file.name}")
                return descriptions
        except Exception as e:
            logger.warning(f"Nie udało się wczytać cache opisów: {e}")
        
        return None
    
    def cache_eda_results(self, df_hash: str, eda_data: dict) -> None:
        """
        Cachuje wyniki analizy EDA.
        
        Args:
            df_hash: Hash DataFrame
            eda_data: Dane EDA do zapisania
        """
        if not self.enabled:
            return
        
        try:
            cache_file = self.cache_dir / f"eda_{df_hash[:16]}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(eda_data, f)
            logger.info(f"Zapisano EDA do cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Nie udało się zapisać cache EDA: {e}")
    
    def get_eda_results(self, df_hash: str) -> Optional[dict]:
        """
        Pobiera wyniki EDA z cache.
        
        Args:
            df_hash: Hash DataFrame
        
        Returns:
            Optional[dict]: Dane EDA lub None
        """
        if not self.enabled:
            return None
        
        try:
            cache_file = self.cache_dir / f"eda_{df_hash[:16]}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    eda_data = pickle.load(f)
                logger.info(f"Wczytano EDA z cache: {cache_file.name}")
                return eda_data
        except Exception as e:
            logger.warning(f"Nie udało się wczytać cache EDA: {e}")
        
        return None
    
    def clear_cache(self) -> int:
        """
        Czyści wszystkie pliki cache.
        
        Returns:
            int: Liczba usuniętych plików
        """
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1
            logger.info(f"Wyczyszczono cache: {count} plików")
        except Exception as e:
            logger.warning(f"Błąd podczas czyszczenia cache: {e}")
        
        return count
    
    def get_cache_size(self) -> float:
        """
        Zwraca rozmiar cache w MB.
        
        Returns:
            float: Rozmiar cache w MB
        """
        try:
            total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            return total_bytes / (1024 ** 2)
        except Exception:
            return 0.0
    
    def get_cache_info(self) -> dict:
        """
        Zwraca informacje o cache.
        
        Returns:
            dict: Informacje o cache
        """
        try:
            files = list(self.cache_dir.glob("*.pkl"))
            return {
                "enabled": self.enabled,
                "n_files": len(files),
                "size_mb": self.get_cache_size(),
                "files": [f.name for f in files],
            }
        except Exception as e:
            logger.warning(f"Błąd podczas pobierania info o cache: {e}")
            return {
                "enabled": self.enabled,
                "n_files": 0,
                "size_mb": 0.0,
                "files": [],
            }


# Singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Zwraca singleton instancji CacheManager.
    
    Returns:
        CacheManager: Instancja managera cache
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager