"""
Feature Engineering module - automatyczne tworzenie i selekcja features.
"""

__version__ = "3.0.0"

from .auto_features import AutoFeatureGenerator
from .feature_selector import FeatureSelector
from .feature_store import FeatureStore

__all__ = [
    'AutoFeatureGenerator',
    'FeatureSelector',
    'FeatureStore',
]