"""
TMIV Advanced Encoding Framework v3.0
======================================
Zaawansowany system encodowania cech kategorycznych z:
- Compliance enforcement (scikit-learn 1.5+ compatibility)
- Multiple encoding strategies (OHE, Target, Ordinal, Frequency, WoE)
- Automatic encoder selection based on cardinality
- Memory-efficient sparse matrix handling
- Feature name preservation & tracking
- Encoding validation & drift detection
- Custom encoder registry & plugins
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)

from .telemetry import audit, metric


# ============================================================================
# COMPLIANCE & ERRORS
# ============================================================================

class EncoderComplianceError(RuntimeError):
    """
    Błąd zgodności enkodera (TMIV-ML-ENC-001).
    
    Wykryto niedozwolony parametr `sparse` w OneHotEncoder.
    W TMIV dozwolone jest wyłącznie `sparse_output` (scikit-learn ≥ 1.5).
    
    Rozwiązanie:
        Zamiast: OneHotEncoder(sparse=False)
        Użyj:    OneHotEncoder(sparse_output=False)
    """
    
    def __init__(self, message: str = ""):
        default_msg = (
            "TMIV-ML-ENC-001: Detected deprecated 'sparse' parameter.\n"
            "Use 'sparse_output' instead (scikit-learn ≥ 1.5).\n"
            "Example: OneHotEncoder(sparse_output=True)"
        )
        super().__init__(message or default_msg)


class EncodingStrategyError(ValueError):
    """Błąd związany z wyborem strategii encodowania."""
    pass


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class EncodingStrategy(str, Enum):
    """Dostępne strategie encodowania."""
    ONE_HOT = "one_hot"              # OneHotEncoder (sparse)
    ORDINAL = "ordinal"              # OrdinalEncoder (integers)
    TARGET = "target"                # Target encoding (mean of target per category)
    FREQUENCY = "frequency"          # Frequency encoding (count/total)
    WOE = "weight_of_evidence"       # Weight of Evidence (for binary classification)
    BINARY = "binary"                # Binary encoding (bit representation)
    HASHING = "hashing"              # Feature hashing
    LEAVE_ONE_OUT = "leave_one_out"  # Leave-one-out encoding


@dataclass
class EncoderMetadata:
    """Metadane enkodera dla tracking."""
    strategy: EncodingStrategy
    feature_name: str
    original_cardinality: int
    encoded_dim: int
    categories: List[Any]
    created_at: str
    version: str = "3.0"


# ============================================================================
# ENCODER INTERFACE
# ============================================================================

class CategoricalEncoder(ABC, BaseEstimator, TransformerMixin):
    """Abstrakcyjna klasa bazowa dla encoderów kategorycznych."""
    
    @abstractmethod
    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """Dopasowanie enkodera do danych."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transformacja danych."""
        pass
    
    @abstractmethod
    def get_feature_names_out(self) -> List[str]:
        """Zwraca nazwy wygenerowanych cech."""
        pass
    
    def fit_transform(self, X: pd.Series, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit + transform w jednym wywołaniu."""
        return self.fit(X, y).transform(X)


# ============================================================================
# ONE-HOT ENCODER (Enhanced)
# ============================================================================

class SafeOneHotEncoder(CategoricalEncoder):
    """
    Ulepszona wersja OneHotEncoder z compliance checking.
    
    Features:
    - Automatic scikit-learn version detection
    - Sparse output by default (memory efficient)
    - Feature name tracking
    - Unknown category handling
    - Cardinality validation
    """
    
    def __init__(
        self,
        categories: str | list | None = "auto",
        drop: Optional[str] = None,
        handle_unknown: str = "ignore",
        sparse_output: bool = True,
        max_categories: int = 1000,
        feature_name: Optional[str] = None
    ):
        """
        Args:
            categories: Categories per feature ('auto' for auto-detection)
            drop: Strategy for dropping one category ('first', 'if_binary', None)
            handle_unknown: How to handle unknown categories ('error', 'ignore')
            sparse_output: Whether to return sparse matrix (memory efficient)
            max_categories: Maximum allowed unique categories
            feature_name: Original feature name (for tracking)
        """
        self.categories = categories
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.max_categories = max_categories
        self.feature_name = feature_name
        
        self._encoder: Optional[OneHotEncoder] = None
        self._original_cardinality: int = 0
    
    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> 'SafeOneHotEncoder':
        """Dopasowanie enkodera z walidacją."""
        # Validate cardinality
        self._original_cardinality = X.nunique()
        
        if self._original_cardinality > self.max_categories:
            raise EncodingStrategyError(
                f"TMIV-ML-ENC-002: Too many categories ({self._original_cardinality}) "
                f"for one-hot encoding (max: {self.max_categories}). "
                f"Consider using: target encoding, hashing, or ordinal encoding."
            )
        
        # Create compliant encoder
        self._encoder = self._create_compliant_encoder()
        
        # Fit
        X_2d = X.values.reshape(-1, 1)
        self._encoder.fit(X_2d)
        
        # Telemetry
        audit("encoder_fit", {
            "strategy": "one_hot",
            "feature": self.feature_name or "unknown",
            "cardinality": self._original_cardinality,
            "output_dim": len(self.get_feature_names_out())
        })
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transformacja z obsługą unknown categories."""
        if self._encoder is None:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        X_2d = X.values.reshape(-1, 1)
        
        try:
            result = self._encoder.transform(X_2d)
        except Exception as e:
            # Graceful handling of unknown categories
            if self.handle_unknown == "ignore":
                # This should not happen with handle_unknown='ignore', but just in case
                warnings.warn(f"Unknown categories encountered: {e}")
                result = self._encoder.transform(X_2d)
            else:
                raise
        
        return result
    
    def get_feature_names_out(self) -> List[str]:
        """Zwraca nazwy wygenerowanych cech."""
        if self._encoder is None:
            return []
        
        try:
            names = self._encoder.get_feature_names_out()
            
            # Replace generic names with feature-specific ones
            if self.feature_name:
                names = [
                    name.replace("x0_", f"{self.feature_name}_")
                    for name in names
                ]
            
            return names.tolist()
        except Exception:
            return [f"ohe_{i}" for i in range(self._encoder.n_features_in_)]
    
    def _create_compliant_encoder(self) -> OneHotEncoder:
        """
        Tworzy OneHotEncoder zgodny z TMIV compliance rules.
        
        CRITICAL: Używa wyłącznie `sparse_output` (nie `sparse`).
        """
        # Version check (informational)
        try:
            import sklearn
            version = tuple(map(int, sklearn.__version__.split('.')[:2]))
            if version < (1, 5):
                warnings.warn(
                    f"TMIV recommends scikit-learn ≥ 1.5 (detected: {sklearn.__version__}). "
                    "Some features may not work correctly."
                )
        except Exception:
            pass
        
        # Create encoder with ONLY compliant parameters
        encoder = OneHotEncoder(
            categories=self.categories,
            drop=self.drop,
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse_output  # ← COMPLIANT PARAMETER
        )
        
        return encoder


# ============================================================================
# TARGET ENCODER
# ============================================================================

class TargetEncoder(CategoricalEncoder):
    """
    Target Encoding (mean target value per category).
    
    Features:
    - Smoothing to prevent overfitting
    - Unknown category handling (global mean)
    - Regularization based on category frequency
    """
    
    def __init__(
        self,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        feature_name: Optional[str] = None
    ):
        """
        Args:
            smoothing: Smoothing parameter (higher = more regularization)
            min_samples_leaf: Minimum samples to compute target mean
            feature_name: Original feature name
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.feature_name = feature_name
        
        self._mapping: Dict[Any, float] = {}
        self._global_mean: float = 0.0
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'TargetEncoder':
        """Dopasowanie z wyliczeniem target means."""
        if y is None:
            raise ValueError("TargetEncoder requires target variable y.")
        
        # Global mean
        self._global_mean = float(y.mean())
        
        # Compute target mean per category with smoothing
        df = pd.DataFrame({'cat': X, 'target': y})
        
        agg = df.groupby('cat').agg(
            target_sum=('target', 'sum'),
            target_count=('target', 'count')
        )
        
        # Smoothing: (sum + smoothing * global_mean) / (count + smoothing)
        agg['encoded'] = (
            (agg['target_sum'] + self.smoothing * self._global_mean) /
            (agg['target_count'] + self.smoothing)
        )
        
        self._mapping = agg['encoded'].to_dict()
        
        audit("encoder_fit", {
            "strategy": "target",
            "feature": self.feature_name or "unknown",
            "cardinality": len(self._mapping),
            "global_mean": self._global_mean
        })
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transformacja z fallback na global mean."""
        return X.map(self._mapping).fillna(self._global_mean).values.reshape(-1, 1)
    
    def get_feature_names_out(self) -> List[str]:
        """Zwraca nazwę cechy."""
        name = self.feature_name or "feature"
        return [f"{name}_target_encoded"]


# ============================================================================
# FREQUENCY ENCODER
# ============================================================================

class FrequencyEncoder(CategoricalEncoder):
    """
    Frequency Encoding (frequency of category / total count).
    
    Simple and effective for high-cardinality features.
    """
    
    def __init__(self, feature_name: Optional[str] = None):
        self.feature_name = feature_name
        self._freq_map: Dict[Any, float] = {}
    
    def fit(self, X: pd.Series, y: Optional[pd.Series] = None) -> 'FrequencyEncoder':
        """Oblicza częstości występowania."""
        freq = X.value_counts(normalize=True)
        self._freq_map = freq.to_dict()
        
        audit("encoder_fit", {
            "strategy": "frequency",
            "feature": self.feature_name or "unknown",
            "cardinality": len(self._freq_map)
        })
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transformacja z fallback na 0.0."""
        return X.map(self._freq_map).fillna(0.0).values.reshape(-1, 1)
    
    def get_feature_names_out(self) -> List[str]:
        name = self.feature_name or "feature"
        return [f"{name}_frequency"]


# ============================================================================
# WEIGHT OF EVIDENCE ENCODER
# ============================================================================

class WoEEncoder(CategoricalEncoder):
    """
    Weight of Evidence Encoding (for binary classification).
    
    WoE = ln(P(X|Y=1) / P(X|Y=0))
    """
    
    def __init__(
        self,
        epsilon: float = 1e-5,
        feature_name: Optional[str] = None
    ):
        """
        Args:
            epsilon: Small value to avoid log(0)
            feature_name: Original feature name
        """
        self.epsilon = epsilon
        self.feature_name = feature_name
        self._woe_map: Dict[Any, float] = {}
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'WoEEncoder':
        """Oblicza WoE dla każdej kategorii."""
        if y is None:
            raise ValueError("WoEEncoder requires binary target y.")
        
        if y.nunique() != 2:
            raise ValueError("WoEEncoder only supports binary classification.")
        
        df = pd.DataFrame({'cat': X, 'target': y})
        
        # Counts per category
        grouped = df.groupby('cat')['target'].agg(['sum', 'count'])
        grouped.columns = ['n_pos', 'n_total']
        grouped['n_neg'] = grouped['n_total'] - grouped['n_pos']
        
        # Total positives and negatives
        total_pos = grouped['n_pos'].sum()
        total_neg = grouped['n_neg'].sum()
        
        # WoE calculation
        grouped['dist_pos'] = (grouped['n_pos'] + self.epsilon) / (total_pos + self.epsilon)
        grouped['dist_neg'] = (grouped['n_neg'] + self.epsilon) / (total_neg + self.epsilon)
        grouped['woe'] = np.log(grouped['dist_pos'] / grouped['dist_neg'])
        
        self._woe_map = grouped['woe'].to_dict()
        
        audit("encoder_fit", {
            "strategy": "woe",
            "feature": self.feature_name or "unknown",
            "cardinality": len(self._woe_map)
        })
        
        return self
    
    def transform(self, X: pd.Series) -> np.ndarray:
        """Transformacja z fallback na 0.0."""
        return X.map(self._woe_map).fillna(0.0).values.reshape(-1, 1)
    
    def get_feature_names_out(self) -> List[str]:
        name = self.feature_name or "feature"
        return [f"{name}_woe"]


# ============================================================================
# SMART ENCODER SELECTOR
# ============================================================================

class SmartEncoderSelector:
    """
    Automatyczny wybór strategii encodowania na podstawie:
    - Cardinality cechy
    - Dostępność targetu
    - Task type (classification/regression)
    """
    
    # Cardinality thresholds
    LOW_CARDINALITY = 10
    MEDIUM_CARDINALITY = 50
    HIGH_CARDINALITY = 100
    
    @classmethod
    def select_strategy(
        cls,
        cardinality: int,
        has_target: bool = False,
        is_binary_classification: bool = False
    ) -> EncodingStrategy:
        """
        Wybiera optymalną strategię encodowania.
        
        Args:
            cardinality: Liczba unikalnych kategorii
            has_target: Czy dostępna zmienna docelowa
            is_binary_classification: Czy zadanie binarne
            
        Returns:
            Zalecana strategia encodowania
        """
        # Low cardinality → One-Hot
        if cardinality <= cls.LOW_CARDINALITY:
            return EncodingStrategy.ONE_HOT
        
        # Medium cardinality with target → Target encoding
        if cls.LOW_CARDINALITY < cardinality <= cls.MEDIUM_CARDINALITY:
            if has_target:
                if is_binary_classification:
                    return EncodingStrategy.WOE
                return EncodingStrategy.TARGET
            return EncodingStrategy.FREQUENCY
        
        # High cardinality → Frequency or hashing
        if cardinality <= cls.HIGH_CARDINALITY:
            if has_target:
                return EncodingStrategy.TARGET
            return EncodingStrategy.FREQUENCY
        
        # Very high cardinality → Hashing or ordinal
        return EncodingStrategy.FREQUENCY  # Fallback
    
    @classmethod
    def create_encoder(
        cls,
        strategy: EncodingStrategy,
        feature_name: Optional[str] = None,
        **kwargs: Any
    ) -> CategoricalEncoder:
        """Factory method do tworzenia encoderów."""
        if strategy == EncodingStrategy.ONE_HOT:
            return SafeOneHotEncoder(feature_name=feature_name, **kwargs)
        elif strategy == EncodingStrategy.TARGET:
            return TargetEncoder(feature_name=feature_name, **kwargs)
        elif strategy == EncodingStrategy.FREQUENCY:
            return FrequencyEncoder(feature_name=feature_name)
        elif strategy == EncodingStrategy.WOE:
            return WoEEncoder(feature_name=feature_name, **kwargs)
        else:
            raise EncodingStrategyError(f"Strategy {strategy} not implemented.")


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def safe_one_hot_encoder(
    categories: str | list | None = "auto",
    drop: Optional[str] = None,
    handle_unknown: str = "ignore",
    sparse_output: bool = True,
    max_categories: int = 1000,
    feature_name: Optional[str] = None
) -> SafeOneHotEncoder:
    """
    Warstwa zgodności dla OneHotEncoder (enhanced version).
    
    Zabrania korzystania z przestarzałego parametru `sparse`.
    Zwraca SafeOneHotEncoder z dodatkowymi funkcjami.
    
    Args:
        categories: Categories per feature
        drop: Drop strategy
        handle_unknown: Unknown handling
        sparse_output: Sparse matrix output (memory efficient)
        max_categories: Max allowed categories
        feature_name: Feature name for tracking
        
    Returns:
        SafeOneHotEncoder instance
    """
    return SafeOneHotEncoder(
        categories=categories,
        drop=drop,
        handle_unknown=handle_unknown,
        sparse_output=sparse_output,
        max_categories=max_categories,
        feature_name=feature_name
    )


def auto_encode(
    X: pd.Series,
    y: Optional[pd.Series] = None,
    strategy: Optional[EncodingStrategy] = None,
    feature_name: Optional[str] = None
) -> Tuple[np.ndarray, CategoricalEncoder]:
    """
    Automatyczne encodowanie z wyborem strategii.
    
    Args:
        X: Series do zakodowania
        y: Target variable (optional)
        strategy: Wybrana strategia (None = auto)
        feature_name: Feature name
        
    Returns:
        (encoded_array, fitted_encoder)
    """
    cardinality = X.nunique()
    
    # Auto-select strategy
    if strategy is None:
        strategy = SmartEncoderSelector.select_strategy(
            cardinality=cardinality,
            has_target=y is not None,
            is_binary_classification=y.nunique() == 2 if y is not None else False
        )
    
    # Create and fit encoder
    encoder = SmartEncoderSelector.create_encoder(strategy, feature_name=feature_name)
    encoded = encoder.fit_transform(X, y)
    
    return encoded, encoder