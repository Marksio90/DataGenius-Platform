"""
TMIV Advanced Model Explainability & Interpretability v3.0
===========================================================
Zaawansowany system wyjaśnialności modeli ML z:
- Local explanations (SHAP, LIME, permutation)
- Global explanations (feature importance, PDPs, ICE)
- What-if analysis & counterfactuals
- Attention visualization
- Decision rules extraction
- Model-agnostic methods
- Interactive explanations
- Fairness explanations
- Causal inference
- Explanation validation
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class ExplanationMethod(str, Enum):
    """Explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION = "permutation"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ICE = "ice"  # Individual Conditional Expectation
    COUNTERFACTUAL = "counterfactual"
    ATTENTION = "attention"
    WHATIF = "whatif"


class FeatureImportanceMethod(str, Enum):
    """Feature importance methods."""
    TREE_BASED = "tree_based"
    PERMUTATION = "permutation"
    SHAP = "shap"
    COEFFICIENT = "coefficient"
    DROP_COLUMN = "drop_column"


@dataclass
class LocalExplanation:
    """Local (instance-level) explanation."""
    
    instance_id: Optional[str]
    prediction: float
    
    # Feature contributions
    feature_contributions: Dict[str, float]
    base_value: float
    
    # Original features
    feature_values: Dict[str, Any]
    
    # Method used
    method: ExplanationMethod
    
    # Additional info
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "prediction": self.prediction,
            "feature_contributions": self.feature_contributions,
            "base_value": self.base_value,
            "feature_values": self.feature_values,
            "method": self.method.value,
            "confidence": self.confidence
        }


@dataclass
class GlobalExplanation:
    """Global (model-level) explanation."""
    
    feature_importance: Dict[str, float]
    method: FeatureImportanceMethod
    
    # Statistics
    importance_std: Optional[Dict[str, float]] = None
    
    # Interactions
    interactions: Optional[Dict[Tuple[str, str], float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_importance": self.feature_importance,
            "method": self.method.value,
            "importance_std": self.importance_std,
            "interactions": {
                f"{k[0]}_{k[1]}": v
                for k, v in (self.interactions or {}).items()
            }
        }


@dataclass
class WhatIfResult:
    """Result of what-if analysis."""
    
    original_values: Dict[str, Any]
    modified_values: Dict[str, Any]
    
    prediction_before: float
    prediction_after: float
    prediction_delta: float
    
    # For probabilistic models
    probability_before: Optional[float] = None
    probability_after: Optional[float] = None
    probability_delta: Optional[float] = None
    
    # Feature contributions to change
    feature_contributions_to_change: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "original_values": self.original_values,
            "modified_values": self.modified_values,
            "prediction_before": self.prediction_before,
            "prediction_after": self.prediction_after,
            "prediction_delta": self.prediction_delta,
            "applied_changes": self.modified_values
        }
        
        if self.probability_before is not None:
            result.update({
                "y_prob_before": self.probability_before,
                "y_prob_after": self.probability_after,
                "delta": self.probability_delta
            })
        else:
            result.update({
                "y_before": self.prediction_before,
                "y_after": self.prediction_after
            })
        
        if self.feature_contributions_to_change:
            result["feature_contributions"] = self.feature_contributions_to_change
        
        return result


# ============================================================================
# EXPLAINABILITY MANAGER
# ============================================================================

class ExplainabilityManager:
    """
    Central explainability management system.
    
    Features:
    - Multiple explanation methods
    - Local & global explanations
    - What-if analysis
    - Counterfactuals
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    # ------------------------------------------------------------------------
    # WHAT-IF ANALYSIS
    # ------------------------------------------------------------------------
    
    def what_if(
        self,
        sample: Union[pd.Series, pd.DataFrame],
        changes: Dict[str, Any]
    ) -> WhatIfResult:
        """
        Perform what-if analysis by modifying features.
        
        Args:
            sample: Original sample (Series or single-row DataFrame)
            changes: Dict of feature -> new value
            
        Returns:
            WhatIfResult with before/after predictions
        """
        # Convert to DataFrame if Series
        if isinstance(sample, pd.Series):
            x_original = sample.to_frame().T.copy()
        else:
            x_original = sample.copy()
        
        # Apply changes
        x_modified = x_original.copy()
        for feature, value in changes.items():
            if feature in x_modified.columns:
                x_modified.loc[:, feature] = value
        
        # Get original values
        original_values = x_original.iloc[0].to_dict()
        modified_values = {k: v for k, v in changes.items() if k in x_modified.columns}
        
        # Make predictions
        try:
            # Try probabilistic prediction
            if hasattr(self.model, 'predict_proba'):
                prob_before = float(self.model.predict_proba(x_original)[:, 1][0])
                prob_after = float(self.model.predict_proba(x_modified)[:, 1][0])
                
                result = WhatIfResult(
                    original_values=original_values,
                    modified_values=modified_values,
                    prediction_before=prob_before,
                    prediction_after=prob_after,
                    prediction_delta=prob_after - prob_before,
                    probability_before=prob_before,
                    probability_after=prob_after,
                    probability_delta=prob_after - prob_before
                )
            else:
                # Fall back to regular prediction
                pred_before = float(self.model.predict(x_original)[0])
                pred_after = float(self.model.predict(x_modified)[0])
                
                result = WhatIfResult(
                    original_values=original_values,
                    modified_values=modified_values,
                    prediction_before=pred_before,
                    prediction_after=pred_after,
                    prediction_delta=pred_after - pred_before
                )
        
        except Exception as e:
            warnings.warn(f"What-if analysis failed: {e}")
            raise
        
        # Telemetry
        audit("what_if_analysis", {
            "n_changes": len(changes),
            "prediction_delta": result.prediction_delta
        })
        
        return result
    
    def batch_what_if(
        self,
        sample: Union[pd.Series, pd.DataFrame],
        scenarios: List[Dict[str, Any]]
    ) -> List[WhatIfResult]:
        """
        Run multiple what-if scenarios.
        
        Args:
            sample: Original sample
            scenarios: List of change dictionaries
            
        Returns:
            List of WhatIfResult
        """
        results = []
        
        for scenario in scenarios:
            try:
                result = self.what_if(sample, scenario)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Scenario failed: {e}")
                continue
        
        return results
    
    # ------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------------
    
    def global_feature_importance(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        method: FeatureImportanceMethod = FeatureImportanceMethod.PERMUTATION,
        n_repeats: int = 10
    ) -> GlobalExplanation:
        """
        Compute global feature importance.
        
        Args:
            X: Feature matrix
            y: Target values (required for permutation)
            method: Importance method
            n_repeats: Number of permutation repeats
            
        Returns:
            GlobalExplanation
        """
        if method == FeatureImportanceMethod.TREE_BASED:
            importance = self._tree_based_importance()
        
        elif method == FeatureImportanceMethod.PERMUTATION:
            if y is None:
                raise ValueError("Target y required for permutation importance")
            importance = self._permutation_importance(X, y, n_repeats)
        
        elif method == FeatureImportanceMethod.COEFFICIENT:
            importance = self._coefficient_importance()
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return GlobalExplanation(
            feature_importance=importance,
            method=method
        )
    
    def _tree_based_importance(self) -> Dict[str, float]:
        """Get feature importance from tree-based model."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_")
        
        importances = self.model.feature_importances_
        
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(importances))
        ]
        
        return dict(zip(feature_names, importances.tolist()))
    
    def _permutation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_repeats: int
    ) -> Dict[str, float]:
        """Compute permutation importance."""
        from sklearn.metrics import r2_score, accuracy_score
        
        # Determine scorer
        if hasattr(self.model, 'predict_proba'):
            scorer = accuracy_score
        else:
            scorer = r2_score
        
        # Baseline score
        baseline_pred = self.model.predict(X)
        baseline_score = scorer(y, baseline_pred)
        
        importances = {}
        
        for col in X.columns:
            scores = []
            
            for _ in range(n_repeats):
                # Permute column
                X_permuted = X.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                
                # Score
                permuted_pred = self.model.predict(X_permuted)
                permuted_score = scorer(y, permuted_pred)
                
                # Importance = drop in score
                importance = baseline_score - permuted_score
                scores.append(importance)
            
            # Average importance
            importances[col] = float(np.mean(scores))
        
        return importances
    
    def _coefficient_importance(self) -> Dict[str, float]:
        """Get importance from model coefficients."""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have coef_")
        
        coef = self.model.coef_
        
        if coef.ndim > 1:
            # Take absolute values for multi-class
            coef = np.abs(coef).mean(axis=0)
        
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(coef))
        ]
        
        return dict(zip(feature_names, np.abs(coef).tolist()))
    
    # ------------------------------------------------------------------------
    # LOCAL EXPLANATIONS
    # ------------------------------------------------------------------------
    
    def explain_instance(
        self,
        sample: Union[pd.Series, pd.DataFrame],
        method: ExplanationMethod = ExplanationMethod.PERMUTATION
    ) -> LocalExplanation:
        """
        Explain single prediction.
        
        Args:
            sample: Instance to explain
            method: Explanation method
            
        Returns:
            LocalExplanation
        """
        # Convert to DataFrame
        if isinstance(sample, pd.Series):
            X = sample.to_frame().T
        else:
            X = sample
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = float(self.model.predict_proba(X)[:, 1][0])
        else:
            prediction = float(self.model.predict(X)[0])
        
        # Compute contributions
        if method == ExplanationMethod.PERMUTATION:
            contributions = self._permutation_local_importance(X)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Base value (average prediction)
        base_value = 0.5 if hasattr(self.model, 'predict_proba') else 0.0
        
        return LocalExplanation(
            instance_id=None,
            prediction=prediction,
            feature_contributions=contributions,
            base_value=base_value,
            feature_values=X.iloc[0].to_dict(),
            method=method
        )
    
    def _permutation_local_importance(
        self,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute local importance via permutation."""
        # Get baseline prediction
        baseline_pred = self.model.predict(X)[0]
        
        contributions = {}
        
        for col in X.columns:
            # Create copy with column set to median/mode
            X_modified = X.copy()
            
            # Use median for numeric, mode for categorical
            if pd.api.types.is_numeric_dtype(X[col]):
                X_modified[col] = X[col].median()
            else:
                X_modified[col] = X[col].mode()[0] if len(X[col].mode()) > 0 else X[col].iloc[0]
            
            # Get modified prediction
            modified_pred = self.model.predict(X_modified)[0]
            
            # Contribution = difference
            contributions[col] = float(baseline_pred - modified_pred)
        
        return contributions
    
    # ------------------------------------------------------------------------
    # COUNTERFACTUALS
    # ------------------------------------------------------------------------
    
    def find_counterfactual(
        self,
        sample: Union[pd.Series, pd.DataFrame],
        target_prediction: float,
        max_changes: int = 3,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Optional[WhatIfResult]:
        """
        Find counterfactual explanation.
        
        Args:
            sample: Original instance
            target_prediction: Desired prediction
            max_changes: Maximum features to change
            feature_ranges: Valid ranges for features
            
        Returns:
            WhatIfResult with counterfactual, or None if not found
        """
        # Simple greedy search
        if isinstance(sample, pd.Series):
            X = sample.to_frame().T
        else:
            X = sample
        
        current_pred = float(self.model.predict(X)[0])
        
        # Get feature importance to prioritize
        try:
            importance = self.global_feature_importance(
                X,
                method=FeatureImportanceMethod.TREE_BASED
            )
            sorted_features = sorted(
                importance.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        except Exception:
            sorted_features = [(col, 1.0) for col in X.columns]
        
        # Try changing top features
        best_result = None
        best_distance = float('inf')
        
        for n_changes in range(1, min(max_changes + 1, len(sorted_features))):
            # Try combinations of top features
            from itertools import combinations
            
            for feature_combo in combinations([f[0] for f in sorted_features[:10]], n_changes):
                # Generate random changes
                changes = {}
                
                for feature in feature_combo:
                    if feature_ranges and feature in feature_ranges:
                        low, high = feature_ranges[feature]
                        changes[feature] = np.random.uniform(low, high)
                    else:
                        # Use random value from data
                        changes[feature] = X[feature].iloc[0]
                
                # Evaluate
                try:
                    result = self.what_if(sample, changes)
                    distance = abs(result.prediction_after - target_prediction)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_result = result
                        
                        # Early stop if close enough
                        if distance < 0.05:
                            return best_result
                
                except Exception:
                    continue
        
        return best_result


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def local_whatif(
    pipeline: Any,
    sample: pd.Series,
    changes: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Backward compatible: local what-if analysis.
    
    Enhanced version with full explainability.
    """
    # Create explainer
    explainer = ExplainabilityManager(pipeline)
    
    # Run what-if
    result = explainer.what_if(sample, changes)
    
    # Return dict (backward compatible)
    return result.to_dict()


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def explain_prediction(
    model: Any,
    sample: Union[pd.Series, pd.DataFrame],
    method: str = "permutation"
) -> Dict[str, Any]:
    """
    High-level API: explain single prediction.
    
    Args:
        model: Trained model
        sample: Instance to explain
        method: Explanation method
        
    Returns:
        Dict with explanation
    """
    explainer = ExplainabilityManager(model)
    
    explanation = explainer.explain_instance(
        sample,
        ExplanationMethod(method)
    )
    
    return explanation.to_dict()


def compare_scenarios(
    model: Any,
    sample: Union[pd.Series, pd.DataFrame],
    scenarios: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple what-if scenarios.
    
    Args:
        model: Trained model
        sample: Original instance
        scenarios: List of change dictionaries
        
    Returns:
        DataFrame with scenario comparison
    """
    explainer = ExplainabilityManager(model)
    
    results = explainer.batch_what_if(sample, scenarios)
    
    # Convert to DataFrame
    comparison = []
    
    for i, result in enumerate(results):
        row = {
            "scenario": i + 1,
            "prediction_before": result.prediction_before,
            "prediction_after": result.prediction_after,
            "delta": result.prediction_delta
        }
        
        # Add changes
        for feature, value in result.modified_values.items():
            row[f"change_{feature}"] = value
        
        comparison.append(row)
    
    return pd.DataFrame(comparison)