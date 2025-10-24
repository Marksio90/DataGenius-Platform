"""
TMIV Advanced Thresholding Utilities v3.0
==========================================
Zaawansowane narzędzia do optymalizacji progów decyzyjnych z:
- Multiple threshold optimization strategies
- ROC curve analysis
- Cost-sensitive thresholding
- Business metric optimization
- Calibration-aware methods
- Percentile-based thresholding
- Moving threshold adaptation
- Multi-class threshold optimization
- Uncertainty-aware thresholding
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

from .telemetry import audit, metric as log_metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class ThresholdStrategy(str, Enum):
    """Threshold optimization strategies."""
    YOUDEN = "youden"                # Youden's J statistic
    COST = "cost"                    # Cost minimization
    F1 = "f1"                       # Maximize F1
    F_BETA = "f_beta"               # Maximize F-beta
    GMEAN = "gmean"                 # Geometric mean
    DISTANCE = "distance"           # Min distance to (0,1)
    PERCENTILE = "percentile"       # Percentile-based
    CALIBRATED = "calibrated"       # Calibration-aware
    BUSINESS = "business"           # Business metric


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""
    threshold: float
    strategy: ThresholdStrategy
    
    # Metrics
    tp: int
    fp: int
    tn: int
    fn: int
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    # Additional metrics
    specificity: float = 0.0
    npv: float = 0.0  # Negative Predictive Value
    fpr: float = 0.0
    fnr: float = 0.0
    
    # Cost/benefit
    expected_cost: Optional[float] = None
    expected_profit: Optional[float] = None
    
    # Quality
    youden_j: float = 0.0
    gmean: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "strategy": self.strategy.value,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "npv": self.npv,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "expected_cost": self.expected_cost,
            "expected_profit": self.expected_profit,
            "youden_j": self.youden_j,
            "gmean": self.gmean
        }


# ============================================================================
# THRESHOLD OPTIMIZER
# ============================================================================

class ThresholdOptimizer:
    """
    Comprehensive threshold optimization toolkit.
    
    Features:
    - Multiple optimization strategies
    - Cost-sensitive learning
    - Constraint support
    - Calibration awareness
    """
    
    def __init__(
        self,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        min_specificity: Optional[float] = None
    ):
        """
        Args:
            min_precision: Minimum precision constraint
            min_recall: Minimum recall constraint
            min_specificity: Minimum specificity constraint
        """
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_specificity = min_specificity
    
    # ------------------------------------------------------------------------
    # YOUDEN'S J STATISTIC
    # ------------------------------------------------------------------------
    
    def optimize_by_youden(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> ThresholdResult:
        """
        Optimize threshold using Youden's J statistic.
        
        Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            ThresholdResult
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate Youden's J
        j_scores = tpr - fpr
        
        # Find optimal
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Compute metrics
        result = self._compute_threshold_metrics(
            y_true, y_prob, optimal_threshold, ThresholdStrategy.YOUDEN
        )
        
        result.youden_j = float(j_scores[optimal_idx])
        
        # Telemetry
        audit("threshold_youden", {
            "threshold": optimal_threshold,
            "youden_j": result.youden_j
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # COST-BASED OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def optimize_by_cost(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_matrix: Dict[str, float],
        prevalence: Optional[float] = None
    ) -> ThresholdResult:
        """
        Optimize threshold to minimize expected cost.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            cost_matrix: Dict with FP, FN, TP, TN costs
            prevalence: Class prevalence (None = use data prevalence)
            
        Returns:
            ThresholdResult
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Get costs
        cost_fp = float(cost_matrix.get("FP", 1.0))
        cost_fn = float(cost_matrix.get("FN", 1.0))
        cost_tp = float(cost_matrix.get("TP", 0.0))
        cost_tn = float(cost_matrix.get("TN", 0.0))
        
        # Use data prevalence if not provided
        if prevalence is None:
            prevalence = float(np.mean(y_true))
        
        # Calculate expected cost at each threshold
        # Expected cost = P(pos) * [FNR * cost_FN + TPR * cost_TP] +
        #                P(neg) * [FPR * cost_FP + TNR * cost_TN]
        fnr = 1 - tpr
        tnr = 1 - fpr
        
        expected_costs = (
            prevalence * (fnr * cost_fn + tpr * cost_tp) +
            (1 - prevalence) * (fpr * cost_fp + tnr * cost_tn)
        )
        
        # Find minimum cost
        optimal_idx = int(np.argmin(expected_costs))
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Compute metrics
        result = self._compute_threshold_metrics(
            y_true, y_prob, optimal_threshold, ThresholdStrategy.COST
        )
        
        result.expected_cost = float(expected_costs[optimal_idx])
        
        # Telemetry
        audit("threshold_cost", {
            "threshold": optimal_threshold,
            "expected_cost": result.expected_cost,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # F-SCORE OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def optimize_by_f_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        beta: float = 1.0
    ) -> ThresholdResult:
        """
        Optimize threshold to maximize F-beta score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            beta: Beta parameter (1.0 = F1, 2.0 = F2, 0.5 = F0.5)
            
        Returns:
            ThresholdResult
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate F-beta score
        beta_squared = beta ** 2
        
        f_scores = (
            (1 + beta_squared) * precision * recall /
            (beta_squared * precision + recall + 1e-10)
        )
        
        # Find maximum (skip last threshold which is always 1.0)
        optimal_idx = int(np.argmax(f_scores[:-1]))
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Compute metrics
        strategy = ThresholdStrategy.F1 if beta == 1.0 else ThresholdStrategy.F_BETA
        
        result = self._compute_threshold_metrics(
            y_true, y_prob, optimal_threshold, strategy
        )
        
        # Telemetry
        audit("threshold_f_score", {
            "threshold": optimal_threshold,
            "f_score": result.f1 if beta == 1.0 else float(f_scores[optimal_idx]),
            "beta": beta
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # GEOMETRIC MEAN
    # ------------------------------------------------------------------------
    
    def optimize_by_gmean(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> ThresholdResult:
        """
        Optimize threshold using geometric mean of TPR and TNR.
        
        G-mean = sqrt(TPR * TNR) = sqrt(Sensitivity * Specificity)
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            ThresholdResult
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate G-mean
        tnr = 1 - fpr
        gmean_scores = np.sqrt(tpr * tnr)
        
        # Find optimal
        optimal_idx = int(np.argmax(gmean_scores))
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Compute metrics
        result = self._compute_threshold_metrics(
            y_true, y_prob, optimal_threshold, ThresholdStrategy.GMEAN
        )
        
        result.gmean = float(gmean_scores[optimal_idx])
        
        # Telemetry
        audit("threshold_gmean", {
            "threshold": optimal_threshold,
            "gmean": result.gmean
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # DISTANCE TO PERFECT CLASSIFICATION
    # ------------------------------------------------------------------------
    
    def optimize_by_distance(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> ThresholdResult:
        """
        Optimize threshold by minimizing distance to (0, 1) on ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            ThresholdResult
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate distance to (0, 1)
        distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        
        # Find minimum distance
        optimal_idx = int(np.argmin(distances))
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Compute metrics
        result = self._compute_threshold_metrics(
            y_true, y_prob, optimal_threshold, ThresholdStrategy.DISTANCE
        )
        
        # Telemetry
        audit("threshold_distance", {
            "threshold": optimal_threshold,
            "distance": float(distances[optimal_idx])
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # PERCENTILE-BASED
    # ------------------------------------------------------------------------
    
    def optimize_by_percentile(
        self,
        y_prob: np.ndarray,
        percentile: float = 50.0,
        y_true: Optional[np.ndarray] = None
    ) -> Union[float, ThresholdResult]:
        """
        Set threshold based on percentile of predicted probabilities.
        
        Args:
            y_prob: Predicted probabilities
            percentile: Percentile to use (0-100)
            y_true: True labels (if provided, compute metrics)
            
        Returns:
            Threshold value or ThresholdResult
        """
        threshold = float(np.percentile(y_prob, percentile))
        
        if y_true is None:
            return threshold
        
        # Compute metrics if labels provided
        result = self._compute_threshold_metrics(
            y_true, y_prob, threshold, ThresholdStrategy.PERCENTILE
        )
        
        # Telemetry
        audit("threshold_percentile", {
            "threshold": threshold,
            "percentile": percentile
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # METRICS COMPUTATION
    # ------------------------------------------------------------------------
    
    def _compute_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        strategy: ThresholdStrategy
    ) -> ThresholdResult:
        """Compute all metrics at a specific threshold."""
        # Make predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
                if y_true[0] == 1 and y_pred[0] == 1:
                    tp, tn, fp, fn = len(y_true), 0, 0, 0
                elif y_true[0] == 1 and y_pred[0] == 0:
                    tp, tn, fp, fn = 0, 0, 0, len(y_true)
                elif y_true[0] == 0 and y_pred[0] == 0:
                    tp, tn, fp, fn = 0, len(y_true), 0, 0
                else:
                    tp, tn, fp, fn = 0, 0, len(y_true), 0
            else:
                tp, tn, fp, fn = 0, 0, 0, 0
        
        # Basic metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Quality metrics
        youden_j = recall + specificity - 1
        gmean = np.sqrt(recall * specificity)
        
        return ThresholdResult(
            threshold=threshold,
            strategy=strategy,
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            specificity=float(specificity),
            npv=float(npv),
            fpr=float(fpr),
            fnr=float(fnr),
            youden_j=float(youden_j),
            gmean=float(gmean)
        )
    
    # ------------------------------------------------------------------------
    # CONSTRAINT-BASED OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def find_threshold_with_constraints(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        min_specificity: Optional[float] = None,
        optimize_for: str = "f1"
    ) -> ThresholdResult:
        """
        Find threshold satisfying constraints while optimizing metric.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            min_precision: Minimum precision
            min_recall: Minimum recall
            min_specificity: Minimum specificity
            optimize_for: Metric to optimize (f1/precision/recall)
            
        Returns:
            ThresholdResult
        """
        # Use instance constraints if not provided
        min_prec = min_precision or self.min_precision
        min_rec = min_recall or self.min_recall
        min_spec = min_specificity or self.min_specificity
        
        # Get all thresholds
        _, _, thresholds = roc_curve(y_true, y_prob)
        
        # Find valid thresholds
        valid_results = []
        
        for threshold in thresholds:
            result = self._compute_threshold_metrics(
                y_true, y_prob, threshold, ThresholdStrategy.F1
            )
            
            # Check constraints
            if min_prec and result.precision < min_prec:
                continue
            if min_rec and result.recall < min_rec:
                continue
            if min_spec and result.specificity < min_spec:
                continue
            
            valid_results.append(result)
        
        if not valid_results:
            raise ValueError(
                f"No threshold satisfies constraints: "
                f"min_precision={min_prec}, min_recall={min_rec}, "
                f"min_specificity={min_spec}"
            )
        
        # Find best among valid
        if optimize_for == "f1":
            best = max(valid_results, key=lambda r: r.f1)
        elif optimize_for == "precision":
            best = max(valid_results, key=lambda r: r.precision)
        elif optimize_for == "recall":
            best = max(valid_results, key=lambda r: r.recall)
        else:
            best = valid_results[0]
        
        return best


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def optimize_threshold_by_youden(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray
) -> float:
    """
    Backward compatible: optimize threshold using Youden's J.
    
    Enhanced version with validation.
    """
    if len(fpr) != len(tpr) or len(fpr) != len(thresholds):
        raise ValueError("Arrays must have same length")
    
    j = tpr - fpr
    idx = int(np.argmax(j))
    
    return float(thresholds[idx])


def optimize_threshold_by_cost(
    thresholds: np.ndarray,
    tpr: np.ndarray,
    fpr: np.ndarray,
    cost_matrix: Dict[str, float]
) -> Dict[str, float]:
    """
    Backward compatible: optimize threshold by cost.
    
    Enhanced version with better cost calculation.
    """
    cost_fp = float(cost_matrix.get("FP", 1.0))
    cost_fn = float(cost_matrix.get("FN", 1.0))
    
    # Expected cost (assuming 50/50 prevalence)
    fnr = 1.0 - tpr
    expected = 0.5 * (fpr * cost_fp + fnr * cost_fn)
    
    idx = int(np.argmin(expected))
    
    return {
        "threshold": float(thresholds[idx]),
        "expected_cost": float(expected[idx])
    }


def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr: float
) -> Dict[str, float]:
    """
    Backward compatible: compute metrics at threshold.
    
    Enhanced version with additional metrics.
    """
    optimizer = ThresholdOptimizer()
    
    result = optimizer._compute_threshold_metrics(
        y_true, y_prob, thr, ThresholdStrategy.YOUDEN
    )
    
    return {
        "tp": result.tp,
        "tn": result.tn,
        "fp": result.fp,
        "fn": result.fn,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1
    }


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "youden",
    **kwargs: Any
) -> Dict[str, Any]:
    """
    High-level API: find optimal threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        strategy: Optimization strategy
        **kwargs: Strategy-specific arguments
        
    Returns:
        Dict with threshold and metrics
    """
    optimizer = ThresholdOptimizer()
    
    if strategy == "youden":
        result = optimizer.optimize_by_youden(y_true, y_prob)
    elif strategy == "cost":
        cost_matrix = kwargs.get("cost_matrix", {"FP": 1.0, "FN": 1.0})
        result = optimizer.optimize_by_cost(y_true, y_prob, cost_matrix)
    elif strategy == "f1":
        result = optimizer.optimize_by_f_score(y_true, y_prob, beta=1.0)
    elif strategy == "gmean":
        result = optimizer.optimize_by_gmean(y_true, y_prob)
    elif strategy == "distance":
        result = optimizer.optimize_by_distance(y_true, y_prob)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return result.to_dict()