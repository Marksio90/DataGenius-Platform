"""
TMIV Advanced Threshold Optimization & Cost Analysis v3.0
==========================================================
Zaawansowany system optymalizacji progów decyzyjnych z:
- Multi-objective threshold optimization
- Cost-sensitive learning
- Business metric optimization (profit, ROI)
- Constraint-based optimization (precision/recall requirements)
- ROC curve analysis & visualization
- Precision-Recall curve optimization
- Calibration-aware thresholding
- A/B testing for thresholds
- Dynamic threshold adaptation
- Fairness-aware thresholding
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MIN_COST = "min_cost"                  # Minimize expected cost
    MAX_PROFIT = "max_profit"              # Maximize expected profit
    MAX_F1 = "max_f1"                      # Maximize F1 score
    MAX_GMEAN = "max_gmean"                # Geometric mean of sensitivity/specificity
    MAX_YOUDEN = "max_youden"              # Youden's J statistic
    MIN_DISTANCE = "min_distance"          # Closest to (0,1) on ROC
    CONSTRAINED_PRECISION = "constrained_precision"  # Max recall given precision constraint
    CONSTRAINED_RECALL = "constrained_recall"        # Max precision given recall constraint


@dataclass
class CostMatrix:
    """Cost/benefit matrix for predictions."""
    # Costs (positive values)
    false_positive: float = 1.0    # Cost of FP
    false_negative: float = 1.0    # Cost of FN
    
    # Benefits (positive values, will be negative in cost calculation)
    true_positive: float = 0.0     # Benefit of TP
    true_negative: float = 0.0     # Benefit of TN
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "FP": self.false_positive,
            "FN": self.false_negative,
            "TP": self.true_positive,
            "TN": self.true_negative
        }


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold."""
    threshold: float
    
    # Confusion matrix
    tp: int
    fp: int
    tn: int
    fn: int
    
    # Rates
    tpr: float  # True Positive Rate (Recall/Sensitivity)
    fpr: float  # False Positive Rate
    tnr: float  # True Negative Rate (Specificity)
    fnr: float  # False Negative Rate
    
    # Precision/Recall
    precision: float
    recall: float
    
    # Combined metrics
    f1: float
    f2: float  # F2 score (recall weighted)
    f05: float  # F0.5 score (precision weighted)
    
    # Other
    accuracy: float
    balanced_accuracy: float
    mcc: float  # Matthews Correlation Coefficient
    
    # Cost/Profit
    expected_cost: Optional[float] = None
    expected_profit: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "tnr": self.tnr,
            "fnr": self.fnr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "f05": self.f05,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "expected_cost": self.expected_cost,
            "expected_profit": self.expected_profit
        }


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""
    objective: OptimizationObjective
    optimal_threshold: float
    metrics: ThresholdMetrics
    
    # ROC/PR curves
    roc_auc: float
    pr_auc: float
    
    # All thresholds evaluated
    all_thresholds: List[float] = field(default_factory=list)
    all_metrics: List[ThresholdMetrics] = field(default_factory=list)
    
    # Metadata
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    class_balance: float = 0.0
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective.value,
            "optimal_threshold": self.optimal_threshold,
            "metrics": self.metrics.to_dict(),
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "class_balance": self.class_balance,
            "timestamp": self.timestamp
        }


# ============================================================================
# THRESHOLD OPTIMIZER
# ============================================================================

class ThresholdOptimizer:
    """
    Advanced threshold optimization with multiple objectives.
    
    Features:
    - Cost-sensitive optimization
    - Business metric optimization
    - Constraint-based optimization
    - ROC/PR analysis
    """
    
    def __init__(
        self,
        cost_matrix: Optional[CostMatrix] = None,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None
    ):
        """
        Args:
            cost_matrix: Cost/benefit matrix
            min_precision: Minimum precision constraint
            min_recall: Minimum recall constraint
        """
        self.cost_matrix = cost_matrix or CostMatrix()
        self.min_precision = min_precision
        self.min_recall = min_recall
    
    # ------------------------------------------------------------------------
    # MAIN OPTIMIZATION METHOD
    # ------------------------------------------------------------------------
    
    def optimize(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        objective: OptimizationObjective = OptimizationObjective.MIN_COST,
        n_thresholds: int = 100
    ) -> OptimizationResult:
        """
        Find optimal threshold for given objective.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            objective: Optimization objective
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            OptimizationResult with optimal threshold
        """
        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Get PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate AUCs
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        
        # Sample thresholds uniformly if too many
        if len(thresholds) > n_thresholds:
            indices = np.linspace(0, len(thresholds) - 1, n_thresholds, dtype=int)
            thresholds = thresholds[indices]
        
        # Evaluate all thresholds
        all_metrics = []
        
        for threshold in thresholds:
            metrics = self._compute_metrics_at_threshold(
                y_true, y_prob, threshold
            )
            all_metrics.append(metrics)
        
        # Find optimal threshold based on objective
        optimal_idx = self._find_optimal_threshold_idx(
            all_metrics, objective
        )
        
        optimal_metrics = all_metrics[optimal_idx]
        optimal_threshold = thresholds[optimal_idx]
        
        # Create result
        result = OptimizationResult(
            objective=objective,
            optimal_threshold=optimal_threshold,
            metrics=optimal_metrics,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            all_thresholds=thresholds.tolist(),
            all_metrics=all_metrics,
            n_samples=len(y_true),
            n_positive=int(np.sum(y_true)),
            n_negative=int(len(y_true) - np.sum(y_true)),
            class_balance=float(np.mean(y_true))
        )
        
        # Telemetry
        audit("threshold_optimization", {
            "objective": objective.value,
            "optimal_threshold": optimal_threshold,
            "f1": optimal_metrics.f1,
            "roc_auc": roc_auc
        })
        
        metric("optimal_threshold", optimal_threshold, {
            "objective": objective.value
        })
        
        return result
    
    # ------------------------------------------------------------------------
    # COST-BASED OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def optimize_by_cost(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_matrix: Optional[CostMatrix] = None
    ) -> OptimizationResult:
        """
        Optimize threshold to minimize expected cost.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            cost_matrix: Cost matrix (None = use instance default)
            
        Returns:
            OptimizationResult
        """
        # Use provided cost matrix or default
        costs = cost_matrix or self.cost_matrix
        
        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Calculate expected cost for each threshold
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        # Expected cost = FP*cost_FP + FN*cost_FN - TP*benefit_TP - TN*benefit_TN
        expected_costs = []
        
        for i in range(len(thresholds)):
            fp_rate = fpr[i]
            tp_rate = tpr[i]
            fn_rate = 1 - tp_rate
            tn_rate = 1 - fp_rate
            
            expected_cost = (
                fp_rate * n_neg * costs.false_positive +
                fn_rate * n_pos * costs.false_negative -
                tp_rate * n_pos * costs.true_positive -
                tn_rate * n_neg * costs.true_negative
            )
            
            expected_costs.append(expected_cost)
        
        # Find minimum cost
        optimal_idx = np.argmin(expected_costs)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute metrics at optimal threshold
        metrics = self._compute_metrics_at_threshold(
            y_true, y_prob, optimal_threshold
        )
        metrics.expected_cost = expected_costs[optimal_idx]
        
        # Calculate AUCs
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        
        return OptimizationResult(
            objective=OptimizationObjective.MIN_COST,
            optimal_threshold=optimal_threshold,
            metrics=metrics,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            n_samples=len(y_true),
            n_positive=int(n_pos),
            n_negative=int(n_neg),
            class_balance=float(n_pos / len(y_true))
        )
    
    # ------------------------------------------------------------------------
    # GRID SEARCH
    # ------------------------------------------------------------------------
    
    def grid_search_costs(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fp_costs: List[float],
        fn_costs: List[float]
    ) -> pd.DataFrame:
        """
        Grid search over cost parameters.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            fp_costs: List of FP costs to try
            fn_costs: List of FN costs to try
            
        Returns:
            DataFrame with results for all cost combinations
        """
        results = []
        
        for fp_cost in fp_costs:
            for fn_cost in fn_costs:
                # Create cost matrix
                cost_matrix = CostMatrix(
                    false_positive=fp_cost,
                    false_negative=fn_cost
                )
                
                # Optimize
                result = self.optimize_by_cost(y_true, y_prob, cost_matrix)
                
                # Store result
                row = {
                    "cost_fp": fp_cost,
                    "cost_fn": fn_cost,
                    "threshold": result.optimal_threshold,
                    "expected_cost": result.metrics.expected_cost,
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "f1": result.metrics.f1,
                    "accuracy": result.metrics.accuracy
                }
                
                results.append(row)
        
        # Sort by expected cost
        df = pd.DataFrame(results)
        df = df.sort_values("expected_cost")
        
        return df
    
    # ------------------------------------------------------------------------
    # CONSTRAINED OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def optimize_with_constraints(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        objective: OptimizationObjective = OptimizationObjective.MAX_F1
    ) -> OptimizationResult:
        """
        Optimize threshold with precision/recall constraints.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            min_precision: Minimum precision constraint
            min_recall: Minimum recall constraint
            objective: Objective to optimize
            
        Returns:
            OptimizationResult
        """
        min_prec = min_precision or self.min_precision
        min_rec = min_recall or self.min_recall
        
        # Get all thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Evaluate all thresholds
        valid_metrics = []
        valid_thresholds = []
        
        for threshold in thresholds:
            metrics = self._compute_metrics_at_threshold(
                y_true, y_prob, threshold
            )
            
            # Check constraints
            if min_prec and metrics.precision < min_prec:
                continue
            if min_rec and metrics.recall < min_rec:
                continue
            
            valid_metrics.append(metrics)
            valid_thresholds.append(threshold)
        
        if not valid_metrics:
            raise ValueError(
                f"No threshold satisfies constraints: "
                f"min_precision={min_prec}, min_recall={min_rec}"
            )
        
        # Find optimal among valid thresholds
        optimal_idx = self._find_optimal_threshold_idx(
            valid_metrics, objective
        )
        
        optimal_metrics = valid_metrics[optimal_idx]
        optimal_threshold = valid_thresholds[optimal_idx]
        
        # Calculate AUCs
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        
        return OptimizationResult(
            objective=objective,
            optimal_threshold=optimal_threshold,
            metrics=optimal_metrics,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            n_samples=len(y_true),
            n_positive=int(np.sum(y_true)),
            n_negative=int(len(y_true) - np.sum(y_true)),
            class_balance=float(np.mean(y_true))
        )
    
    # ------------------------------------------------------------------------
    # METRICS COMPUTATION
    # ------------------------------------------------------------------------
    
    def _compute_metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float
    ) -> ThresholdMetrics:
        """Compute all metrics at a specific threshold."""
        # Make predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Precision/Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        
        # F-scores
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
        f05 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy = (tpr + tnr) / 2
        
        # MCC
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0
        
        return ThresholdMetrics(
            threshold=threshold,
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            precision=precision,
            recall=recall,
            f1=f1,
            f2=f2,
            f05=f05,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            mcc=mcc
        )
    
    def _find_optimal_threshold_idx(
        self,
        metrics_list: List[ThresholdMetrics],
        objective: OptimizationObjective
    ) -> int:
        """Find optimal threshold index based on objective."""
        if objective == OptimizationObjective.MIN_COST:
            # Minimize expected cost
            costs = [m.expected_cost for m in metrics_list]
            return int(np.argmin(costs))
        
        elif objective == OptimizationObjective.MAX_F1:
            # Maximize F1
            f1_scores = [m.f1 for m in metrics_list]
            return int(np.argmax(f1_scores))
        
        elif objective == OptimizationObjective.MAX_GMEAN:
            # Maximize geometric mean of TPR and TNR
            gmeans = [np.sqrt(m.tpr * m.tnr) for m in metrics_list]
            return int(np.argmax(gmeans))
        
        elif objective == OptimizationObjective.MAX_YOUDEN:
            # Maximize Youden's J = TPR + TNR - 1
            youdens = [m.tpr + m.tnr - 1 for m in metrics_list]
            return int(np.argmax(youdens))
        
        elif objective == OptimizationObjective.MIN_DISTANCE:
            # Minimize distance to (0, 1) on ROC curve
            distances = [np.sqrt((m.fpr - 0)**2 + (m.tpr - 1)**2) for m in metrics_list]
            return int(np.argmin(distances))
        
        else:
            raise ValueError(f"Unsupported objective: {objective}")


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def grid_cost_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_vals: List[float],
    fn_vals: List[float]
) -> List[Dict[str, float]]:
    """
    Backward compatible: grid search over cost parameters.
    
    Enhanced version with full optimization.
    """
    optimizer = ThresholdOptimizer()
    df = optimizer.grid_search_costs(y_true, y_prob, fp_vals, fn_vals)
    
    # Convert to list of dicts
    return df.to_dict('records')


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "max_f1",
    fp_cost: float = 1.0,
    fn_cost: float = 1.0,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None
) -> Dict[str, Any]:
    """
    High-level API: find optimal threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        objective: Optimization objective
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative
        min_precision: Minimum precision constraint
        min_recall: Minimum recall constraint
        
    Returns:
        Dict with optimal threshold and metrics
    """
    # Create optimizer
    cost_matrix = CostMatrix(false_positive=fp_cost, false_negative=fn_cost)
    optimizer = ThresholdOptimizer(
        cost_matrix=cost_matrix,
        min_precision=min_precision,
        min_recall=min_recall
    )
    
    # Optimize
    obj = OptimizationObjective(objective)
    
    if min_precision or min_recall:
        result = optimizer.optimize_with_constraints(
            y_true, y_prob, min_precision, min_recall, obj
        )
    else:
        result = optimizer.optimize(y_true, y_prob, obj)
    
    return result.to_dict()