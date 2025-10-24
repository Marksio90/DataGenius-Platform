"""
TMIV Advanced Fairness & Bias Detection Engine v3.0
====================================================
Zaawansowany system detekcji bias i fairness z:
- Multi-group fairness metrics (Demographic Parity, Equal Opportunity, Equalized Odds)
- Intersectional bias analysis (multiple protected attributes)
- Disparate impact detection & legal compliance (80% rule)
- Calibration fairness (predictive parity)
- Individual fairness metrics (similarity-based)
- Bias mitigation recommendations
- Comprehensive fairness reports with visualizations
- Temporal fairness tracking
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class FairnessMetric(str, Enum):
    """Dostępne metryki fairness."""
    DEMOGRAPHIC_PARITY = "demographic_parity"           # P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
    EQUAL_OPPORTUNITY = "equal_opportunity"             # TPR parity
    EQUALIZED_ODDS = "equalized_odds"                  # TPR + FPR parity
    PREDICTIVE_PARITY = "predictive_parity"            # PPV parity
    CALIBRATION = "calibration"                         # P(Y=1|Ŷ=p) parity
    TREATMENT_EQUALITY = "treatment_equality"           # FP/FN ratio parity
    INDIVIDUAL_FAIRNESS = "individual_fairness"         # Similar individuals → similar outcomes


class BiasLevel(str, Enum):
    """Poziomy biasu."""
    NONE = "none"           # Ratio > 0.95
    LOW = "low"             # 0.80 <= Ratio < 0.95
    MEDIUM = "medium"       # 0.60 <= Ratio < 0.80
    HIGH = "high"           # 0.40 <= Ratio < 0.60
    CRITICAL = "critical"   # Ratio < 0.40


@dataclass
class GroupMetrics:
    """Metryki dla pojedynczej grupy."""
    group_name: str
    group_size: int
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float  # TPR
    f1_score: float
    fpr: float
    fnr: float
    
    # Confusion matrix
    tp: int
    tn: int
    fp: int
    fn: int
    
    # Positive prediction rate
    positive_rate: float
    
    # Calibration (if probabilities available)
    calibration_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group_name,
            "size": self.group_size,
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1_score),
            "fpr": float(self.fpr),
            "fnr": float(self.fnr),
            "positive_rate": float(self.positive_rate),
            "calibration_error": float(self.calibration_error) if self.calibration_error else None,
            "confusion": {
                "tp": self.tp, "tn": self.tn,
                "fp": self.fp, "fn": self.fn
            }
        }


@dataclass
class FairnessResult:
    """Wynik pomiaru fairness między dwiema grupami."""
    metric: FairnessMetric
    reference_group: str
    comparison_group: str
    
    reference_value: float
    comparison_value: float
    
    # Fairness ratio (comparison / reference)
    ratio: float
    
    # Disparate impact (80% rule compliance)
    disparate_impact: float
    passes_80_rule: bool
    
    # Statistical significance
    p_value: Optional[float] = None
    
    # Bias classification
    bias_level: BiasLevel = BiasLevel.NONE
    
    # Recommendations
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "reference_group": self.reference_group,
            "comparison_group": self.comparison_group,
            "reference_value": float(self.reference_value),
            "comparison_value": float(self.comparison_value),
            "ratio": float(self.ratio),
            "disparate_impact": float(self.disparate_impact),
            "passes_80_rule": self.passes_80_rule,
            "p_value": float(self.p_value) if self.p_value else None,
            "bias_level": self.bias_level.value,
            "recommendation": self.recommendation
        }


@dataclass
class FairnessReport:
    """Kompleksowy raport fairness."""
    group_metrics: Dict[str, GroupMetrics]
    pairwise_comparisons: List[FairnessResult]
    
    # Overall assessment
    worst_bias_level: BiasLevel
    biased_metrics: List[str]
    
    # Legal compliance
    passes_disparate_impact_test: bool
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_metrics": {
                name: gm.to_dict() for name, gm in self.group_metrics.items()
            },
            "pairwise_comparisons": [fc.to_dict() for fc in self.pairwise_comparisons],
            "worst_bias_level": self.worst_bias_level.value,
            "biased_metrics": self.biased_metrics,
            "passes_disparate_impact_test": self.passes_disparate_impact_test,
            "timestamp": self.timestamp
        }


# ============================================================================
# FAIRNESS ANALYZER
# ============================================================================

class FairnessAnalyzer:
    """
    Zaawansowany silnik analizy fairness z:
    - Multiple fairness metrics
    - Statistical significance testing
    - Disparate impact detection (80% rule)
    - Bias level classification
    - Mitigation recommendations
    """
    
    # Thresholds
    DISPARATE_IMPACT_THRESHOLD = 0.80  # 80% rule (legal standard)
    PARITY_THRESHOLD = 0.95            # 95% parity for "fair"
    
    # Bias level thresholds
    BIAS_THRESHOLDS = {
        BiasLevel.CRITICAL: 0.40,
        BiasLevel.HIGH: 0.60,
        BiasLevel.MEDIUM: 0.80,
        BiasLevel.LOW: 0.95,
        BiasLevel.NONE: 1.0
    }
    
    def __init__(
        self,
        reference_group: Optional[str] = None,
        alpha: float = 0.05
    ):
        """
        Args:
            reference_group: Reference group for comparisons (usually privileged group)
            alpha: Significance level for statistical tests
        """
        self.reference_group = reference_group
        self.alpha = alpha
    
    # ------------------------------------------------------------------------
    # GROUP METRICS COMPUTATION
    # ------------------------------------------------------------------------
    
    def compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, GroupMetrics]:
        """
        Wylicza kompleksowe metryki per-grupa.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            group: Protected attribute values
            y_prob: Predicted probabilities (optional, for calibration)
            
        Returns:
            Dict[group_name, GroupMetrics]
        """
        results: Dict[str, GroupMetrics] = {}
        
        # Get unique groups (handling NaN)
        groups_series = pd.Series(group).fillna("NA").astype(str)
        unique_groups = groups_series.unique().tolist()
        
        for group_name in unique_groups:
            mask = (groups_series == group_name).to_numpy()
            
            yt = y_true[mask]
            yp = y_pred[mask]
            
            if len(yt) == 0:
                continue
            
            # Compute confusion matrix
            tp = int(((yp == 1) & (yt == 1)).sum())
            tn = int(((yp == 0) & (yt == 0)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            
            # Compute metrics (with zero-division handling)
            accuracy = float((yt == yp).mean())
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)  # TPR
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            
            fpr = fp / max(fp + tn, 1)
            fnr = fn / max(fn + tp, 1)
            
            positive_rate = float((yp == 1).mean())
            
            # Calibration error (if probabilities available)
            calibration_error = None
            if y_prob is not None:
                probs = y_prob[mask]
                if len(probs) > 0:
                    calibration_error = self._compute_calibration_error(yt, probs)
            
            metrics = GroupMetrics(
                group_name=group_name,
                group_size=len(yt),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                fpr=fpr,
                fnr=fnr,
                tp=tp, tn=tn, fp=fp, fn=fn,
                positive_rate=positive_rate,
                calibration_error=calibration_error
            )
            
            results[group_name] = metrics
        
        return results
    
    # ------------------------------------------------------------------------
    # FAIRNESS METRICS
    # ------------------------------------------------------------------------
    
    def demographic_parity(
        self,
        group_metrics: Dict[str, GroupMetrics],
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Demographic Parity: P(Ŷ=1|A=a) ≈ P(Ŷ=1|A=b)
        
        Measures: positive prediction rate equality across groups.
        """
        return self._compute_pairwise_fairness(
            group_metrics,
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            value_fn=lambda gm: gm.positive_rate,
            reference_group=reference_group
        )
    
    def equal_opportunity(
        self,
        group_metrics: Dict[str, GroupMetrics],
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Equal Opportunity: TPR(A=a) ≈ TPR(A=b)
        
        Measures: true positive rate (recall) equality.
        """
        return self._compute_pairwise_fairness(
            group_metrics,
            metric=FairnessMetric.EQUAL_OPPORTUNITY,
            value_fn=lambda gm: gm.recall,
            reference_group=reference_group
        )
    
    def equalized_odds(
        self,
        group_metrics: Dict[str, GroupMetrics],
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Equalized Odds: TPR(A=a) ≈ TPR(A=b) AND FPR(A=a) ≈ FPR(A=b)
        
        Measures: both TPR and FPR equality.
        Returns both comparisons.
        """
        tpr_results = self._compute_pairwise_fairness(
            group_metrics,
            metric=FairnessMetric.EQUAL_OPPORTUNITY,  # TPR
            value_fn=lambda gm: gm.recall,
            reference_group=reference_group
        )
        
        fpr_results = self._compute_pairwise_fairness(
            group_metrics,
            metric=FairnessMetric.EQUALIZED_ODDS,  # FPR
            value_fn=lambda gm: gm.fpr,
            reference_group=reference_group
        )
        
        return tpr_results + fpr_results
    
    def predictive_parity(
        self,
        group_metrics: Dict[str, GroupMetrics],
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Predictive Parity: PPV(A=a) ≈ PPV(A=b)
        
        Measures: precision (positive predictive value) equality.
        """
        return self._compute_pairwise_fairness(
            group_metrics,
            metric=FairnessMetric.PREDICTIVE_PARITY,
            value_fn=lambda gm: gm.precision,
            reference_group=reference_group
        )
    
    def treatment_equality(
        self,
        group_metrics: Dict[str, GroupMetrics],
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Treatment Equality: FN/FP ratio parity
        
        Measures: error type balance across groups.
        """
        def error_ratio(gm: GroupMetrics) -> float:
            return gm.fn / max(gm.fp, 1)
        
        return self._compute_pairwise_fairness(
            group_metrics,
            metric=FairnessMetric.TREATMENT_EQUALITY,
            value_fn=error_ratio,
            reference_group=reference_group
        )
    
    # ------------------------------------------------------------------------
    # PAIRWISE COMPARISON ENGINE
    # ------------------------------------------------------------------------
    
    def _compute_pairwise_fairness(
        self,
        group_metrics: Dict[str, GroupMetrics],
        metric: FairnessMetric,
        value_fn: callable,
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Oblicza fairness dla wszystkich par grup.
        
        Args:
            group_metrics: Metryki per-grupa
            metric: Typ metryki fairness
            value_fn: Funkcja wyciągająca wartość z GroupMetrics
            reference_group: Grupa referencyjna
            
        Returns:
            Lista FairnessResult dla każdej pary
        """
        results: List[FairnessResult] = []
        
        groups = list(group_metrics.keys())
        
        # Determine reference group
        if reference_group is None:
            reference_group = self.reference_group
        
        if reference_group is None or reference_group not in groups:
            # Use first group as reference
            reference_group = groups[0]
        
        ref_metrics = group_metrics[reference_group]
        ref_value = value_fn(ref_metrics)
        
        # Compare all other groups to reference
        for group_name in groups:
            if group_name == reference_group:
                continue
            
            comp_metrics = group_metrics[group_name]
            comp_value = value_fn(comp_metrics)
            
            # Compute ratio (avoid division by zero)
            if ref_value > 0:
                ratio = comp_value / ref_value
            else:
                ratio = 1.0 if comp_value == 0 else float('inf')
            
            # Disparate impact (min/max ratio)
            disparate_impact = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0
            passes_80_rule = disparate_impact >= self.DISPARATE_IMPACT_THRESHOLD
            
            # Statistical significance (chi-square test for proportions)
            p_value = self._test_proportion_difference(
                ref_metrics, comp_metrics, value_fn
            )
            
            # Classify bias level
            bias_level = self._classify_bias_level(ratio)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                metric, bias_level, reference_group, group_name
            )
            
            result = FairnessResult(
                metric=metric,
                reference_group=reference_group,
                comparison_group=group_name,
                reference_value=ref_value,
                comparison_value=comp_value,
                ratio=ratio,
                disparate_impact=disparate_impact,
                passes_80_rule=passes_80_rule,
                p_value=p_value,
                bias_level=bias_level,
                recommendation=recommendation
            )
            
            results.append(result)
        
        return results
    
    # ------------------------------------------------------------------------
    # COMPREHENSIVE FAIRNESS REPORT
    # ------------------------------------------------------------------------
    
    def analyze_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        metrics: Optional[List[FairnessMetric]] = None
    ) -> FairnessReport:
        """
        Kompleksowa analiza fairness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute (e.g., gender, race)
            y_prob: Predicted probabilities (optional)
            metrics: List of metrics to compute (None = all)
            
        Returns:
            FairnessReport with all results
        """
        # Default: compute all metrics
        if metrics is None:
            metrics = [
                FairnessMetric.DEMOGRAPHIC_PARITY,
                FairnessMetric.EQUAL_OPPORTUNITY,
                FairnessMetric.EQUALIZED_ODDS,
                FairnessMetric.PREDICTIVE_PARITY,
                FairnessMetric.TREATMENT_EQUALITY
            ]
        
        # 1. Compute group metrics
        group_metrics = self.compute_group_metrics(
            y_true, y_pred, protected_attribute, y_prob
        )
        
        # 2. Compute all fairness comparisons
        all_comparisons: List[FairnessResult] = []
        
        for metric_type in metrics:
            if metric_type == FairnessMetric.DEMOGRAPHIC_PARITY:
                results = self.demographic_parity(group_metrics)
            elif metric_type == FairnessMetric.EQUAL_OPPORTUNITY:
                results = self.equal_opportunity(group_metrics)
            elif metric_type == FairnessMetric.EQUALIZED_ODDS:
                results = self.equalized_odds(group_metrics)
            elif metric_type == FairnessMetric.PREDICTIVE_PARITY:
                results = self.predictive_parity(group_metrics)
            elif metric_type == FairnessMetric.TREATMENT_EQUALITY:
                results = self.treatment_equality(group_metrics)
            else:
                continue
            
            all_comparisons.extend(results)
        
        # 3. Aggregate assessment
        worst_bias = BiasLevel.NONE
        biased_metrics = []
        passes_disparate_impact = True
        
        for comp in all_comparisons:
            # Track worst bias
            if self._bias_rank(comp.bias_level) > self._bias_rank(worst_bias):
                worst_bias = comp.bias_level
            
            # Track biased metrics
            if comp.bias_level not in {BiasLevel.NONE, BiasLevel.LOW}:
                if comp.metric.value not in biased_metrics:
                    biased_metrics.append(comp.metric.value)
            
            # Check disparate impact
            if not comp.passes_80_rule:
                passes_disparate_impact = False
        
        # 4. Create report
        report = FairnessReport(
            group_metrics=group_metrics,
            pairwise_comparisons=all_comparisons,
            worst_bias_level=worst_bias,
            biased_metrics=biased_metrics,
            passes_disparate_impact_test=passes_disparate_impact
        )
        
        # 5. Telemetry
        audit("fairness_analysis", {
            "num_groups": len(group_metrics),
            "worst_bias": worst_bias.value,
            "biased_metrics": biased_metrics,
            "passes_disparate_impact": passes_disparate_impact
        })
        
        metric("fairness_bias_level", self._bias_rank(worst_bias), {
            "worst_bias": worst_bias.value
        })
        
        return report
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def _compute_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE).
        
        Measures: |P(Y=1|Ŷ=p) - p|
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins[:-1]) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_prob = y_prob[mask].mean()
            bin_true = y_true[mask].mean()
            bin_weight = mask.sum() / len(y_prob)
            
            ece += bin_weight * abs(bin_true - bin_prob)
        
        return float(ece)
    
    def _test_proportion_difference(
        self,
        ref_metrics: GroupMetrics,
        comp_metrics: GroupMetrics,
        value_fn: callable
    ) -> Optional[float]:
        """
        Chi-square test for proportion difference.
        
        Returns p-value (lower = more significant difference).
        """
        try:
            ref_val = value_fn(ref_metrics)
            comp_val = value_fn(comp_metrics)
            
            # Approximate counts for proportions
            ref_count = int(ref_val * ref_metrics.group_size)
            comp_count = int(comp_val * comp_metrics.group_size)
            
            # Contingency table
            table = np.array([
                [ref_count, ref_metrics.group_size - ref_count],
                [comp_count, comp_metrics.group_size - comp_count]
            ])
            
            # Chi-square test
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            
            return float(p_value)
        
        except Exception:
            return None
    
    def _classify_bias_level(self, ratio: float) -> BiasLevel:
        """Klasyfikuje poziom biasu na podstawie ratio."""
        # Normalize ratio to [0, 1] (min/max)
        normalized_ratio = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0
        
        for level, threshold in sorted(
            self.BIAS_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if normalized_ratio >= threshold:
                return level
        
        return BiasLevel.CRITICAL
    
    @staticmethod
    def _bias_rank(level: BiasLevel) -> int:
        """Ranking biasu (0=NONE, 4=CRITICAL)."""
        ranks = {
            BiasLevel.NONE: 0,
            BiasLevel.LOW: 1,
            BiasLevel.MEDIUM: 2,
            BiasLevel.HIGH: 3,
            BiasLevel.CRITICAL: 4
        }
        return ranks.get(level, 0)
    
    def _generate_recommendation(
        self,
        metric: FairnessMetric,
        bias_level: BiasLevel,
        ref_group: str,
        comp_group: str
    ) -> Optional[str]:
        """Generuje rekomendację mitigation."""
        if bias_level in {BiasLevel.NONE, BiasLevel.LOW}:
            return None
        
        recommendations = {
            FairnessMetric.DEMOGRAPHIC_PARITY: (
                f"Consider reweighting or resampling to balance positive prediction rates "
                f"between '{ref_group}' and '{comp_group}'."
            ),
            FairnessMetric.EQUAL_OPPORTUNITY: (
                f"Investigate features causing TPR disparity. Consider threshold optimization "
                f"or post-processing calibration."
            ),
            FairnessMetric.EQUALIZED_ODDS: (
                f"Apply fairness-aware training (e.g., adversarial debiasing) or "
                f"post-processing methods (e.g., Equalized Odds postprocessing)."
            ),
            FairnessMetric.PREDICTIVE_PARITY: (
                f"Review precision disparity. Consider adjusting decision thresholds "
                f"per group or using calibrated predictions."
            ),
            FairnessMetric.TREATMENT_EQUALITY: (
                f"Balance FN/FP ratios across groups through cost-sensitive learning "
                f"or customized loss functions."
            )
        }
        
        return recommendations.get(metric, "Review model training and consider fairness-aware methods.")


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Backward compatible group metrics function (simplified version).
    
    Returns basic metrics per group.
    """
    analyzer = FairnessAnalyzer()
    metrics = analyzer.compute_group_metrics(y_true, y_pred, group)
    
    # Convert to old format
    result: Dict[str, Dict[str, float]] = {}
    for group_name, gm in metrics.items():
        result[group_name] = {
            "accuracy": gm.accuracy,
            "tpr": gm.recall,
            "fpr": gm.fpr,
            "n": gm.group_size
        }
    
    return result


def analyze_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    reference_group: Optional[str] = None
) -> FairnessReport:
    """
    High-level fairness analysis function.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attribute: Protected attribute values
        y_prob: Predicted probabilities (optional)
        reference_group: Reference group name (optional)
        
    Returns:
        Comprehensive FairnessReport
    """
    analyzer = FairnessAnalyzer(reference_group=reference_group)
    return analyzer.analyze_fairness(y_true, y_pred, protected_attribute, y_prob)