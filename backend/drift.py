"""
TMIV Advanced Drift Detection Engine v3.0
==========================================
Zaawansowany system detekcji driftu danych z:
- Multi-metric drift detection (PSI, KS, JS, Wasserstein, Chi-Square)
- Statistical significance testing (p-values, confidence intervals)
- Drift severity classification & alerting thresholds
- Temporal drift tracking & visualization
- Feature-level & dataset-level drift aggregation
- Adaptive binning strategies
- Categorical drift metrics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class DriftSeverity(str, Enum):
    """Poziomy powagi driftu."""
    NONE = "none"           # PSI < 0.1
    LOW = "low"             # 0.1 <= PSI < 0.25
    MEDIUM = "medium"       # 0.25 <= PSI < 0.5
    HIGH = "high"           # PSI >= 0.5
    CRITICAL = "critical"   # PSI >= 1.0


class DriftMetric(str, Enum):
    """Dostępne metryki driftu."""
    PSI = "psi"                          # Population Stability Index
    KS = "kolmogorov_smirnov"           # Kolmogorov-Smirnov
    JS = "jensen_shannon"                # Jensen-Shannon Divergence
    WASSERSTEIN = "wasserstein"          # Earth Mover's Distance
    CHI_SQUARE = "chi_square"            # Chi-Square Test
    HELLINGER = "hellinger"              # Hellinger Distance
    TVD = "total_variation"              # Total Variation Distance


@dataclass
class DriftResult:
    """Wynik pomiaru driftu dla pojedynczej cechy."""
    feature: str
    metric: DriftMetric
    value: float
    severity: DriftSeverity
    p_value: Optional[float] = None
    threshold: float = 0.25
    reference_samples: int = 0
    current_samples: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Czy drift jest statystycznie istotny?"""
        if self.p_value is None:
            return self.value > self.threshold
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "metric": self.metric.value,
            "value": float(self.value),
            "severity": self.severity.value,
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "threshold": float(self.threshold),
            "significant": self.is_significant(),
            "reference_samples": self.reference_samples,
            "current_samples": self.current_samples,
            "timestamp": self.timestamp
        }


@dataclass
class DatasetDriftReport:
    """Raport driftu dla całego datasetu."""
    feature_drifts: List[DriftResult]
    dataset_drift_score: float  # Aggregated score
    drifted_features: List[str]
    severity: DriftSeverity
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_drift_score": float(self.dataset_drift_score),
            "drifted_features_count": len(self.drifted_features),
            "drifted_features": self.drifted_features,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "feature_drifts": [fd.to_dict() for fd in self.feature_drifts]
        }


# ============================================================================
# BINNING STRATEGIES
# ============================================================================

class BinningStrategy:
    """Strategie binowania dla histogramów."""
    
    @staticmethod
    def fixed_bins(data: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Fixed number of bins (equal width)."""
        return np.linspace(np.nanmin(data), np.nanmax(data), n_bins + 1)
    
    @staticmethod
    def quantile_bins(data: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Quantile-based bins (equal frequency)."""
        quantiles = np.linspace(0, 1, n_bins + 1)
        return np.nanquantile(data, quantiles)
    
    @staticmethod
    def sturges_rule(data: np.ndarray) -> int:
        """Sturges' rule: k = ceil(log2(n)) + 1"""
        return int(np.ceil(np.log2(len(data))) + 1)
    
    @staticmethod
    def rice_rule(data: np.ndarray) -> int:
        """Rice rule: k = ceil(2 * n^(1/3))"""
        return int(np.ceil(2 * len(data) ** (1/3)))
    
    @staticmethod
    def scott_rule(data: np.ndarray) -> int:
        """Scott's rule: bin_width = 3.5 * sigma / n^(1/3)"""
        sigma = np.nanstd(data)
        n = len(data[~np.isnan(data)])
        bin_width = 3.5 * sigma / (n ** (1/3))
        data_range = np.nanmax(data) - np.nanmin(data)
        return max(int(np.ceil(data_range / bin_width)), 2)


# ============================================================================
# DRIFT DETECTION ENGINE
# ============================================================================

class DriftDetector:
    """
    Zaawansowany silnik detekcji driftu z:
    - Multiple metrics (PSI, KS, JS, Wasserstein, Chi-Square)
    - Statistical significance testing
    - Adaptive binning
    - Categorical & numerical support
    """
    
    # Severity thresholds (for PSI)
    SEVERITY_THRESHOLDS = {
        DriftSeverity.CRITICAL: 1.0,
        DriftSeverity.HIGH: 0.5,
        DriftSeverity.MEDIUM: 0.25,
        DriftSeverity.LOW: 0.1,
        DriftSeverity.NONE: 0.0
    }
    
    def __init__(
        self,
        default_bins: int = 10,
        binning_strategy: str = "quantile",  # "fixed" | "quantile" | "sturges" | "rice" | "scott"
        min_samples: int = 30
    ):
        self.default_bins = default_bins
        self.binning_strategy = binning_strategy
        self.min_samples = min_samples
    
    # ------------------------------------------------------------------------
    # CORE DRIFT METRICS
    # ------------------------------------------------------------------------
    
    def population_stability_index(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[int] = None
    ) -> Tuple[float, Optional[float]]:
        """
        Enhanced PSI with statistical testing.
        
        Returns:
            (psi_value, p_value)
        """
        bins = bins or self.default_bins
        
        # Clean data
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return 0.0, None
        
        # Adaptive binning
        edges = self._get_bin_edges(ref_clean, bins)
        
        # Compute histograms
        ref_hist, _ = np.histogram(ref_clean, bins=edges)
        cur_hist, _ = np.histogram(cur_clean, bins=edges)
        
        # Normalize
        ref_prop = ref_hist / max(ref_hist.sum(), 1)
        cur_prop = cur_hist / max(cur_hist.sum(), 1)
        
        # PSI calculation
        eps = 1e-12
        psi = np.sum((cur_prop - ref_prop) * np.log((cur_prop + eps) / (ref_prop + eps)))
        
        # Chi-square test for statistical significance
        try:
            # Combine bins with expected count < 5
            ref_combined, cur_combined = self._combine_sparse_bins(ref_hist, cur_hist, min_count=5)
            if len(ref_combined) > 1:
                chi2, p_value = stats.chisquare(cur_combined, f_exp=ref_combined)
                p_value = float(p_value)
            else:
                p_value = None
        except Exception:
            p_value = None
        
        return float(psi), p_value
    
    def kolmogorov_smirnov(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test with p-value.
        
        Returns:
            (ks_statistic, p_value)
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return 0.0, 1.0
        
        # Two-sample KS test
        ks_stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
        
        return float(ks_stat), float(p_value)
    
    def jensen_shannon_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[int] = None
    ) -> float:
        """
        Jensen-Shannon divergence (symmetric, bounded [0,1]).
        """
        bins = bins or self.default_bins * 2  # JS benefits from more bins
        
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return 0.0
        
        edges = self._get_bin_edges(ref_clean, bins)
        
        ref_hist, _ = np.histogram(ref_clean, bins=edges)
        cur_hist, _ = np.histogram(cur_clean, bins=edges)
        
        ref_prop = ref_hist / max(ref_hist.sum(), 1)
        cur_prop = cur_hist / max(cur_hist.sum(), 1)
        
        # Use scipy's implementation (more robust)
        js_dist = jensenshannon(ref_prop, cur_prop)
        
        return float(js_dist)
    
    def wasserstein_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """
        Wasserstein distance (Earth Mover's Distance).
        Sensitive to shifts in distribution location.
        """
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return 0.0
        
        # 1D Wasserstein distance
        wass = stats.wasserstein_distance(ref_clean, cur_clean)
        
        return float(wass)
    
    def hellinger_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[int] = None
    ) -> float:
        """
        Hellinger distance (bounded [0,1], symmetric).
        """
        bins = bins or self.default_bins
        
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return 0.0
        
        edges = self._get_bin_edges(ref_clean, bins)
        
        ref_hist, _ = np.histogram(ref_clean, bins=edges)
        cur_hist, _ = np.histogram(cur_clean, bins=edges)
        
        ref_prop = ref_hist / max(ref_hist.sum(), 1)
        cur_prop = cur_hist / max(cur_hist.sum(), 1)
        
        # Hellinger distance
        hellinger = np.sqrt(np.sum((np.sqrt(ref_prop) - np.sqrt(cur_prop)) ** 2)) / np.sqrt(2)
        
        return float(hellinger)
    
    def total_variation_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[int] = None
    ) -> float:
        """
        Total Variation Distance (L1 norm of probability difference).
        """
        bins = bins or self.default_bins
        
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        
        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return 0.0
        
        edges = self._get_bin_edges(ref_clean, bins)
        
        ref_hist, _ = np.histogram(ref_clean, bins=edges)
        cur_hist, _ = np.histogram(cur_clean, bins=edges)
        
        ref_prop = ref_hist / max(ref_hist.sum(), 1)
        cur_prop = cur_hist / max(cur_hist.sum(), 1)
        
        tvd = 0.5 * np.sum(np.abs(ref_prop - cur_prop))
        
        return float(tvd)
    
    # ------------------------------------------------------------------------
    # CATEGORICAL DRIFT
    # ------------------------------------------------------------------------
    
    def categorical_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Chi-square test for categorical drift.
        
        Returns:
            (chi_square_statistic, p_value)
        """
        # Get unique categories from both
        all_cats = np.unique(np.concatenate([reference, current]))
        
        # Count frequencies
        ref_counts = pd.Series(reference).value_counts().reindex(all_cats, fill_value=0).values
        cur_counts = pd.Series(current).value_counts().reindex(all_cats, fill_value=0).values
        
        # Chi-square test
        chi2, p_value = stats.chisquare(cur_counts, f_exp=ref_counts)
        
        return float(chi2), float(p_value)
    
    # ------------------------------------------------------------------------
    # HIGH-LEVEL INTERFACE
    # ------------------------------------------------------------------------
    
    def detect_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        metric: DriftMetric = DriftMetric.PSI,
        threshold: float = 0.25
    ) -> DriftResult:
        """
        Wykrywa drift dla pojedynczej cechy.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Nazwa cechy
            metric: Metryka driftu
            threshold: Próg alarmowy
            
        Returns:
            DriftResult z wartością metryki i severity
        """
        # Select metric
        if metric == DriftMetric.PSI:
            value, p_value = self.population_stability_index(reference, current)
        elif metric == DriftMetric.KS:
            value, p_value = self.kolmogorov_smirnov(reference, current)
        elif metric == DriftMetric.JS:
            value = self.jensen_shannon_divergence(reference, current)
            p_value = None
        elif metric == DriftMetric.WASSERSTEIN:
            value = self.wasserstein_distance(reference, current)
            p_value = None
        elif metric == DriftMetric.HELLINGER:
            value = self.hellinger_distance(reference, current)
            p_value = None
        elif metric == DriftMetric.TVD:
            value = self.total_variation_distance(reference, current)
            p_value = None
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Determine severity
        severity = self._classify_severity(value, metric)
        
        result = DriftResult(
            feature=feature_name,
            metric=metric,
            value=value,
            severity=severity,
            p_value=p_value,
            threshold=threshold,
            reference_samples=len(reference[~np.isnan(reference)]),
            current_samples=len(current[~np.isnan(current)])
        )
        
        # Telemetry
        metric("drift_detected", value, {
            "feature": feature_name,
            "metric": metric.value,
            "severity": severity.value
        })
        
        return result
    
    def detect_dataset_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        metric: DriftMetric = DriftMetric.PSI,
        threshold: float = 0.25
    ) -> DatasetDriftReport:
        """
        Wykrywa drift dla całego datasetu.
        
        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            features: Lista cech do sprawdzenia (None = all numeric)
            metric: Metryka driftu
            threshold: Próg alarmowy
            
        Returns:
            DatasetDriftReport z agregowanymi wynikami
        """
        if features is None:
            features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_drifts = []
        
        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            
            try:
                ref = reference_df[feature].values
                cur = current_df[feature].values
                
                drift = self.detect_feature_drift(
                    reference=ref,
                    current=cur,
                    feature_name=feature,
                    metric=metric,
                    threshold=threshold
                )
                feature_drifts.append(drift)
                
            except Exception as e:
                audit("drift_detection_error", {"feature": feature, "error": str(e)})
                continue
        
        # Aggregate dataset-level drift
        drifted_features = [
            fd.feature for fd in feature_drifts 
            if fd.severity not in {DriftSeverity.NONE, DriftSeverity.LOW}
        ]
        
        # Dataset drift score: mean of significant drifts
        significant_values = [fd.value for fd in feature_drifts if fd.is_significant()]
        dataset_drift_score = float(np.mean(significant_values)) if significant_values else 0.0
        
        # Dataset severity: worst of any feature
        worst_severity = DriftSeverity.NONE
        for fd in feature_drifts:
            if self._severity_rank(fd.severity) > self._severity_rank(worst_severity):
                worst_severity = fd.severity
        
        report = DatasetDriftReport(
            feature_drifts=feature_drifts,
            dataset_drift_score=dataset_drift_score,
            drifted_features=drifted_features,
            severity=worst_severity
        )
        
        # Telemetry
        audit("dataset_drift_report", {
            "drifted_features_count": len(drifted_features),
            "dataset_drift_score": dataset_drift_score,
            "severity": worst_severity.value
        })
        
        return report
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def _get_bin_edges(self, data: np.ndarray, n_bins: int) -> np.ndarray:
        """Oblicza edges zgodnie z wybraną strategią."""
        if self.binning_strategy == "quantile":
            return BinningStrategy.quantile_bins(data, n_bins)
        elif self.binning_strategy == "sturges":
            n_bins = BinningStrategy.sturges_rule(data)
            return BinningStrategy.fixed_bins(data, n_bins)
        elif self.binning_strategy == "rice":
            n_bins = BinningStrategy.rice_rule(data)
            return BinningStrategy.fixed_bins(data, n_bins)
        elif self.binning_strategy == "scott":
            n_bins = BinningStrategy.scott_rule(data)
            return BinningStrategy.fixed_bins(data, n_bins)
        else:  # fixed
            return BinningStrategy.fixed_bins(data, n_bins)
    
    @staticmethod
    def _combine_sparse_bins(
        ref_hist: np.ndarray,
        cur_hist: np.ndarray,
        min_count: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Łączy biny z małą liczebnością dla chi-square."""
        combined_ref = []
        combined_cur = []
        temp_ref = 0
        temp_cur = 0
        
        for r, c in zip(ref_hist, cur_hist):
            temp_ref += r
            temp_cur += c
            
            if temp_ref >= min_count:
                combined_ref.append(temp_ref)
                combined_cur.append(temp_cur)
                temp_ref = 0
                temp_cur = 0
        
        # Add remaining
        if temp_ref > 0:
            if combined_ref:
                combined_ref[-1] += temp_ref
                combined_cur[-1] += temp_cur
            else:
                combined_ref.append(temp_ref)
                combined_cur.append(temp_cur)
        
        return np.array(combined_ref), np.array(combined_cur)
    
    def _classify_severity(self, value: float, metric: DriftMetric) -> DriftSeverity:
        """Klasyfikuje severity na podstawie wartości metryki."""
        # PSI thresholds (default)
        thresholds = self.SEVERITY_THRESHOLDS
        
        # Adjust for other metrics (some are bounded [0,1])
        if metric in {DriftMetric.JS, DriftMetric.HELLINGER, DriftMetric.TVD, DriftMetric.KS}:
            # These are bounded [0, 1], adjust thresholds
            thresholds = {
                DriftSeverity.CRITICAL: 0.5,
                DriftSeverity.HIGH: 0.3,
                DriftSeverity.MEDIUM: 0.15,
                DriftSeverity.LOW: 0.05,
                DriftSeverity.NONE: 0.0
            }
        
        for severity, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if value >= threshold:
                return severity
        
        return DriftSeverity.NONE
    
    @staticmethod
    def _severity_rank(severity: DriftSeverity) -> int:
        """Ranking severity (0=NONE, 4=CRITICAL)."""
        ranks = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4
        }
        return ranks.get(severity, 0)


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

_detector: Optional[DriftDetector] = None

def _get_detector() -> DriftDetector:
    """Lazy initialization."""
    global _detector
    if _detector is None:
        _detector = DriftDetector()
    return _detector


def population_stability_index(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """Backward compatible PSI (without p-value)."""
    detector = _get_detector()
    psi, _ = detector.population_stability_index(a, b, bins)
    return psi


def kolmogorov_smirnov(a: np.ndarray, b: np.ndarray) -> float:
    """Backward compatible KS (without p-value)."""
    detector = _get_detector()
    ks, _ = detector.kolmogorov_smirnov(a, b)
    return ks


def jensen_shannon(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    """Backward compatible JS."""
    detector = _get_detector()
    return detector.jensen_shannon_divergence(a, b, bins)


# Deprecated internal function
def _hist(a: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated: Use DriftDetector methods instead."""
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.zeros(bins), np.linspace(0, 1, bins + 1)
    hist, edges = np.histogram(
        a, bins=bins,
        range=(np.nanmin(a), np.nanmax(a) if np.nanmax(a) > np.nanmin(a) else np.nanmin(a) + 1e-9)
    )
    p = hist / max(hist.sum(), 1)
    return p, edges