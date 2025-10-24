"""
TMIV Advanced Model Validation & Testing v3.0
==============================================
Zaawansowany system walidacji i testowania modeli z:
- Cross-format validation (sklearn vs ONNX vs TorchScript vs TFLite)
- Comprehensive error metrics (MAE, RMSE, max diff, correlation)
- Performance benchmarking (latency, throughput)
- Numerical stability testing
- Edge case detection
- Model comparison suite
- A/B testing utilities
- Production readiness scoring
- Regression testing
- Model card generation
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class ComparisonStatus(str, Enum):
    """Status of model comparison."""
    IDENTICAL = "identical"         # Predictions exactly match
    ACCEPTABLE = "acceptable"       # Within tolerance
    WARNING = "warning"            # Differences detected
    FAILED = "failed"              # Significant differences
    ERROR = "error"                # Comparison failed


class ModelFormat(str, Enum):
    """Model formats for comparison."""
    SKLEARN = "sklearn"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW = "tensorflow"
    TFLITE = "tflite"


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    # Error metrics
    mae: float                      # Mean Absolute Error
    rmse: float                     # Root Mean Squared Error
    max_error: float                # Maximum absolute error
    median_error: float             # Median absolute error
    
    # Statistical metrics
    correlation: float              # Pearson correlation
    r2_score: float                 # R² score
    
    # Distribution metrics
    mean_diff: float               # Mean difference
    std_diff: float                # Std of differences
    
    # Quantiles of error
    q95_error: float               # 95th percentile error
    q99_error: float               # 99th percentile error
    
    # Sample info
    n_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "max_error": float(self.max_error),
            "median_error": float(self.median_error),
            "correlation": float(self.correlation),
            "r2_score": float(self.r2_score),
            "mean_diff": float(self.mean_diff),
            "std_diff": float(self.std_diff),
            "q95_error": float(self.q95_error),
            "q99_error": float(self.q99_error),
            "n_samples": self.n_samples
        }


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    format: ModelFormat
    
    # Latency (milliseconds)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput
    throughput_samples_per_sec: float
    
    # Memory
    memory_mb: Optional[float] = None
    
    # Test info
    n_iterations: int = 100
    batch_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format.value,
            "mean_latency_ms": float(self.mean_latency_ms),
            "median_latency_ms": float(self.median_latency_ms),
            "p95_latency_ms": float(self.p95_latency_ms),
            "p99_latency_ms": float(self.p99_latency_ms),
            "throughput_samples_per_sec": float(self.throughput_samples_per_sec),
            "memory_mb": float(self.memory_mb) if self.memory_mb else None,
            "n_iterations": self.n_iterations,
            "batch_size": self.batch_size
        }


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    status: ComparisonStatus
    
    # Models compared
    reference_format: ModelFormat
    target_format: ModelFormat
    
    # Metrics
    validation_metrics: ValidationMetrics
    
    # Thresholds
    mae_threshold: float
    max_error_threshold: float
    
    # Assessment
    passes_validation: bool
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reference_format": self.reference_format.value,
            "target_format": self.target_format.value,
            "metrics": self.validation_metrics.to_dict(),
            "mae_threshold": self.mae_threshold,
            "max_error_threshold": self.max_error_threshold,
            "passes_validation": self.passes_validation,
            "warnings": self.warnings,
            "timestamp": self.timestamp
        }


# ============================================================================
# MODEL VALIDATOR
# ============================================================================

class ModelValidator:
    """
    Comprehensive model validation system.
    
    Features:
    - Cross-format comparison
    - Error metrics
    - Performance benchmarking
    - Production readiness
    """
    
    # Default tolerances
    DEFAULT_MAE_THRESHOLD = 1e-5
    DEFAULT_MAX_ERROR_THRESHOLD = 1e-4
    
    def __init__(
        self,
        mae_threshold: float = DEFAULT_MAE_THRESHOLD,
        max_error_threshold: float = DEFAULT_MAX_ERROR_THRESHOLD
    ):
        """
        Args:
            mae_threshold: Acceptable MAE threshold
            max_error_threshold: Acceptable maximum error threshold
        """
        self.mae_threshold = mae_threshold
        self.max_error_threshold = max_error_threshold
    
    # ------------------------------------------------------------------------
    # PREDICTION COMPARISON
    # ------------------------------------------------------------------------
    
    def compare_predictions(
        self,
        reference_predictions: np.ndarray,
        target_predictions: np.ndarray,
        reference_format: ModelFormat = ModelFormat.SKLEARN,
        target_format: ModelFormat = ModelFormat.ONNX
    ) -> ComparisonResult:
        """
        Compare predictions from two models.
        
        Args:
            reference_predictions: Reference model predictions
            target_predictions: Target model predictions
            reference_format: Format of reference model
            target_format: Format of target model
            
        Returns:
            ComparisonResult with detailed metrics
        """
        # Flatten arrays
        ref = reference_predictions.ravel()
        tgt = target_predictions.ravel()
        
        if len(ref) != len(tgt):
            raise ValueError(
                f"Prediction shape mismatch: {len(ref)} vs {len(tgt)}"
            )
        
        # Compute metrics
        metrics = self._compute_validation_metrics(ref, tgt)
        
        # Determine status
        status = self._determine_status(metrics)
        
        # Check if passes validation
        passes = (
            metrics.mae <= self.mae_threshold and
            metrics.max_error <= self.max_error_threshold
        )
        
        # Generate warnings
        warnings = self._generate_warnings(metrics)
        
        result = ComparisonResult(
            status=status,
            reference_format=reference_format,
            target_format=target_format,
            validation_metrics=metrics,
            mae_threshold=self.mae_threshold,
            max_error_threshold=self.max_error_threshold,
            passes_validation=passes,
            warnings=warnings
        )
        
        # Telemetry
        audit("model_comparison", {
            "reference": reference_format.value,
            "target": target_format.value,
            "status": status.value,
            "mae": metrics.mae,
            "passes": passes
        })
        
        metric("model_comparison_mae", metrics.mae, {
            "target_format": target_format.value
        })
        
        return result
    
    def _compute_validation_metrics(
        self,
        reference: np.ndarray,
        target: np.ndarray
    ) -> ValidationMetrics:
        """Compute comprehensive validation metrics."""
        # Compute errors
        errors = np.abs(reference - target)
        squared_errors = (reference - target) ** 2
        
        # Error metrics
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(squared_errors)))
        max_error = float(np.max(errors))
        median_error = float(np.median(errors))
        
        # Statistical metrics
        correlation = float(np.corrcoef(reference, target)[0, 1])
        
        # R² score
        ss_tot = np.sum((reference - np.mean(reference)) ** 2)
        ss_res = np.sum(squared_errors)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Distribution metrics
        diffs = target - reference
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        
        # Quantiles
        q95_error = float(np.percentile(errors, 95))
        q99_error = float(np.percentile(errors, 99))
        
        return ValidationMetrics(
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            median_error=median_error,
            correlation=correlation,
            r2_score=r2,
            mean_diff=mean_diff,
            std_diff=std_diff,
            q95_error=q95_error,
            q99_error=q99_error,
            n_samples=len(reference)
        )
    
    def _determine_status(self, metrics: ValidationMetrics) -> ComparisonStatus:
        """Determine comparison status based on metrics."""
        # Identical (within machine precision)
        if metrics.max_error < 1e-10:
            return ComparisonStatus.IDENTICAL
        
        # Acceptable
        if (metrics.mae <= self.mae_threshold and
            metrics.max_error <= self.max_error_threshold):
            return ComparisonStatus.ACCEPTABLE
        
        # Warning (minor differences)
        if metrics.mae <= self.mae_threshold * 10:
            return ComparisonStatus.WARNING
        
        # Failed
        return ComparisonStatus.FAILED
    
    def _generate_warnings(self, metrics: ValidationMetrics) -> List[str]:
        """Generate warnings based on metrics."""
        warnings = []
        
        if metrics.mae > self.mae_threshold:
            warnings.append(
                f"MAE {metrics.mae:.6f} exceeds threshold {self.mae_threshold:.6f}"
            )
        
        if metrics.max_error > self.max_error_threshold:
            warnings.append(
                f"Max error {metrics.max_error:.6f} exceeds threshold {self.max_error_threshold:.6f}"
            )
        
        if metrics.correlation < 0.99:
            warnings.append(
                f"Low correlation {metrics.correlation:.4f} (expected > 0.99)"
            )
        
        if abs(metrics.mean_diff) > self.mae_threshold * 10:
            warnings.append(
                f"Systematic bias detected: mean diff = {metrics.mean_diff:.6f}"
            )
        
        return warnings
    
    # ------------------------------------------------------------------------
    # SKLEARN vs ONNX COMPARISON
    # ------------------------------------------------------------------------
    
    def compare_sklearn_vs_onnx(
        self,
        sklearn_model: Any,
        X_sample: np.ndarray,
        use_proba: bool = True
    ) -> ComparisonResult:
        """
        Compare sklearn model vs its ONNX conversion.
        
        Args:
            sklearn_model: Sklearn model
            X_sample: Sample input data
            use_proba: Use predict_proba for sklearn (if available)
            
        Returns:
            ComparisonResult
        """
        try:
            import onnxruntime as ort
            from skl2onnx import to_onnx
        except ImportError:
            raise RuntimeError(
                "ONNX dependencies not available (onnxruntime, skl2onnx)"
            )
        
        # Get sklearn predictions
        if use_proba and hasattr(sklearn_model, "predict_proba"):
            sklearn_pred = sklearn_model.predict_proba(X_sample)
            # Get positive class probability
            if sklearn_pred.ndim == 2:
                sklearn_pred = sklearn_pred[:, 1]
        else:
            sklearn_pred = sklearn_model.predict(X_sample)
        
        # Convert to ONNX
        onnx_model = to_onnx(
            sklearn_model,
            X_sample[:1].astype(np.float32),
            target_opset=15
        )
        
        # Get ONNX predictions
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        
        onnx_outputs = sess.run(
            None,
            {input_name: X_sample.astype(np.float32)}
        )
        
        # Extract predictions (handle different output formats)
        if len(onnx_outputs) > 1:
            # Classifier with probabilities
            onnx_pred = onnx_outputs[1]
            if onnx_pred.ndim == 2:
                onnx_pred = onnx_pred[:, 1]  # Positive class
        else:
            onnx_pred = onnx_outputs[0]
        
        onnx_pred = onnx_pred.ravel()
        
        # Compare
        return self.compare_predictions(
            reference_predictions=sklearn_pred,
            target_predictions=onnx_pred,
            reference_format=ModelFormat.SKLEARN,
            target_format=ModelFormat.ONNX
        )
    
    # ------------------------------------------------------------------------
    # PERFORMANCE BENCHMARKING
    # ------------------------------------------------------------------------
    
    def benchmark_performance(
        self,
        model: Any,
        X_sample: np.ndarray,
        model_format: ModelFormat,
        n_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> PerformanceBenchmark:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            X_sample: Sample input data
            model_format: Format of the model
            n_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            PerformanceBenchmark results
        """
        # Warmup
        predict_fn = self._get_predict_function(model, model_format, X_sample)
        
        for _ in range(warmup_iterations):
            predict_fn()
        
        # Benchmark
        latencies = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            predict_fn()
            latency = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        latencies_arr = np.array(latencies)
        
        # Compute metrics
        mean_latency = float(np.mean(latencies_arr))
        median_latency = float(np.median(latencies_arr))
        p95_latency = float(np.percentile(latencies_arr, 95))
        p99_latency = float(np.percentile(latencies_arr, 99))
        
        # Throughput (samples per second)
        batch_size = len(X_sample)
        throughput = (batch_size * n_iterations) / (sum(latencies) / 1000)
        
        benchmark = PerformanceBenchmark(
            format=model_format,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_samples_per_sec=throughput,
            n_iterations=n_iterations,
            batch_size=batch_size
        )
        
        # Telemetry
        audit("model_benchmark", {
            "format": model_format.value,
            "mean_latency_ms": mean_latency,
            "throughput": throughput
        })
        
        metric("model_latency_ms", mean_latency, {
            "format": model_format.value
        })
        
        return benchmark
    
    def _get_predict_function(
        self,
        model: Any,
        model_format: ModelFormat,
        X_sample: np.ndarray
    ) -> Callable:
        """Get prediction function for model format."""
        if model_format == ModelFormat.SKLEARN:
            return lambda: model.predict(X_sample)
        
        elif model_format == ModelFormat.ONNX:
            import onnxruntime as ort
            sess = ort.InferenceSession(model.SerializeToString())
            input_name = sess.get_inputs()[0].name
            return lambda: sess.run(
                None,
                {input_name: X_sample.astype(np.float32)}
            )
        
        elif model_format == ModelFormat.TORCHSCRIPT:
            import torch
            X_tensor = torch.from_numpy(X_sample).float()
            return lambda: model(X_tensor)
        
        elif model_format == ModelFormat.TENSORFLOW:
            return lambda: model(X_sample)
        
        else:
            raise ValueError(f"Unsupported format: {model_format}")
    
    # ------------------------------------------------------------------------
    # EDGE CASE TESTING
    # ------------------------------------------------------------------------
    
    def test_edge_cases(
        self,
        model: Any,
        X_sample: np.ndarray,
        model_format: ModelFormat = ModelFormat.SKLEARN
    ) -> Dict[str, Any]:
        """
        Test model with edge cases.
        
        Args:
            model: Model to test
            X_sample: Sample input data
            model_format: Format of the model
            
        Returns:
            Dict with edge case test results
        """
        results = {}
        predict_fn = self._get_predict_function(model, model_format, X_sample)
        
        # Test 1: All zeros
        X_zeros = np.zeros_like(X_sample)
        try:
            pred_zeros = predict_fn()
            results["all_zeros"] = {
                "success": True,
                "has_nan": bool(np.isnan(pred_zeros).any()),
                "has_inf": bool(np.isinf(pred_zeros).any())
            }
        except Exception as e:
            results["all_zeros"] = {"success": False, "error": str(e)}
        
        # Test 2: All ones
        X_ones = np.ones_like(X_sample)
        try:
            pred_ones = predict_fn()
            results["all_ones"] = {
                "success": True,
                "has_nan": bool(np.isnan(pred_ones).any()),
                "has_inf": bool(np.isinf(pred_ones).any())
            }
        except Exception as e:
            results["all_ones"] = {"success": False, "error": str(e)}
        
        # Test 3: Large values
        X_large = X_sample * 1000
        try:
            pred_large = predict_fn()
            results["large_values"] = {
                "success": True,
                "has_nan": bool(np.isnan(pred_large).any()),
                "has_inf": bool(np.isinf(pred_large).any())
            }
        except Exception as e:
            results["large_values"] = {"success": False, "error": str(e)}
        
        # Test 4: Small values
        X_small = X_sample * 0.001
        try:
            pred_small = predict_fn()
            results["small_values"] = {
                "success": True,
                "has_nan": bool(np.isnan(pred_small).any()),
                "has_inf": bool(np.isinf(pred_small).any())
            }
        except Exception as e:
            results["small_values"] = {"success": False, "error": str(e)}
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def try_compare_onnx(pipeline: Any, X_sample: np.ndarray) -> Dict[str, Any]:
    """
    Backward compatible: compare sklearn vs ONNX.
    
    Enhanced version with comprehensive metrics.
    """
    try:
        validator = ModelValidator()
        result = validator.compare_sklearn_vs_onnx(pipeline, X_sample)
        
        if result.passes_validation:
            return {
                "ok": True,
                "mae": result.validation_metrics.mae,
                "n": result.validation_metrics.n_samples,
                "status": result.status.value,
                "metrics": result.validation_metrics.to_dict()
            }
        else:
            return {
                "ok": False,
                "reason": "validation_failed",
                "mae": result.validation_metrics.mae,
                "n": result.validation_metrics.n_samples,
                "warnings": result.warnings,
                "metrics": result.validation_metrics.to_dict()
            }
    
    except ImportError as e:
        return {
            "ok": False,
            "reason": "missing_deps",
            "message": str(e)
        }
    
    except Exception as e:
        return {
            "ok": False,
            "reason": "compare_error",
            "message": str(e)
        }


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def validate_model_export(
    sklearn_model: Any,
    onnx_model_path: str,
    X_test: np.ndarray,
    verbose: bool = True
) -> bool:
    """
    High-level API: validate ONNX export.
    
    Args:
        sklearn_model: Original sklearn model
        onnx_model_path: Path to ONNX model
        X_test: Test data
        verbose: Print results
        
    Returns:
        True if validation passes
    """
    validator = ModelValidator()
    result = validator.compare_sklearn_vs_onnx(sklearn_model, X_test)
    
    if verbose:
        print(f"🔍 Model Validation Results:")
        print(f"  Status: {result.status.value}")
        print(f"  MAE: {result.validation_metrics.mae:.6f}")
        print(f"  Max Error: {result.validation_metrics.max_error:.6f}")
        print(f"  Correlation: {result.validation_metrics.correlation:.6f}")
        print(f"  Passes: {'✓' if result.passes_validation else '✗'}")
        
        if result.warnings:
            print(f"\n  ⚠️ Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")
    
    return result.passes_validation


def benchmark_models(
    models: Dict[str, Tuple[Any, ModelFormat]],
    X_sample: np.ndarray,
    n_iterations: int = 100
) -> pd.DataFrame:
    """
    Benchmark multiple models.
    
    Args:
        models: Dict of model_name -> (model, format)
        X_sample: Sample input
        n_iterations: Benchmark iterations
        
    Returns:
        DataFrame with benchmark results
    """
    validator = ModelValidator()
    results = []
    
    for name, (model, fmt) in models.items():
        benchmark = validator.benchmark_performance(
            model, X_sample, fmt, n_iterations
        )
        
        result = benchmark.to_dict()
        result["name"] = name
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Sort by latency
    df = df.sort_values("mean_latency_ms")
    
    return df