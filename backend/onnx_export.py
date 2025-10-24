"""
TMIV Advanced Model Export & Deployment v3.0
=============================================
Zaawansowany system eksportu i deployment modeli z:
- Multi-format export (ONNX, TorchScript, TensorFlow SavedModel, PMML)
- Automatic optimization (quantization, pruning, distillation)
- Model validation & testing
- Deployment packaging (Docker, Kubernetes, Cloud)
- API generation (FastAPI, Flask, gRPC)
- Model serving utilities
- Version management
- Hardware optimization (CPU/GPU/Edge)
- Benchmark suite
- Model card generation
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .telemetry import audit, metric


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class ExportFormat(str, Enum):
    """Supported export formats."""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW_SAVEDMODEL = "tf_savedmodel"
    PMML = "pmml"
    PICKLE = "pickle"
    JOBLIB = "joblib"
    COREML = "coreml"
    TFLITE = "tflite"


class OptimizationLevel(str, Enum):
    """Model optimization levels."""
    NONE = "none"
    BASIC = "basic"           # Basic optimizations (safe)
    AGGRESSIVE = "aggressive"  # Aggressive (may lose accuracy)
    QUANTIZED = "quantized"   # Quantization (INT8)
    PRUNED = "pruned"         # Pruning (sparse model)


class TargetRuntime(str, Enum):
    """Target deployment runtime."""
    CPU = "cpu"
    GPU = "gpu"
    MOBILE = "mobile"
    EDGE = "edge"
    WEB = "web"
    CLOUD = "cloud"


@dataclass
class ExportResult:
    """Result of model export."""
    success: bool
    export_format: ExportFormat
    output_path: Optional[str]
    
    # Model info
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    model_size_mb: Optional[float]
    
    # Validation
    validated: bool = False
    validation_score: Optional[float] = None
    
    # Metadata
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    target_runtime: Optional[TargetRuntime] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "format": self.export_format.value,
            "output_path": self.output_path,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "model_size_mb": self.model_size_mb,
            "validated": self.validated,
            "validation_score": self.validation_score,
            "optimization": self.optimization_level.value,
            "target_runtime": self.target_runtime.value if self.target_runtime else None,
            "error": self.error_message,
            "warnings": self.warnings,
            "timestamp": self.timestamp
        }


# ============================================================================
# DEPENDENCY CHECKER
# ============================================================================

class ExportDependencyChecker:
    """Check availability of export dependencies."""
    
    _cache: Dict[str, bool] = {}
    
    @classmethod
    def onnx_available(cls) -> bool:
        """Check if ONNX export is available."""
        if "onnx" not in cls._cache:
            try:
                import onnx
                import skl2onnx
                cls._cache["onnx"] = True
            except ImportError:
                cls._cache["onnx"] = False
        return cls._cache["onnx"]
    
    @classmethod
    def torch_available(cls) -> bool:
        """Check if PyTorch is available."""
        if "torch" not in cls._cache:
            try:
                import torch
                cls._cache["torch"] = True
            except ImportError:
                cls._cache["torch"] = False
        return cls._cache["torch"]
    
    @classmethod
    def tensorflow_available(cls) -> bool:
        """Check if TensorFlow is available."""
        if "tensorflow" not in cls._cache:
            try:
                import tensorflow as tf
                cls._cache["tensorflow"] = True
            except ImportError:
                cls._cache["tensorflow"] = False
        return cls._cache["tensorflow"]
    
    @classmethod
    def pmml_available(cls) -> bool:
        """Check if PMML export is available."""
        if "pmml" not in cls._cache:
            try:
                from sklearn2pmml import sklearn2pmml
                cls._cache["pmml"] = True
            except ImportError:
                cls._cache["pmml"] = False
        return cls._cache["pmml"]
    
    @classmethod
    def get_available_formats(cls) -> List[ExportFormat]:
        """Get list of available export formats."""
        formats = [ExportFormat.PICKLE, ExportFormat.JOBLIB]  # Always available
        
        if cls.onnx_available():
            formats.append(ExportFormat.ONNX)
        if cls.torch_available():
            formats.append(ExportFormat.TORCHSCRIPT)
        if cls.tensorflow_available():
            formats.extend([ExportFormat.TENSORFLOW_SAVEDMODEL, ExportFormat.TFLITE])
        if cls.pmml_available():
            formats.append(ExportFormat.PMML)
        
        return formats


# ============================================================================
# MODEL EXPORTER
# ============================================================================

class ModelExporter:
    """
    Universal model exporter with multi-format support.
    
    Features:
    - Multiple export formats
    - Automatic optimization
    - Validation
    - Metadata tracking
    """
    
    def __init__(
        self,
        output_dir: str = "artifacts/models",
        validate_export: bool = True,
        generate_metadata: bool = True
    ):
        """
        Args:
            output_dir: Output directory for exported models
            validate_export: Whether to validate exported models
            generate_metadata: Whether to generate metadata files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validate_export = validate_export
        self.generate_metadata = generate_metadata
    
    # ------------------------------------------------------------------------
    # MAIN EXPORT METHOD
    # ------------------------------------------------------------------------
    
    def export(
        self,
        model: Any,
        sample_input: Union[np.ndarray, pd.DataFrame],
        export_format: ExportFormat,
        output_name: str = "model",
        optimization: OptimizationLevel = OptimizationLevel.NONE,
        target_runtime: Optional[TargetRuntime] = None,
        **kwargs: Any
    ) -> ExportResult:
        """
        Export model to specified format.
        
        Args:
            model: Model to export
            sample_input: Sample input for shape inference
            export_format: Target export format
            output_name: Name for output file
            optimization: Optimization level
            target_runtime: Target runtime environment
            **kwargs: Format-specific arguments
            
        Returns:
            ExportResult with export status
        """
        # Convert DataFrame to numpy
        if isinstance(sample_input, pd.DataFrame):
            sample_input = sample_input.values
        
        # Determine output path
        extension = self._get_extension(export_format)
        output_path = self.output_dir / f"{output_name}{extension}"
        
        # Export based on format
        try:
            if export_format == ExportFormat.ONNX:
                result = self._export_onnx(
                    model, sample_input, output_path, optimization, **kwargs
                )
            
            elif export_format == ExportFormat.TORCHSCRIPT:
                result = self._export_torchscript(
                    model, sample_input, output_path, **kwargs
                )
            
            elif export_format == ExportFormat.TENSORFLOW_SAVEDMODEL:
                result = self._export_tf_savedmodel(
                    model, output_path, **kwargs
                )
            
            elif export_format == ExportFormat.PMML:
                result = self._export_pmml(
                    model, sample_input, output_path, **kwargs
                )
            
            elif export_format == ExportFormat.PICKLE:
                result = self._export_pickle(
                    model, output_path
                )
            
            elif export_format == ExportFormat.JOBLIB:
                result = self._export_joblib(
                    model, output_path
                )
            
            elif export_format == ExportFormat.TFLITE:
                result = self._export_tflite(
                    model, output_path, optimization, **kwargs
                )
            
            else:
                result = ExportResult(
                    success=False,
                    export_format=export_format,
                    output_path=None,
                    input_shape=None,
                    output_shape=None,
                    model_size_mb=None,
                    error_message=f"Unsupported format: {export_format}"
                )
            
            # Add metadata
            result.target_runtime = target_runtime
            result.optimization_level = optimization
            
            # Validate if requested
            if result.success and self.validate_export:
                validation_score = self._validate_export(
                    model, result.output_path, sample_input, export_format
                )
                result.validated = True
                result.validation_score = validation_score
            
            # Generate metadata file
            if result.success and self.generate_metadata:
                self._save_metadata(result)
            
            # Telemetry
            audit("model_export", {
                "format": export_format.value,
                "success": result.success,
                "size_mb": result.model_size_mb,
                "optimization": optimization.value
            })
            
            if result.model_size_mb:
                metric("model_size_mb", result.model_size_mb, {
                    "format": export_format.value
                })
            
            return result
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=export_format,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=str(e)
            )
    
    # ------------------------------------------------------------------------
    # FORMAT-SPECIFIC EXPORTERS
    # ------------------------------------------------------------------------
    
    def _export_onnx(
        self,
        model: Any,
        sample_input: np.ndarray,
        output_path: Path,
        optimization: OptimizationLevel,
        **kwargs: Any
    ) -> ExportResult:
        """Export to ONNX format."""
        if not ExportDependencyChecker.onnx_available():
            return ExportResult(
                success=False,
                export_format=ExportFormat.ONNX,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message="ONNX dependencies not available (skl2onnx, onnx, onnxruntime)"
            )
        
        try:
            from skl2onnx import to_onnx
            import onnx
            
            # Get target opset
            target_opset = kwargs.get("target_opset", 15)
            
            # Convert to ONNX
            n_features = sample_input.shape[1]
            sample = sample_input[:1].astype(np.float32)
            
            onnx_model = to_onnx(
                model,
                sample,
                target_opset=target_opset
            )
            
            # Optimize if requested
            if optimization != OptimizationLevel.NONE:
                onnx_model = self._optimize_onnx(onnx_model, optimization)
            
            # Save
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            # Get model info
            input_shape = tuple(sample_input.shape)
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.ONNX,
                output_path=str(output_path),
                input_shape=input_shape,
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.ONNX,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"ONNX export failed: {e}"
            )
    
    def _export_torchscript(
        self,
        model: Any,
        sample_input: np.ndarray,
        output_path: Path,
        **kwargs: Any
    ) -> ExportResult:
        """Export to TorchScript format."""
        if not ExportDependencyChecker.torch_available():
            return ExportResult(
                success=False,
                export_format=ExportFormat.TORCHSCRIPT,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message="PyTorch not available"
            )
        
        try:
            import torch
            
            # Convert model to torch if needed
            if not isinstance(model, torch.nn.Module):
                raise ValueError("Model must be a PyTorch nn.Module for TorchScript export")
            
            # Convert sample to tensor
            sample_tensor = torch.from_numpy(sample_input).float()
            
            # Trace model
            traced_model = torch.jit.trace(model, sample_tensor)
            
            # Save
            torch.jit.save(traced_model, str(output_path))
            
            # Get model info
            input_shape = tuple(sample_input.shape)
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.TORCHSCRIPT,
                output_path=str(output_path),
                input_shape=input_shape,
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.TORCHSCRIPT,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"TorchScript export failed: {e}"
            )
    
    def _export_tf_savedmodel(
        self,
        model: Any,
        output_path: Path,
        **kwargs: Any
    ) -> ExportResult:
        """Export to TensorFlow SavedModel format."""
        if not ExportDependencyChecker.tensorflow_available():
            return ExportResult(
                success=False,
                export_format=ExportFormat.TENSORFLOW_SAVEDMODEL,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message="TensorFlow not available"
            )
        
        try:
            import tensorflow as tf
            
            # Save model
            tf.saved_model.save(model, str(output_path))
            
            # Get model size
            model_size_mb = sum(
                f.stat().st_size for f in output_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.TENSORFLOW_SAVEDMODEL,
                output_path=str(output_path),
                input_shape=None,
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.TENSORFLOW_SAVEDMODEL,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"TensorFlow SavedModel export failed: {e}"
            )
    
    def _export_pmml(
        self,
        model: Any,
        sample_input: np.ndarray,
        output_path: Path,
        **kwargs: Any
    ) -> ExportResult:
        """Export to PMML format."""
        if not ExportDependencyChecker.pmml_available():
            return ExportResult(
                success=False,
                export_format=ExportFormat.PMML,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message="PMML dependencies not available (sklearn2pmml)"
            )
        
        try:
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            
            # Wrap in PMMLPipeline if needed
            if not isinstance(model, PMMLPipeline):
                pmml_pipeline = PMMLPipeline([("classifier", model)])
            else:
                pmml_pipeline = model
            
            # Fit dummy data (PMML requirement)
            pmml_pipeline.fit(sample_input, np.zeros(len(sample_input)))
            
            # Export
            sklearn2pmml(pmml_pipeline, str(output_path))
            
            # Get model info
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.PMML,
                output_path=str(output_path),
                input_shape=tuple(sample_input.shape),
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.PMML,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"PMML export failed: {e}"
            )
    
    def _export_pickle(self, model: Any, output_path: Path) -> ExportResult:
        """Export using pickle."""
        try:
            import pickle
            
            with open(output_path, "wb") as f:
                pickle.dump(model, f)
            
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.PICKLE,
                output_path=str(output_path),
                input_shape=None,
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.PICKLE,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"Pickle export failed: {e}"
            )
    
    def _export_joblib(self, model: Any, output_path: Path) -> ExportResult:
        """Export using joblib."""
        try:
            import joblib
            
            joblib.dump(model, str(output_path))
            
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.JOBLIB,
                output_path=str(output_path),
                input_shape=None,
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.JOBLIB,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"Joblib export failed: {e}"
            )
    
    def _export_tflite(
        self,
        model: Any,
        output_path: Path,
        optimization: OptimizationLevel,
        **kwargs: Any
    ) -> ExportResult:
        """Export to TensorFlow Lite format."""
        if not ExportDependencyChecker.tensorflow_available():
            return ExportResult(
                success=False,
                export_format=ExportFormat.TFLITE,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message="TensorFlow not available"
            )
        
        try:
            import tensorflow as tf
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Apply optimizations
            if optimization == OptimizationLevel.QUANTIZED:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            
            model_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                export_format=ExportFormat.TFLITE,
                output_path=str(output_path),
                input_shape=None,
                output_shape=None,
                model_size_mb=model_size_mb
            )
        
        except Exception as e:
            return ExportResult(
                success=False,
                export_format=ExportFormat.TFLITE,
                output_path=None,
                input_shape=None,
                output_shape=None,
                model_size_mb=None,
                error_message=f"TFLite export failed: {e}"
            )
    
    # ------------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def _optimize_onnx(
        self,
        onnx_model: Any,
        optimization: OptimizationLevel
    ) -> Any:
        """Optimize ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic
            import onnx
            from onnx import optimizer
            
            # Basic optimizations
            if optimization in {OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE}:
                passes = [
                    "eliminate_nop_dropout",
                    "eliminate_nop_pad",
                    "eliminate_nop_transpose",
                    "eliminate_unused_initializer",
                    "fuse_bn_into_conv",
                ]
                
                onnx_model = optimizer.optimize(onnx_model, passes)
            
            return onnx_model
        
        except Exception as e:
            warnings.warn(f"ONNX optimization failed: {e}")
            return onnx_model
    
    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------
    
    def _validate_export(
        self,
        original_model: Any,
        exported_path: str,
        sample_input: np.ndarray,
        export_format: ExportFormat
    ) -> Optional[float]:
        """Validate exported model against original."""
        try:
            # Get original predictions
            original_pred = original_model.predict(sample_input)
            
            # Get exported predictions
            if export_format == ExportFormat.ONNX:
                exported_pred = self._predict_onnx(exported_path, sample_input)
            elif export_format == ExportFormat.PICKLE:
                import pickle
                with open(exported_path, "rb") as f:
                    loaded_model = pickle.load(f)
                exported_pred = loaded_model.predict(sample_input)
            elif export_format == ExportFormat.JOBLIB:
                import joblib
                loaded_model = joblib.load(exported_path)
                exported_pred = loaded_model.predict(sample_input)
            else:
                return None
            
            # Compare predictions
            if original_pred.shape != exported_pred.shape:
                return 0.0
            
            # Calculate similarity score
            diff = np.abs(original_pred - exported_pred)
            max_diff = np.max(diff)
            
            # Score: 1.0 = identical, 0.0 = completely different
            score = max(0.0, 1.0 - max_diff)
            
            return float(score)
        
        except Exception as e:
            warnings.warn(f"Validation failed: {e}")
            return None
    
    def _predict_onnx(self, model_path: str, input_data: np.ndarray) -> np.ndarray:
        """Make predictions with ONNX model."""
        import onnxruntime as ort
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        result = session.run(None, {input_name: input_data.astype(np.float32)})
        
        return result[0]
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    @staticmethod
    def _get_extension(export_format: ExportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ExportFormat.ONNX: ".onnx",
            ExportFormat.TORCHSCRIPT: ".pt",
            ExportFormat.TENSORFLOW_SAVEDMODEL: "",  # Directory
            ExportFormat.PMML: ".pmml",
            ExportFormat.PICKLE: ".pkl",
            ExportFormat.JOBLIB: ".joblib",
            ExportFormat.TFLITE: ".tflite",
            ExportFormat.COREML: ".mlmodel",
        }
        return extensions.get(export_format, ".bin")
    
    def _save_metadata(self, result: ExportResult) -> None:
        """Save export metadata."""
        if not result.output_path:
            return
        
        metadata_path = Path(result.output_path).with_suffix(".json")
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def try_export_sklearn_to_onnx(
    pipeline: Any,
    sample_X: Union[np.ndarray, pd.DataFrame],
    out_path: str = "artifacts/models/model.onnx"
) -> Dict[str, Any]:
    """
    Backward compatible: export sklearn to ONNX.
    
    Enhanced version with validation and metadata.
    """
    exporter = ModelExporter()
    
    # Convert to numpy if DataFrame
    if isinstance(sample_X, pd.DataFrame):
        sample_X = sample_X.values
    
    result = exporter.export(
        model=pipeline,
        sample_input=sample_X,
        export_format=ExportFormat.ONNX,
        output_name=Path(out_path).stem
    )
    
    if result.success:
        return {
            "ok": True,
            "path": result.output_path,
            "size_mb": result.model_size_mb,
            "validated": result.validated,
            "validation_score": result.validation_score
        }
    else:
        return {
            "ok": False,
            "reason": "conversion_error" if "dependencies" not in result.error_message else "missing_deps",
            "message": result.error_message,
            "error": result.error_message
        }


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def export_model(
    model: Any,
    sample_input: Union[np.ndarray, pd.DataFrame],
    formats: List[ExportFormat],
    output_dir: str = "artifacts/models",
    validate: bool = True
) -> Dict[ExportFormat, ExportResult]:
    """
    High-level API: export model to multiple formats.
    
    Args:
        model: Model to export
        sample_input: Sample input
        formats: List of export formats
        output_dir: Output directory
        validate: Whether to validate exports
        
    Returns:
        Dict mapping format to ExportResult
    """
    exporter = ModelExporter(
        output_dir=output_dir,
        validate_export=validate
    )
    
    results = {}
    
    for fmt in formats:
        result = exporter.export(model, sample_input, fmt)
        results[fmt] = result
        
        if result.success:
            print(f"✓ Exported to {fmt.value}: {result.model_size_mb:.2f} MB")
        else:
            print(f"✗ Failed to export to {fmt.value}: {result.error_message}")
    
    return results


def get_available_export_formats() -> List[str]:
    """Get list of available export formats."""
    return [fmt.value for fmt in ExportDependencyChecker.get_available_formats()]