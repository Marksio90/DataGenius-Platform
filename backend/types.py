"""
TMIV Advanced Schema Definitions & Type System v3.0
====================================================
Zaawansowany system schematów i typów z:
- Comprehensive type definitions
- Pydantic validation
- JSON schema generation
- API contracts
- Event streaming schemas
- Artifact metadata
- Configuration schemas
- Versioned schemas
- Schema evolution support
- OpenAPI integration
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator


# ============================================================================
# ENUMS
# ============================================================================

class Stage(str, Enum):
    """Pipeline execution stages."""
    UPLOAD = "upload"
    SANITY = "sanity"
    CONTRACTS = "contracts"
    EDA = "eda"
    PLAN = "plan"
    TRAIN = "train"
    RESULTS = "results"
    EXPORT = "export"
    DEPLOY = "deploy"
    MONITOR = "monitor"


class NoticeLevel(str, Enum):
    """Notice severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ArtifactKind(str, Enum):
    """Types of artifacts."""
    MODEL = "model"
    METRICS = "metrics"
    PLOT = "plot"
    PDF = "pdf"
    SBOM = "sbom"
    MANIFEST = "manifest"
    BASELINE = "baseline"
    THRESHOLD = "threshold"
    DATASET = "dataset"
    CONFIG = "config"
    REPORT = "report"
    SCHEMA = "schema"
    FEATURE_SET = "feature_set"


class OptimizationObjective(str, Enum):
    """Threshold optimization objectives."""
    YOUDEN = "youden"
    COST = "cost"
    F1 = "f1"
    GMEAN = "gmean"
    DISTANCE = "distance"
    BUSINESS = "business"


class ModelFramework(str, Enum):
    """ML frameworks."""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class ExportFormat(str, Enum):
    """Model export formats."""
    ONNX = "onnx"
    PICKLE = "pickle"
    JOBLIB = "joblib"
    TORCHSCRIPT = "torchscript"
    SAVEDMODEL = "savedmodel"
    PMML = "pmml"


# ============================================================================
# NOTIFICATIONS & EVENTS
# ============================================================================

class UiNotice(BaseModel):
    """UI notification with internationalization support."""
    
    code: str = Field(
        ...,
        description="Notification code (e.g., TMIV-001)",
        regex=r"^[A-Z]+-\d+$"
    )
    level: NoticeLevel
    message_pl: str = Field(..., description="Message in Polish")
    message_en: Optional[str] = Field(None, description="Message in English")
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context"
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dismissible: bool = True
    action_url: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "TMIV-001",
                "level": "WARN",
                "message_pl": "Wykryto brakujące wartości w kolumnie 'age'",
                "message_en": "Missing values detected in column 'age'",
                "details": {"column": "age", "missing_count": 42},
                "dismissible": True
            }
        }


class ProgressEvent(BaseModel):
    """Real-time progress event for UI timeline."""
    
    run_id: str = Field(..., description="Unique run identifier")
    stage: Stage = Field(..., description="Current pipeline stage")
    
    pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Completion percentage"
    )
    
    note_pl: Optional[str] = Field(None, description="Progress note (Polish)")
    note_en: Optional[str] = Field(None, description="Progress note (English)")
    
    eta_seconds: Optional[float] = Field(None, description="Estimated time to completion")
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Stage-specific metadata"
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "run-20250124-123456",
                "stage": "train",
                "pct": 65.5,
                "note_pl": "Trenowanie modelu RandomForest",
                "eta_seconds": 45.2
            }
        }


class StageResult(BaseModel):
    """Result of a pipeline stage execution."""
    
    stage: Stage
    status: Literal["success", "failure", "skipped"] = "success"
    
    duration_sec: float = Field(..., ge=0.0)
    start_time: datetime
    end_time: datetime
    
    artifacts: List['ArtifactRef'] = Field(default_factory=list)
    notices: List[UiNotice] = Field(default_factory=list)
    
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


# ============================================================================
# ARTIFACTS
# ============================================================================

class ArtifactRef(BaseModel):
    """Reference to a generated artifact."""
    
    kind: ArtifactKind = Field(..., description="Type of artifact")
    path: str = Field(..., description="File path or URL")
    
    sha256: str = Field(
        ...,
        description="SHA256 hash for integrity",
        regex=r"^[a-f0-9]{64}$"
    )
    
    size_bytes: Optional[int] = Field(None, ge=0)
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Artifact-specific metadata"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "kind": "model",
                "path": "artifacts/models/model_v1.pkl",
                "sha256": "abc123...",
                "size_bytes": 1048576,
                "version": "1.0.0"
            }
        }


class ModelArtifact(ArtifactRef):
    """Model artifact with extended metadata."""
    
    kind: Literal[ArtifactKind.MODEL] = ArtifactKind.MODEL
    
    framework: ModelFramework
    export_format: ExportFormat
    
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    hyperparameters: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, float]] = None
    
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None


class DatasetArtifact(ArtifactRef):
    """Dataset artifact metadata."""
    
    kind: Literal[ArtifactKind.DATASET] = ArtifactKind.DATASET
    
    n_rows: int = Field(..., ge=0)
    n_columns: int = Field(..., ge=0)
    
    schema_version: Optional[str] = None
    split: Optional[Literal["train", "val", "test", "full"]] = None
    
    column_types: Optional[Dict[str, str]] = None
    data_quality: Optional['DataQualityMetrics'] = None


# ============================================================================
# THRESHOLDS
# ============================================================================

class ThresholdManifest(BaseModel):
    """Manifest for decision threshold selection."""
    
    run_id: str = Field(..., description="Run identifier")
    version: str = Field(default="1.0.0")
    
    threshold: float = Field(..., ge=0.0, le=1.0)
    
    objective: OptimizationObjective = Field(
        ...,
        description="Optimization objective used"
    )
    
    cost_matrix: Optional[Dict[str, float]] = Field(
        None,
        description="Cost matrix (FP, FN, TP, TN)"
    )
    
    metrics_at_threshold: Dict[str, float] = Field(
        ...,
        description="Performance metrics at selected threshold"
    )
    
    constraints: Optional[Dict[str, float]] = Field(
        None,
        description="Constraints used (min_precision, min_recall, etc.)"
    )
    
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0)
    pr_auc: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('cost_matrix')
    def validate_cost_matrix(cls, v):
        """Validate cost matrix has required keys."""
        if v is not None:
            required_keys = {'FP', 'FN'}
            if not required_keys.issubset(v.keys()):
                raise ValueError(f"Cost matrix must contain: {required_keys}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "run-20250124-123456",
                "version": "1.0.0",
                "threshold": 0.42,
                "objective": "cost",
                "cost_matrix": {"FP": 1.0, "FN": 5.0},
                "metrics_at_threshold": {
                    "precision": 0.85,
                    "recall": 0.78,
                    "f1": 0.81
                }
            }
        }


# ============================================================================
# DATA QUALITY
# ============================================================================

class DataQualityMetrics(BaseModel):
    """Comprehensive data quality metrics."""
    
    completeness: float = Field(..., ge=0.0, le=1.0)
    uniqueness: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    
    missing_rate: Dict[str, float] = Field(default_factory=dict)
    duplicate_rate: float = Field(0.0, ge=0.0, le=1.0)
    
    outlier_counts: Optional[Dict[str, int]] = None
    drift_detected: Optional[Dict[str, bool]] = None


class DataQualityBaseline(BaseModel):
    """Baseline for data quality monitoring."""
    
    run_id: str = Field(..., description="Baseline run ID")
    schema_hash: str = Field(
        ...,
        description="Schema fingerprint",
        regex=r"^[a-f0-9]{64}$"
    )
    
    missingness: Dict[str, float] = Field(
        ...,
        description="Missing value rate per column"
    )
    
    cardinality: Dict[str, int] = Field(
        ...,
        description="Unique value counts per column"
    )
    
    dtypes: Dict[str, str] = Field(
        ...,
        description="Data types per column"
    )
    
    statistics: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Statistical summaries (mean, std, min, max, etc.)"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    
    @validator('missingness')
    def validate_missingness_range(cls, v):
        """Ensure missingness values are in [0, 1]."""
        for col, rate in v.items():
            if not 0 <= rate <= 1:
                raise ValueError(
                    f"Missingness rate for {col} must be in [0, 1], got {rate}"
                )
        return v


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    run_id: str = Field(..., description="Unique run identifier")
    
    # Data
    input_path: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    
    # Stages to execute
    stages: List[Stage] = Field(
        default_factory=lambda: [
            Stage.UPLOAD,
            Stage.SANITY,
            Stage.CONTRACTS,
            Stage.EDA,
            Stage.PLAN,
            Stage.TRAIN,
            Stage.RESULTS,
            Stage.EXPORT
        ]
    )
    
    # Training
    model_type: ModelFramework = ModelFramework.SKLEARN
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Validation
    test_size: float = Field(0.2, ge=0.0, le=1.0)
    cv_folds: int = Field(5, ge=2)
    
    # Thresholds
    threshold_objective: OptimizationObjective = OptimizationObjective.YOUDEN
    cost_matrix: Optional[Dict[str, float]] = None
    
    # Export
    export_formats: List[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.ONNX, ExportFormat.PICKLE]
    )
    
    # Monitoring
    enable_drift_detection: bool = True
    enable_fairness_check: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# RUN RESULTS
# ============================================================================

class PipelineRun(BaseModel):
    """Complete pipeline run metadata and results."""
    
    run_id: str = Field(..., description="Unique run identifier")
    config: PipelineConfig
    
    status: Literal["running", "success", "failure", "cancelled"] = "running"
    
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_sec: Optional[float] = None
    
    stage_results: Dict[Stage, StageResult] = Field(default_factory=dict)
    
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    notices: List[UiNotice] = Field(default_factory=list)
    
    final_metrics: Optional[Dict[str, float]] = None
    threshold_manifest: Optional[ThresholdManifest] = None
    
    error: Optional[str] = None
    
    @root_validator
    def compute_duration(cls, values):
        """Automatically compute duration if both times are set."""
        start = values.get('start_time')
        end = values.get('end_time')
        
        if start and end and values.get('duration_sec') is None:
            values['duration_sec'] = (end - start).total_seconds()
        
        return values


# ============================================================================
# API REQUESTS/RESPONSES
# ============================================================================

class TrainRequest(BaseModel):
    """API request to start training."""
    
    dataset_path: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    
    model_type: ModelFramework = ModelFramework.SKLEARN
    hyperparameters: Optional[Dict[str, Any]] = None
    
    test_size: float = Field(0.2, ge=0.0, le=1.0)
    random_state: Optional[int] = None


class TrainResponse(BaseModel):
    """API response after starting training."""
    
    run_id: str
    status: str = "started"
    message: str = "Training pipeline started"
    
    artifacts_url: Optional[str] = None
    progress_url: Optional[str] = None


class PredictRequest(BaseModel):
    """API request for predictions."""
    
    model_version: str
    data: Union[List[Dict[str, Any]], Dict[str, List[Any]]]
    
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_probabilities: bool = True


class PredictResponse(BaseModel):
    """API response with predictions."""
    
    predictions: List[Union[int, float]]
    probabilities: Optional[List[float]] = None
    
    model_version: str
    threshold_used: Optional[float] = None
    
    latency_ms: float


# ============================================================================
# FEATURE STORE
# ============================================================================

class FeatureSetMetadata(BaseModel):
    """Metadata for feature set."""
    
    name: str = Field(..., description="Feature set name")
    version: str = Field(default="1.0.0")
    
    features: List[str] = Field(..., description="List of feature names")
    
    dtype_map: Dict[str, str] = Field(
        ...,
        description="Data types for each feature"
    )
    
    statistics: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Feature statistics"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    tags: Optional[Dict[str, str]] = None


# ============================================================================
# MONITORING
# ============================================================================

class DriftReport(BaseModel):
    """Data/model drift detection report."""
    
    run_id: str
    baseline_id: str
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    drift_detected: bool
    drifted_features: List[str] = Field(default_factory=list)
    
    drift_scores: Dict[str, float] = Field(
        ...,
        description="Drift score per feature"
    )
    
    drift_method: Literal["psi", "ks", "chi2", "wasserstein"] = "psi"
    threshold: float = 0.1


class FairnessReport(BaseModel):
    """Model fairness assessment report."""
    
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    protected_attributes: List[str]
    
    disparate_impact: Dict[str, float]
    demographic_parity: Dict[str, float]
    equalized_odds: Dict[str, float]
    
    bias_detected: bool
    recommendations: List[str] = Field(default_factory=list)


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Keep original simple types for backward compatibility
StageType = Stage
NoticeLevelType = NoticeLevel
ArtifactKindType = ArtifactKind


# Update forward references
StageResult.model_rebuild()
PipelineRun.model_rebuild()
DatasetArtifact.model_rebuild()