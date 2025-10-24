"""
TMIV MLflow Integration & Experiment Management v3.0
=====================================================
Zaawansowana integracja MLflow z:
- Automatic MLflow setup & configuration
- Experiment lifecycle management
- Model registry integration
- Metric tracking with history
- Artifact versioning
- Auto-logging for popular frameworks
- Comparative analysis tools
- MLflow server management
- Model deployment utilities
- Experiment comparison & visualization
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from .telemetry import audit, metric as log_metric


# ============================================================================
# MLFLOW AVAILABILITY & SETUP
# ============================================================================

def mlflow_available() -> bool:
    """Check if MLflow is available."""
    try:
        import mlflow
        return True
    except ImportError:
        return False


def get_mlflow_version() -> Optional[str]:
    """Get MLflow version."""
    try:
        import mlflow
        return mlflow.__version__
    except Exception:
        return None


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "default"
    artifact_location: Optional[str] = None
    registry_uri: Optional[str] = None
    
    # Auto-logging
    auto_log_sklearn: bool = True
    auto_log_tensorflow: bool = True
    auto_log_pytorch: bool = True
    auto_log_xgboost: bool = True
    
    # Settings
    nested_runs: bool = False
    log_system_metrics: bool = True


# ============================================================================
# MLFLOW MANAGER
# ============================================================================

class MLflowManager:
    """
    Comprehensive MLflow management system.
    
    Features:
    - Automatic setup & configuration
    - Experiment lifecycle
    - Model registry
    - Artifact management
    - Comparative analysis
    """
    
    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Args:
            config: MLflow configuration (None = defaults)
        """
        self.config = config or MLflowConfig()
        self._mlflow_available = mlflow_available()
        self._current_run_id: Optional[str] = None
        
        if self._mlflow_available:
            self._setup_mlflow()
    
    # ------------------------------------------------------------------------
    # SETUP & CONFIGURATION
    # ------------------------------------------------------------------------
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            import mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)
            
            # Set registry URI (if provided)
            if self.config.registry_uri:
                mlflow.set_registry_uri(self.config.registry_uri)
            
            # Enable auto-logging
            if self.config.auto_log_sklearn:
                try:
                    mlflow.sklearn.autolog()
                except Exception:
                    pass
            
            if self.config.auto_log_xgboost:
                try:
                    mlflow.xgboost.autolog()
                except Exception:
                    pass
            
            audit("mlflow_setup", {
                "tracking_uri": self.config.tracking_uri,
                "experiment": self.config.experiment_name
            })
        
        except Exception as e:
            warnings.warn(f"MLflow setup failed: {e}")
    
    # ------------------------------------------------------------------------
    # RUN MANAGEMENT
    # ------------------------------------------------------------------------
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Start MLflow run.
        
        Args:
            run_name: Name for the run
            nested: Whether this is a nested run
            tags: Tags to add to the run
            
        Returns:
            Run ID if successful
        """
        if not self._mlflow_available:
            warnings.warn("MLflow not available")
            return None
        
        try:
            import mlflow
            
            run = mlflow.start_run(
                run_name=run_name,
                nested=nested or self.config.nested_runs,
                tags=tags
            )
            
            self._current_run_id = run.info.run_id
            
            # Log system info
            if self.config.log_system_metrics:
                self._log_system_info()
            
            audit("mlflow_run_start", {
                "run_id": self._current_run_id,
                "run_name": run_name
            })
            
            return self._current_run_id
        
        except Exception as e:
            warnings.warn(f"Failed to start MLflow run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End current MLflow run.
        
        Args:
            status: Run status (FINISHED/FAILED/KILLED)
        """
        if not self._mlflow_available:
            return
        
        try:
            import mlflow
            mlflow.end_run(status=status)
            
            audit("mlflow_run_end", {
                "run_id": self._current_run_id,
                "status": status
            })
            
            self._current_run_id = None
        
        except Exception as e:
            warnings.warn(f"Failed to end MLflow run: {e}")
    
    # ------------------------------------------------------------------------
    # LOGGING METHODS
    # ------------------------------------------------------------------------
    
    def log_params(self, params: Dict[str, Any]) -> bool:
        """Log parameters."""
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception as e:
                    warnings.warn(f"Failed to log param {key}: {e}")
            
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to log params: {e}")
            return False
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None
    ) -> bool:
        """Log a metric."""
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            mlflow.log_metric(key, float(value), step=step)
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to log metric {key}: {e}")
            return False
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dict of metric name -> value
            step: Step number
            
        Returns:
            Dict of metric name -> success status
        """
        results = {}
        
        for key, value in metrics.items():
            try:
                results[key] = self.log_metric(key, float(value), step)
            except Exception:
                results[key] = False
        
        return results
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None
    ) -> bool:
        """Log an artifact file."""
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            mlflow.log_artifact(str(local_path), artifact_path)
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to log artifact: {e}")
            return False
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ) -> bool:
        """
        Log a model.
        
        Args:
            model: Model to log
            artifact_path: Path in artifact store
            registered_model_name: Name for model registry
            signature: Model signature
            input_example: Example input
            
        Returns:
            Success status
        """
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            
            # Try sklearn first
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    input_example=input_example
                )
                return True
            except Exception:
                pass
            
            # Try pyfunc fallback
            try:
                mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    registered_model_name=registered_model_name
                )
                return True
            except Exception:
                pass
            
            # Last resort: pickle
            import pickle
            model_path = Path(f"/tmp/{artifact_path}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            return self.log_artifact(model_path, artifact_path)
        
        except Exception as e:
            warnings.warn(f"Failed to log model: {e}")
            return False
    
    def log_figure(
        self,
        figure: Any,
        artifact_file: str
    ) -> bool:
        """Log a matplotlib/plotly figure."""
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            mlflow.log_figure(figure, artifact_file)
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to log figure: {e}")
            return False
    
    def log_text(
        self,
        text: str,
        artifact_file: str
    ) -> bool:
        """Log text content."""
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            mlflow.log_text(text, artifact_file)
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to log text: {e}")
            return False
    
    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_file: str
    ) -> bool:
        """Log dictionary as JSON."""
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            mlflow.log_dict(dictionary, artifact_file)
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to log dict: {e}")
            return False
    
    # ------------------------------------------------------------------------
    # MODEL REGISTRY
    # ------------------------------------------------------------------------
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> Optional[str]:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model
            name: Name for registered model
            tags: Tags for the model
            description: Model description
            
        Returns:
            Model version if successful
        """
        if not self._mlflow_available:
            return None
        
        try:
            import mlflow
            
            result = mlflow.register_model(model_uri, name)
            
            if tags or description:
                client = mlflow.tracking.MlflowClient()
                
                if tags:
                    for key, value in tags.items():
                        client.set_model_version_tag(name, result.version, key, value)
                
                if description:
                    client.update_model_version(
                        name, result.version, description=description
                    )
            
            audit("mlflow_model_register", {
                "name": name,
                "version": result.version
            })
            
            return result.version
        
        except Exception as e:
            warnings.warn(f"Failed to register model: {e}")
            return None
    
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> bool:
        """
        Transition model to a new stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage (Staging/Production/Archived)
            archive_existing: Archive existing versions in target stage
            
        Returns:
            Success status
        """
        if not self._mlflow_available:
            return False
        
        try:
            import mlflow
            
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name,
                version,
                stage,
                archive_existing_versions=archive_existing
            )
            
            audit("mlflow_model_transition", {
                "name": name,
                "version": version,
                "stage": stage
            })
            
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to transition model: {e}")
            return False
    
    # ------------------------------------------------------------------------
    # EXPERIMENT ANALYSIS
    # ------------------------------------------------------------------------
    
    def get_experiment_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: Optional[str] = None,
        max_results: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Get runs from an experiment.
        
        Args:
            experiment_name: Experiment name (None = current)
            filter_string: Filter string
            max_results: Maximum number of results
            
        Returns:
            DataFrame with run data
        """
        if not self._mlflow_available:
            return None
        
        try:
            import mlflow
            
            if experiment_name is None:
                experiment_name = self.config.experiment_name
            
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            
            return runs
        
        except Exception as e:
            warnings.warn(f"Failed to get runs: {e}")
            return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_names: Metrics to compare (None = all)
            
        Returns:
            Comparison DataFrame
        """
        if not self._mlflow_available:
            return None
        
        try:
            import mlflow
            
            client = mlflow.tracking.MlflowClient()
            
            comparison_data = []
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                
                row = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                }
                
                # Add metrics
                if metric_names:
                    for metric in metric_names:
                        row[f"metric_{metric}"] = run.data.metrics.get(metric)
                else:
                    for metric, value in run.data.metrics.items():
                        row[f"metric_{metric}"] = value
                
                # Add params
                for param, value in run.data.params.items():
                    row[f"param_{param}"] = value
                
                comparison_data.append(row)
            
            return pd.DataFrame(comparison_data)
        
        except Exception as e:
            warnings.warn(f"Failed to compare runs: {e}")
            return None
    
    def get_best_run(
        self,
        metric_name: str,
        experiment_name: Optional[str] = None,
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get best run based on a metric.
        
        Args:
            metric_name: Metric to optimize
            experiment_name: Experiment name (None = current)
            maximize: Whether to maximize the metric
            
        Returns:
            Best run info
        """
        runs_df = self.get_experiment_runs(experiment_name)
        
        if runs_df is None or runs_df.empty:
            return None
        
        metric_col = f"metrics.{metric_name}"
        if metric_col not in runs_df.columns:
            return None
        
        # Sort by metric
        sorted_df = runs_df.sort_values(
            by=metric_col,
            ascending=not maximize
        )
        
        best_run = sorted_df.iloc[0]
        
        return {
            "run_id": best_run.get("run_id"),
            "run_name": best_run.get("tags.mlflow.runName", ""),
            metric_name: best_run.get(metric_col),
            "start_time": best_run.get("start_time")
        }
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def _log_system_info(self) -> None:
        """Log system information."""
        import platform
        
        system_info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        self.log_params(system_info)


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility + enhancement)
# ============================================================================

def log_run_metrics(task: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatible: log metrics to MLflow.
    
    Enhanced version with auto-setup and fallback.
    """
    if not mlflow_available():
        return {
            "ok": False,
            "reason": "missing_mlflow",
            "message": "MLflow not available"
        }
    
    try:
        import mlflow
        
        with mlflow.start_run(run_name=f"tmiv-{task}"):
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
        
        audit("mlflow_metrics_logged", {
            "task": task,
            "metrics_count": len(metrics)
        })
        
        return {"ok": True}
    
    except Exception as e:
        return {
            "ok": False,
            "reason": "mlflow_error",
            "message": str(e)
        }


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def quick_log(
    experiment_name: str,
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Path]] = None,
    model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Quick logging API for simple use cases.
    
    Args:
        experiment_name: Experiment name
        run_name: Run name
        params: Parameters to log
        metrics: Metrics to log
        artifacts: Artifact files to log
        model: Model to log
        
    Returns:
        Result dict with run_id
    """
    if not mlflow_available():
        return {"ok": False, "reason": "mlflow_not_available"}
    
    manager = MLflowManager(MLflowConfig(experiment_name=experiment_name))
    run_id = manager.start_run(run_name=run_name)
    
    if params:
        manager.log_params(params)
    
    if metrics:
        manager.log_metrics(metrics)
    
    if artifacts:
        for artifact in artifacts:
            manager.log_artifact(artifact)
    
    if model is not None:
        manager.log_model(model)
    
    manager.end_run()
    
    return {"ok": True, "run_id": run_id}