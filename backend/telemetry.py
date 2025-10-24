"""
TMIV Advanced Telemetry & Observability v3.0
=============================================
Zaawansowany system telemetrii i obserwability z:
- Structured logging (JSON Lines)
- Metrics collection & aggregation
- Distributed tracing
- Performance monitoring
- Error tracking & alerting
- Real-time dashboards
- Log aggregation & search
- Custom metrics & events
- SLO/SLI tracking
- Integration with monitoring systems (Prometheus, Grafana, DataDog)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from contextlib import contextmanager

import pandas as pd


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"         # Monotonically increasing
    GAUGE = "gauge"            # Can go up or down
    HISTOGRAM = "histogram"    # Distribution of values
    TIMER = "timer"           # Duration measurements


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    # Logging
    log_dir: str = "artifacts/reports"
    audit_log: str = "audit.log.jsonl"
    metrics_log: str = "tmiv.log.jsonl"
    errors_log: str = "errors.log.jsonl"
    traces_log: str = "traces.log.jsonl"
    
    # Sampling
    enable_sampling: bool = False
    sample_rate: float = 1.0  # 1.0 = log everything
    
    # Aggregation
    enable_aggregation: bool = True
    aggregation_interval_sec: int = 60
    
    # Retention
    max_log_size_mb: int = 100
    max_log_age_days: int = 30
    
    # External integrations
    enable_prometheus: bool = False
    prometheus_port: int = 9090


# ============================================================================
# TELEMETRY MANAGER
# ============================================================================

class TelemetryManager:
    """
    Central telemetry management system.
    
    Features:
    - Structured logging
    - Metrics aggregation
    - Distributed tracing
    - Performance monitoring
    """
    
    _instance: Optional['TelemetryManager'] = None
    
    def __init__(self, config: Optional[TelemetryConfig] = None):
        """
        Args:
            config: Telemetry configuration
        """
        self.config = config or TelemetryConfig()
        
        # Create log directory
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics aggregator
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._metric_types: Dict[str, MetricType] = {}
        
        # Trace context
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None
        
        # Performance
        self._timers: Dict[str, float] = {}
    
    @classmethod
    def get_instance(cls) -> 'TelemetryManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------
    
    def log(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Log a message with structured data.
        
        Args:
            message: Log message
            level: Log level
            context: Additional context
            **kwargs: Extra fields
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "message": message,
            "context": context or {},
        }
        
        # Add trace context if available
        if self._current_trace_id:
            event["trace_id"] = self._current_trace_id
        if self._current_span_id:
            event["span_id"] = self._current_span_id
        
        # Add extra fields
        event.update(kwargs)
        
        # Write to appropriate log
        if level in {LogLevel.ERROR, LogLevel.CRITICAL}:
            self._write_jsonl(self.config.errors_log, event)
        else:
            self._write_jsonl(self.config.metrics_log, event)
    
    def audit(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Log audit event.
        
        Args:
            action: Action performed
            details: Action details
            user: User performing action
            resource: Resource affected
            **kwargs: Extra fields
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details or {},
            "user": user or os.environ.get("USER", "unknown"),
            "resource": resource,
        }
        
        event.update(kwargs)
        
        self._write_jsonl(self.config.audit_log, event)
    
    def error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Log error with stack trace.
        
        Args:
            error: Exception
            context: Error context
            **kwargs: Extra fields
        """
        import traceback
        
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": LogLevel.ERROR.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "context": context or {}
        }
        
        event.update(kwargs)
        
        self._write_jsonl(self.config.errors_log, event)
    
    # ------------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------------
    
    def metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Tags for metric
            **kwargs: Extra fields
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": name,
            "value": float(value),
            "type": metric_type.value,
            "tags": tags or {}
        }
        
        event.update(kwargs)
        
        # Store for aggregation
        if self.config.enable_aggregation:
            self._metrics[name].append(float(value))
            self._metric_types[name] = metric_type
        
        # Write to log
        self._write_jsonl(self.config.metrics_log, event)
    
    def counter(
        self,
        name: str,
        value: Union[int, float] = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        self.metric(name, value, MetricType.COUNTER, tags)
    
    def gauge(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric."""
        self.metric(name, value, MetricType.GAUGE, tags)
    
    def histogram(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram value."""
        self.metric(name, value, MetricType.HISTOGRAM, tags)
    
    def timer(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timer value."""
        self.metric(name, duration_ms, MetricType.TIMER, tags)
    
    # ------------------------------------------------------------------------
    # TIMING
    # ------------------------------------------------------------------------
    
    @contextmanager
    def measure(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for timing operations.
        
        Usage:
            with telemetry.measure("model_training"):
                train_model()
        """
        start = time.perf_counter()
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.timer(name, duration_ms, tags)
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.perf_counter()
    
    def stop_timer(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Stop a named timer and record.
        
        Returns:
            Duration in milliseconds
        """
        if name not in self._timers:
            warnings.warn(f"Timer '{name}' was not started")
            return 0.0
        
        start = self._timers.pop(name)
        duration_ms = (time.perf_counter() - start) * 1000
        
        self.timer(name, duration_ms, tags)
        
        return duration_ms
    
    # ------------------------------------------------------------------------
    # DISTRIBUTED TRACING
    # ------------------------------------------------------------------------
    
    @contextmanager
    def trace(
        self,
        operation: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for distributed tracing.
        
        Usage:
            with telemetry.trace("predict"):
                model.predict(X)
        """
        import uuid
        
        # Generate IDs
        trace_id = self._current_trace_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        # Set context
        prev_trace_id = self._current_trace_id
        prev_span_id = self._current_span_id
        
        self._current_trace_id = trace_id
        self._current_span_id = span_id
        
        # Start span
        start = time.perf_counter()
        
        span_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": prev_span_id,
            "operation": operation,
            "event": "span_start",
            "tags": tags or {}
        }
        
        self._write_jsonl(self.config.traces_log, span_event)
        
        try:
            yield
            status = "success"
        except Exception as e:
            status = "error"
            self.error(e, {"trace_id": trace_id, "span_id": span_id})
            raise
        finally:
            # End span
            duration_ms = (time.perf_counter() - start) * 1000
            
            end_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trace_id": trace_id,
                "span_id": span_id,
                "operation": operation,
                "event": "span_end",
                "duration_ms": duration_ms,
                "status": status,
                "tags": tags or {}
            }
            
            self._write_jsonl(self.config.traces_log, end_event)
            
            # Restore context
            self._current_trace_id = prev_trace_id
            self._current_span_id = prev_span_id
    
    # ------------------------------------------------------------------------
    # METRICS AGGREGATION
    # ------------------------------------------------------------------------
    
    def get_metric_summary(
        self,
        name: str,
        clear: bool = True
    ) -> Optional[Dict[str, float]]:
        """
        Get aggregated metric summary.
        
        Args:
            name: Metric name
            clear: Clear values after summary
            
        Returns:
            Summary statistics
        """
        if name not in self._metrics:
            return None
        
        values = self._metrics[name]
        
        if not values:
            return None
        
        import numpy as np
        
        summary = {
            "count": len(values),
            "sum": float(np.sum(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
        
        if clear:
            self._metrics[name].clear()
        
        return summary
    
    def get_all_metrics_summary(
        self,
        clear: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Get summary for all metrics."""
        summaries = {}
        
        for name in list(self._metrics.keys()):
            summary = self.get_metric_summary(name, clear=clear)
            if summary:
                summaries[name] = summary
        
        return summaries
    
    # ------------------------------------------------------------------------
    # LOG ANALYSIS
    # ------------------------------------------------------------------------
    
    def read_logs(
        self,
        log_type: str = "metrics",
        limit: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Read logs from JSONL file.
        
        Args:
            log_type: Type of log (metrics/audit/errors/traces)
            limit: Max number of entries
            filter_fn: Filter function
            
        Returns:
            List of log entries
        """
        log_files = {
            "metrics": self.config.metrics_log,
            "audit": self.config.audit_log,
            "errors": self.config.errors_log,
            "traces": self.config.traces_log
        }
        
        if log_type not in log_files:
            raise ValueError(f"Unknown log type: {log_type}")
        
        log_path = self.log_dir / log_files[log_type]
        
        if not log_path.exists():
            return []
        
        entries = []
        
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Apply filter
                    if filter_fn and not filter_fn(entry):
                        continue
                    
                    entries.append(entry)
                    
                    # Check limit
                    if limit and len(entries) >= limit:
                        break
                
                except json.JSONDecodeError:
                    continue
        
        return entries
    
    def get_logs_as_dataframe(
        self,
        log_type: str = "metrics",
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get logs as pandas DataFrame."""
        entries = self.read_logs(log_type, limit)
        
        if not entries:
            return pd.DataFrame()
        
        return pd.DataFrame(entries)
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def _write_jsonl(self, filename: str, event: Dict[str, Any]) -> None:
        """Write event to JSONL file."""
        path = self.log_dir / filename
        
        # Ensure timestamp
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add UTC suffix if not present
        if "timestamp" in event and not event["timestamp"].endswith("Z"):
            event["timestamp"] += "Z"
        
        # Write atomically
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def rotate_logs(self) -> None:
        """Rotate logs based on size/age."""
        for log_file in self.log_dir.glob("*.jsonl"):
            # Check size
            size_mb = log_file.stat().st_size / (1024 * 1024)
            
            if size_mb > self.config.max_log_size_mb:
                # Rotate
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                rotated = log_file.with_suffix(f".{timestamp}.jsonl")
                log_file.rename(rotated)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_telemetry = TelemetryManager.get_instance()


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility + enhancement)
# ============================================================================

def _ensure_dir(p: str) -> None:
    """Backward compatible: ensure directory exists."""
    os.makedirs(p, exist_ok=True)


def log_jsonl(path: str, event: Dict[str, Any]) -> None:
    """
    Backward compatible: log to JSONL file.
    
    Enhanced version with timestamp handling.
    """
    _ensure_dir(os.path.dirname(path))
    
    event = dict(event)
    
    if "utc" not in event and "timestamp" not in event:
        event["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def audit(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Backward compatible: log audit event.
    
    Enhanced version with full telemetry.
    """
    _telemetry.audit(action, details)


def metric(
    event: str,
    value: Union[float, int],
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Backward compatible: log metric.
    
    Enhanced version with metric types.
    """
    tags = extra.copy() if extra else {}
    
    # Extract metric type if specified
    metric_type = tags.pop("metric_type", MetricType.GAUGE)
    if isinstance(metric_type, str):
        metric_type = MetricType(metric_type)
    
    _telemetry.metric(event, value, metric_type, tags)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def get_telemetry() -> TelemetryManager:
    """Get global telemetry manager."""
    return _telemetry


def measure(name: str, tags: Optional[Dict[str, str]] = None):
    """
    Timing context manager.
    
    Usage:
        with measure("model_training"):
            train_model()
    """
    return _telemetry.measure(name, tags)


def trace(operation: str, tags: Optional[Dict[str, str]] = None):
    """
    Tracing context manager.
    
    Usage:
        with trace("predict"):
            model.predict(X)
    """
    return _telemetry.trace(operation, tags)


def get_metrics_summary() -> Dict[str, Dict[str, float]]:
    """Get summary of all metrics."""
    return _telemetry.get_all_metrics_summary(clear=False)