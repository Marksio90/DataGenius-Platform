# backend/telemetry.py
"""
Telemetry & instrumentation for TMIV – Advanced ML Platform.

Design goals
------------
- *Optional* OpenTelemetry support (no hard deps). When OTEL is not installed,
  everything still works and falls back to structured logging.
- One, process-local singleton with a tiny API:
    - init_telemetry(...), shutdown_telemetry()
    - start_span(name, attrs=None) -> context manager
    - add_event(name, attrs=None)
    - incr_counter(name, value=1, attrs=None)
    - observe_value(name, value, attrs=None)   # histogram-like
    - start_heartbeat(interval_sec=60), stop_heartbeat()
    - snapshot()  # short, debug-friendly state
- OTLP over HTTP if available (reads standard OTEL_* envs); console exporters as fallback.

Environment (typical)
---------------------
- OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.example.com
- OTEL_RESOURCE_ATTRIBUTES=service.name=tmiv,service.version=2.0
- OTEL_EXPORTER_OTLP_HEADERS=...    (optional)
- Set TMIV_TELEMETRY=off to hard-disable (e.g., in CI).

This module intentionally avoids Streamlit/UI imports.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("app.telemetry")

# -----------------------------
# Optional OpenTelemetry imports
# -----------------------------
_OTEL_OK = True
try:  # Trace core
    from opentelemetry import trace
    from opentelemetry.trace import Tracer
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    # Prefer OTLP exporter if present
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
    except Exception:  # pragma: no cover
        OTLPSpanExporter = None  # type: ignore
except Exception:  # pragma: no cover
    _OTEL_OK = False
    trace = None  # type: ignore
    Tracer = object  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore

try:  # Resource (optional)
    from opentelemetry.sdk.resources import Resource  # type: ignore
except Exception:  # pragma: no cover
    Resource = None  # type: ignore

# Metrics API is still evolving; use best-effort if available
_METRICS_OK = True
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
    from opentelemetry.sdk.metrics.export import (  # type: ignore
        PeriodicExportingMetricReader,
    )
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore
            OTLPMetricExporter,
        )
    except Exception:  # pragma: no cover
        OTLPMetricExporter = None  # type: ignore
except Exception:  # pragma: no cover
    _METRICS_OK = False
    metrics = None  # type: ignore
    MeterProvider = None  # type: ignore
    PeriodicExportingMetricReader = None  # type: ignore
    OTLPMetricExporter = None  # type: ignore


# -----------------------------
# Data classes / config
# -----------------------------

@dataclass
class TelemetryConfig:
    service_name: str = "tmiv"
    service_version: str = "dev"
    exporter: str = "auto"  # "auto" | "otlp" | "console" | "none"
    otlp_endpoint: Optional[str] = None  # if None, taken from env
    metrics_interval_sec: float = 30.0
    heartbeat_interval_sec: float = 60.0
    disabled: bool = False


@dataclass
class _State:
    tracer: Any = None
    meter: Any = None
    provider_trace: Any = None
    provider_metrics: Any = None
    heartbeat_thread: Optional[threading.Thread] = None
    heartbeat_stop: threading.Event = field(default_factory=threading.Event)
    initialized: bool = False
    backend: str = "logger-only"  # "otlp" | "console" | "logger-only"


# -----------------------------
# Core implementation
# -----------------------------

class Telemetry:
    def __init__(self) -> None:
        self._cfg = TelemetryConfig()
        self._state = _State()

    # ---- lifecycle ----
    def init(self, cfg: TelemetryConfig | None = None) -> None:
        if self._state.initialized:
            return
        # Resolve config (env can override)
        self._cfg = self._merge_with_env(cfg or TelemetryConfig())
        if self._cfg.disabled or os.getenv("TMIV_TELEMETRY", "").lower() in {"0", "false", "off"}:
            self._state.initialized = True
            self._state.backend = "disabled"
            logger.info("Telemetry disabled.")
            return

        # Setup resources
        resource = None
        if Resource is not None:
            attrs = {
                "service.name": self._cfg.service_name,
                "service.version": self._cfg.service_version,
                "telemetry.sdk.name": "tmiv",
            }
            resource = Resource.create(attrs)

        # Tracing
        if _OTEL_OK and self._cfg.exporter != "none":
            try:
                tp = TracerProvider(resource=resource) if resource is not None else TracerProvider()
                exporter = None
                chosen = "console"
                if self._should_use_otlp():
                    if OTLPSpanExporter is not None:
                        exporter = OTLPSpanExporter(endpoint=self._cfg.otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None))
                        chosen = "otlp"
                if exporter is None and ConsoleSpanExporter is not None:
                    exporter = ConsoleSpanExporter()
                    chosen = "console"
                if exporter is not None and BatchSpanProcessor is not None:
                    tp.add_span_processor(BatchSpanProcessor(exporter))
                trace.set_tracer_provider(tp)  # type: ignore[attr-defined]
                tracer = trace.get_tracer(self._cfg.service_name)  # type: ignore[attr-defined]
                self._state.provider_trace = tp
                self._state.tracer = tracer
                self._state.backend = chosen
            except Exception as e:
                logger.exception("Failed to initialize tracing; falling back to logging: %s", e)
                self._state.tracer = None
                self._state.backend = "logger-only"
        else:
            self._state.tracer = None
            self._state.backend = "logger-only"

        # Metrics (best effort)
        if _METRICS_OK and self._cfg.exporter != "none":
            try:
                readers = []
                exporter = None
                if self._should_use_otlp() and OTLPMetricExporter is not None and PeriodicExportingMetricReader is not None:
                    exporter = OTLPMetricExporter(endpoint=self._cfg.otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None))
                    readers.append(PeriodicExportingMetricReader(exporter, export_interval_millis=int(self._cfg.metrics_interval_sec * 1000)))
                if MeterProvider is not None:
                    mp = MeterProvider(metric_readers=readers, resource=resource)
                    metrics.set_meter_provider(mp)  # type: ignore[attr-defined]
                    meter = metrics.get_meter(self._cfg.service_name)  # type: ignore[attr-defined]
                    self._state.provider_metrics = mp
                    self._state.meter = meter
            except Exception as e:
                logger.exception("Failed to initialize metrics; falling back to logging: %s", e)
                self._state.meter = None

        self._state.initialized = True
        logger.info("Telemetry initialized (backend=%s).", self._state.backend)

    def shutdown(self) -> None:
        # Stop heartbeat
        self.stop_heartbeat()
        # Providers usually have shutdown() but not strictly required
        try:
            if self._state.provider_metrics and hasattr(self._state.provider_metrics, "shutdown"):
                self._state.provider_metrics.shutdown()
        except Exception:
            pass
        try:
            if self._state.provider_trace and hasattr(self._state.provider_trace, "shutdown"):
                self._state.provider_trace.shutdown()
        except Exception:
            pass
        self._state = _State(initialized=False)
        logger.info("Telemetry shutdown complete.")

    # ---- spans / events ----
    @contextmanager
    def start_span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing spans.
        Usage:
            with telemetry.start_span("train", {"model":"lgbm"}):
                ...
        """
        if self._state.tracer is None or not _OTEL_OK:
            # Logging fallback
            logger.info("SPAN start: %s | %s", name, attrs or {})
            try:
                yield
            except Exception:
                logger.exception("SPAN error: %s", name)
                raise
            finally:
                logger.info("SPAN end  : %s", name)
            return

        # Real span
        attrs = attrs or {}
        try:
            with self._state.tracer.start_as_current_span(name) as span:  # type: ignore[union-attr]
                for k, v in (attrs or {}).items():
                    try:
                        span.set_attribute(str(k), v)  # type: ignore[attr-defined]
                    except Exception:
                        continue
                yield
        except Exception:
            # Ensure exception gets recorded on span if possible
            try:
                span = trace.get_current_span()  # type: ignore[attr-defined]
                if span:
                    span.record_exception(Exception("Unhandled error in span"))  # type: ignore[attr-defined]
            except Exception:
                pass
            raise

    def add_event(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an event to the current span (if any) and log it.
        """
        attrs = attrs or {}
        if self._state.tracer is not None and _OTEL_OK:
            try:
                span = trace.get_current_span()  # type: ignore[attr-defined]
                if span is not None:
                    span.add_event(name, attributes=attrs)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Always log
        logger.info("EVENT %s | %s", name, attrs)

    # ---- metrics (best effort) ----
    def incr_counter(self, name: str, value: float = 1.0, attrs: Optional[Dict[str, Any]] = None) -> None:
        """
        Increment a monotonic counter (fallback: log).
        """
        attrs = attrs or {}
        if self._state.meter is not None and _METRICS_OK:
            try:
                counter = getattr(self, "_counter_cache", {}).get(name)
                if counter is None:
                    counter = self._state.meter.create_counter(name)  # type: ignore[union-attr]
                    if not hasattr(self, "_counter_cache"):
                        self._counter_cache = {}
                    self._counter_cache[name] = counter
                counter.add(float(value), attributes=attrs)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        logger.info("METRIC counter %s += %s | %s", name, value, attrs)

    def observe_value(self, name: str, value: float, attrs: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a value measurement (histogram-like). Fallback to logging.
        """
        attrs = attrs or {}
        if self._state.meter is not None and _METRICS_OK:
            try:
                hist = getattr(self, "_hist_cache", {}).get(name)
                if hist is None:
                    hist = self._state.meter.create_histogram(name)  # type: ignore[union-attr]
                    if not hasattr(self, "_hist_cache"):
                        self._hist_cache = {}
                    self._hist_cache[name] = hist
                hist.record(float(value), attributes=attrs)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        logger.info("METRIC observe %s = %s | %s", name, value, attrs)

    # ---- heartbeat ----
    def start_heartbeat(self, interval_sec: Optional[float] = None) -> None:
        """
        Emit periodic heartbeat events. Safe to call multiple times.
        """
        interval = float(interval_sec or self._cfg.heartbeat_interval_sec)
        if self._state.heartbeat_thread and self._state.heartbeat_thread.is_alive():
            return
        self._state.heartbeat_stop.clear()

        def _loop():
            logger.info("Telemetry heartbeat thread started (interval=%ss).", interval)
            while not self._state.heartbeat_stop.is_set():
                try:
                    self.add_event(
                        "heartbeat",
                        {
                            "service": self._cfg.service_name,
                            "version": self._cfg.service_version,
                            "backend": self._state.backend,
                        },
                    )
                except Exception:
                    logger.exception("Heartbeat emit failed.")
                self._state.heartbeat_stop.wait(interval)
            logger.info("Telemetry heartbeat thread stopped.")

        t = threading.Thread(target=_loop, name="tmiv-telemetry-heartbeat", daemon=True)
        t.start()
        self._state.heartbeat_thread = t

    def stop_heartbeat(self) -> None:
        if self._state.heartbeat_thread:
            self._state.heartbeat_stop.set()
            try:
                self._state.heartbeat_thread.join(timeout=2.0)
            except Exception:
                pass
            self._state.heartbeat_thread = None

    # ---- utils ----
    def _should_use_otlp(self) -> bool:
        if self._cfg.exporter == "otlp":
            return True
        if self._cfg.exporter == "console":
            return False
        # auto
        endpoint = self._cfg.otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        return bool(endpoint)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "initialized": self._state.initialized,
            "backend": self._state.backend,
            "has_tracer": bool(self._state.tracer is not None),
            "has_meter": bool(self._state.meter is not None),
            "heartbeat_running": bool(self._state.heartbeat_thread and self._state.heartbeat_thread.is_alive()),
            "service_name": self._cfg.service_name,
            "service_version": self._cfg.service_version,
        }

    @staticmethod
    def _merge_with_env(cfg: TelemetryConfig) -> TelemetryConfig:
        # Allow env overrides
        svc = os.getenv("TMIV_SERVICE_NAME", cfg.service_name)
        ver = os.getenv("TMIV_SERVICE_VERSION", cfg.service_version)
        exp = os.getenv("TMIV_TELEMETRY_EXPORTER", cfg.exporter)
        ep = os.getenv("TMIV_OTLP_ENDPOINT", cfg.otlp_endpoint) or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", cfg.otlp_endpoint)
        hb = float(os.getenv("TMIV_HEARTBEAT_SEC", cfg.heartbeat_interval_sec))
        mi = float(os.getenv("TMIV_METRICS_INTERVAL_SEC", cfg.metrics_interval_sec))
        dis = os.getenv("TMIV_TELEMETRY", "").lower() in {"0", "false", "off"} or cfg.disabled
        return TelemetryConfig(
            service_name=svc,
            service_version=ver,
            exporter=exp,
            otlp_endpoint=ep,
            heartbeat_interval_sec=hb,
            metrics_interval_sec=mi,
            disabled=dis,
        )


# -----------------------------
# Singleton helpers
# -----------------------------

_singleton: Optional[Telemetry] = None
_singleton_lock = threading.Lock()


def get_telemetry() -> Telemetry:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = Telemetry()
        return _singleton


# Convenience top-level wrappers
def init_telemetry(cfg: TelemetryConfig | None = None) -> None:
    get_telemetry().init(cfg)


def shutdown_telemetry() -> None:
    get_telemetry().shutdown()


__all__ = [
    "Telemetry",
    "TelemetryConfig",
    "get_telemetry",
    "init_telemetry",
    "shutdown_telemetry",
]
