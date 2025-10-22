# core/services/telemetry_service.py
"""
TelemetryService – lekka telemetria (OpenTelemetry jeśli dostępny, inaczej log-only)
dla TMIV – Advanced ML Platform.

Zgodne z `ITelemetryService` (core/interfaces.py).

Co potrafi:
- start_span(name, attrs)  -> context manager na span (OTel lub no-op).
- add_event(name, attrs)   -> event do bieżącego spanu (lub do logów).
- incr_counter(name, val)  -> licznik metryk (OTel Meter lub wewnętrzny).
- observe_value(name, val) -> histogram/gauge (OTel Meter lub wewnętrzny).
- start_heartbeat()/stop_heartbeat() -> wątek wysyłający beaty co N sekund.
- snapshot()               -> zrzut stanu (bez sekretów).

Uwagi:
- Brak twardych zależności: jeśli OpenTelemetry nie jest zainstalowane lub wyłączone
  w ustawieniach, wszystko działa jako no-op/logging.
- Inicjalizacja eksportera OTel pozostaje po stronie aplikacji/środowiska.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ContextManager, Dict, Optional

# --- konfiguracja (opcjonalna) ---
try:  # pragma: no cover
    from config.settings import get_settings
except Exception:  # pragma: no cover
    def get_settings():
        class _S:
            tmiv_telemetry = "on"
            tmiv_telemetry_exporter = "auto"
            app_name = "tmiv"
        return _S()

# --- OpenTelemetry (opcjonalnie) ---
_OTEL = True
try:  # pragma: no cover
    from opentelemetry import trace, metrics  # type: ignore
    from opentelemetry.trace import Tracer  # type: ignore
    from opentelemetry.metrics import Meter  # type: ignore
except Exception:  # pragma: no cover
    _OTEL = False
    Tracer = object  # type: ignore
    Meter = object  # type: ignore


@dataclass
class _MetricStore:
    """Bardzo prosty fallback liczników/metryk, jeśli OTel nie jest dostępny."""
    counters: Dict[str, float] = field(default_factory=dict)
    last_values: Dict[str, float] = field(default_factory=dict)

    def incr(self, name: str, value: float) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def observe(self, name: str, value: float) -> None:
        self.last_values[name] = float(value)


class TelemetryService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._enabled = str(getattr(self._settings, "tmiv_telemetry", "on")).lower() not in {"off", "false", "0", ""}
        self._exporter = str(getattr(self._settings, "tmiv_telemetry_exporter", "auto"))

        self._logger = logging.getLogger("telemetry")
        self._metric_store = _MetricStore()

        # OpenTelemetry tracer/meter (bez twardej inicjalizacji eksportera)
        self._tracer: Tracer | None = None
        self._meter: Meter | None = None
        if _OTEL and self._enabled:  # pragma: no cover
            try:
                svc = getattr(self._settings, "app_name", "tmiv")
                self._tracer = trace.get_tracer(f"{svc}.tracer")
                self._meter = metrics.get_meter(f"{svc}.meter")
            except Exception:
                self._tracer = None
                self._meter = None

        # Lazy registry instrumentów OTel (tworzymy przy pierwszym użyciu)
        self._otel_counters: Dict[str, Any] = {}
        self._otel_histos: Dict[str, Any] = {}

        # Heartbeat
        self._hb_thread: threading.Thread | None = None
        self._hb_stop = threading.Event()
        self._hb_interval = 30.0  # sekundy

    # =========================
    # API
    # =========================

    def start_span(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> ContextManager[None]:
        """
        Zwraca context manager; gdy OTel jest dostępny i włączony – startuje span.
        W przeciwnym razie zwraca `nullcontext()`.
        """
        if self._enabled and self._tracer is not None:  # pragma: no cover
            cm = self._tracer.start_as_current_span(name)
            # ustaw atrybuty po wejściu do spanu
            class _AttrCM:
                def __enter__(self, *a):
                    span = cm.__enter__()
                    try:
                        if attrs:
                            for k, v in attrs.items():
                                span.set_attribute(str(k), _safe_attr(v))
                    except Exception:
                        pass
                    return span
                def __exit__(self, exc_type, exc, tb):
                    return cm.__exit__(exc_type, exc, tb)
            return _AttrCM()
        # fallback
        return nullcontext()

    def add_event(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        if self._enabled and _OTEL and self._tracer is not None:  # pragma: no cover
            try:
                span = trace.get_current_span()
                if span is not None:
                    span.add_event(name, attributes={k: _safe_attr(v) for k, v in (attrs or {}).items()})
                    return
            except Exception:
                pass
        # fallback → log
        if self._enabled:
            self._logger.debug("event %s attrs=%s", name, (attrs or {}))

    def incr_counter(self, name: str, value: float = 1.0, attrs: Optional[Dict[str, Any]] = None) -> None:
        if self._enabled and _OTEL and self._meter is not None:  # pragma: no cover
            try:
                ctr = self._otel_counters.get(name)
                if ctr is None:
                    ctr = self._meter.create_counter(name)  # type: ignore[attr-defined]
                    self._otel_counters[name] = ctr
                ctr.add(float(value), attributes={k: _safe_attr(v) for k, v in (attrs or {}).items()})
                return
            except Exception:
                pass
        # fallback
        self._metric_store.incr(name, float(value))

    def observe_value(self, name: str, value: float, attrs: Optional[Dict[str, Any]] = None) -> None:
        if self._enabled and _OTEL and self._meter is not None:  # pragma: no cover
            try:
                hist = self._otel_histos.get(name)
                if hist is None:
                    hist = self._meter.create_histogram(name)  # type: ignore[attr-defined]
                    self._otel_histos[name] = hist
                hist.record(float(value), attributes={k: _safe_attr(v) for k, v in (attrs or {}).items()})
                return
            except Exception:
                pass
        # fallback
        self._metric_store.observe(name, float(value))

    def start_heartbeat(self, interval_sec: Optional[float] = None) -> None:
        """
        Uruchom lekki wątek wysyłający heartbeat (event + counter).
        Idempotentne – wielokrotne wywołanie nie uruchomi kilku wątków.
        """
        if interval_sec is not None:
            self._hb_interval = float(interval_sec)

        if self._hb_thread and self._hb_thread.is_alive():
            return

        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="tmiv-heartbeat", daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_stop.set()
            self._hb_thread.join(timeout=2.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._enabled),
            "otel_available": bool(_OTEL),
            "exporter": self._exporter,
            "hb_running": bool(self._hb_thread and self._hb_thread.is_alive()),
            "fallback_counters": dict(self._metric_store.counters),
            "fallback_last_values": dict(self._metric_store.last_values),
        }

    # =========================
    # Helpers
    # =========================

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.is_set():
            try:
                with self.start_span("heartbeat"):
                    self.add_event("heartbeat", {"ts": int(time.time())})
                    self.incr_counter("tmiv.heartbeat", 1.0)
            except Exception:
                # nie przerywaj pętli
                pass
            finally:
                self._hb_stop.wait(self._hb_interval)


def _safe_attr(v: Any) -> Any:
    """OTel atrybuty muszą być prostymi typami; nadmiarowe typy rzutujemy do str/float/int/bool."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return float(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return "<unserializable>"


__all__ = ["TelemetryService"]
