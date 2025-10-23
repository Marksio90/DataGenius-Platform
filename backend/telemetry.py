"""
Telemetria i monitorowanie aplikacji.

Funkcjonalności:
- Zbieranie metryk wydajności
- Śledzenie użycia funkcji
- Statystyki sesji
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """Event telemetryczny."""
    event_type: str
    timestamp: datetime
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """
    Kolektor telemetrii aplikacji.
    
    Zbiera metryki:
    - Czasy wykonania operacji
    - Liczba wywołań funkcji
    - Użycie zasobów
    """

    def __init__(self):
        """Inicjalizacja kolektora."""
        self.events: List[TelemetryEvent] = []
        self.function_calls: Dict[str, int] = defaultdict(int)
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.session_start = datetime.now()

    def record_event(
        self,
        event_type: str,
        duration: Optional[float] = None,
        **metadata
    ) -> None:
        """
        Rejestruje event telemetryczny.

        Args:
            event_type: Typ eventu
            duration: Czas trwania w sekundach
            **metadata: Dodatkowe metadane

        Example:
            >>> tc = TelemetryCollector()
            >>> tc.record_event("data_load", duration=1.5, rows=1000)
        """
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            duration_seconds=duration,
            metadata=metadata
        )
        self.events.append(event)

        # Zlicz wywołania
        self.function_calls[event_type] += 1

        # Zapisz czas wykonania
        if duration is not None:
            self.execution_times[event_type].append(duration)

        logger.debug(f"Telemetry event: {event_type} ({duration}s)")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Zwraca statystyki telemetryczne.

        Returns:
            Dict: Słownik ze statystykami

        Example:
            >>> tc = TelemetryCollector()
            >>> stats = tc.get_statistics()
            >>> 'total_events' in stats
            True
        """
        stats = {
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "total_events": len(self.events),
            "function_calls": dict(self.function_calls),
        }

        # Statystyki czasów wykonania
        execution_stats = {}
        for func_name, times in self.execution_times.items():
            if times:
                execution_stats[func_name] = {
                    "count": len(times),
                    "total_seconds": sum(times),
                    "avg_seconds": sum(times) / len(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                }

        stats["execution_times"] = execution_stats

        return stats

    def get_recent_events(self, n: int = 10) -> List[TelemetryEvent]:
        """
        Zwraca ostatnie n eventów.

        Args:
            n: Liczba eventów

        Returns:
            List[TelemetryEvent]: Lista eventów

        Example:
            >>> tc = TelemetryCollector()
            >>> tc.record_event("test")
            >>> events = tc.get_recent_events(5)
            >>> len(events) >= 0
            True
        """
        return self.events[-n:] if self.events else []

    def clear(self) -> None:
        """Czyści zebrane dane telemetryczne."""
        self.events.clear()
        self.function_calls.clear()
        self.execution_times.clear()
        logger.info("Telemetry data cleared")


class PerformanceTimer:
    """
    Context manager do mierzenia czasu wykonania.

    Example:
        >>> tc = TelemetryCollector()
        >>> with PerformanceTimer("my_operation", tc):
        ...     time.sleep(0.1)
    """

    def __init__(self, operation_name: str, collector: TelemetryCollector, **metadata):
        """
        Inicjalizacja timera.

        Args:
            operation_name: Nazwa operacji
            collector: Kolektor telemetrii
            **metadata: Dodatkowe metadane
        """
        self.operation_name = operation_name
        self.collector = collector
        self.metadata = metadata
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Rozpoczyna pomiar."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Kończy pomiar i zapisuje wynik."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_event(
                self.operation_name,
                duration=duration,
                **self.metadata
            )


# Singleton instance
_telemetry_collector: Optional[TelemetryCollector] = None


def get_telemetry_collector() -> TelemetryCollector:
    """
    Zwraca singleton instancji TelemetryCollector.

    Returns:
        TelemetryCollector: Instancja kolektora
    """
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector


def init_telemetry():
    """Inicjalizuje telemetrię w sesji Streamlit."""
    if "telemetry_collector" not in st.session_state:
        st.session_state.telemetry_collector = TelemetryCollector()
        logger.info("Telemetry collector initialized in session")


def record_telemetry_event(event_type: str, duration: Optional[float] = None, **metadata):
    """
    Rejestruje event w telemetrii sesji.

    Args:
        event_type: Typ eventu
        duration: Czas trwania
        **metadata: Metadane
    """
    if "telemetry_collector" in st.session_state:
        st.session_state.telemetry_collector.record_event(event_type, duration, **metadata)