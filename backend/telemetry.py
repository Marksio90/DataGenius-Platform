"""
telemetry.py (Stage 8)
Docstring (PL): Prosty tracer + timer + JSONL logger do artifacts/logs/tmiv_telemetry.jsonl
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import time, json, os
from contextlib import contextmanager

LOG_DIR = "artifacts/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "tmiv_telemetry.jsonl")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
    HAS_OTEL = True
except Exception:
    HAS_OTEL = False

_tracer = None

def init():
    global _tracer
    if not HAS_OTEL:
        return None
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)
    return _tracer

def _write_jsonl(obj: Dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

@contextmanager
def timer(name: str, extra: Optional[Dict[str, Any]] = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        rec = {"event": "timer", "name": name, "elapsed_sec": round(dt, 4)}
        if extra: rec.update(extra)
        _write_jsonl(rec)

def log_event(name: str, payload: Optional[Dict[str, Any]] = None):
    rec = {"event": name, "ts": time.time()}
    if payload: rec.update(payload)
    _write_jsonl(rec)