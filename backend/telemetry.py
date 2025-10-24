
from __future__ import annotations
import json, os, datetime
from typing import Dict, Any

def _ensure_dir(p: str)->None:
    os.makedirs(p, exist_ok=True)

def log_jsonl(path: str, event: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    event = dict(event)
    event.setdefault("utc", datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def audit(action: str, details: Dict[str, Any] | None = None) -> None:
    log_jsonl("artifacts/reports/audit.log.jsonl", {"action": action, "details": details or {}})

def metric(event: str, value: float | int, extra: Dict[str, Any] | None = None) -> None:
    payload = {"event": event, "value": value}
    if extra: payload.update(extra)
    log_jsonl("artifacts/reports/tmiv.log.jsonl", payload)
