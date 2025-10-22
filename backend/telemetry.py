# -*- coding: utf-8 -*-
"""
Lightweight CSV telemetry with safety features:
- Auto-create directory/file with header
- Thread-safe writes (+ optional cross-process locking if `portalocker` is installed)
- Optional size-based rotation with backups
- JSON-safe, single-line `detail` (accepts str/dict/list/anything)
- CSV injection mitigation for text fields
- Context manager `timeit()` for timing
- Resilient: buffers failed writes in-memory and retries next time
- Can be disabled via env: TMIV_TELEMETRY_DISABLE=1

Columns: ts, session, event, detail, elapsed_sec
"""

import os
import csv
import time
import json
import threading
from typing import Optional, List, Tuple
from datetime import datetime, timezone

# ───────────────────────────── Defaults ─────────────────────────────
_TELEMETRY_DIR = os.path.join("telemetry")
os.makedirs(_TELEMETRY_DIR, exist_ok=True)
_DEFAULT_LOG = os.path.join(_TELEMETRY_DIR, "log.csv")


# ─────────────────────── Optional cross-proc lock ───────────────────
try:
    import portalocker  # type: ignore
    _HAVE_PORTALOCKER = True
except Exception:
    portalocker = None  # type: ignore
    _HAVE_PORTALOCKER = False


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _sanitize_for_csv(text: str) -> str:
    """
    Prevent spreadsheet formula injection while keeping content human-readable.
    Prefix with a single quote if string starts with one of: = + - @ \t
    """
    if not text:
        return text
    if text[0] in ("=", "+", "-", "@", "\t"):
        return "'" + text
    return text


class Telemetry:
    """
    Lekki logger CSV:
      - automatycznie tworzy katalog/plik i nagłówek,
      - detail może być str lub dict/list (serializowane do JSON),
      - thread-safe, optional cross-process lock (portalocker),
      - contextmanager timeit() do pomiaru czasu,
      - opcjonalna rotacja pliku po rozmiarze,
      - buforowanie na wypadek błędów IO.

    Kolumny: ts, session, event, detail, elapsed_sec
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        disabled: Optional[bool] = None,
        max_bytes: int = 5 * 1024 * 1024,  # 5 MB
        backup_count: int = 5,
        max_detail_len: int = 8000,       # hard cap for 'detail' cell
    ):
        self.path = path or _DEFAULT_LOG
        self._lock = threading.Lock()
        self.max_bytes = int(max_bytes) if max_bytes and max_bytes > 0 else 0
        self.backup_count = max(0, int(backup_count or 0))
        self.max_detail_len = max(256, int(max_detail_len or 0))
        # enable/disable via ctor or env
        self.disabled = _env_bool("TMIV_TELEMETRY_DISABLE", False) if disabled is None else bool(disabled)
        # failed rows buffer (persist across failures)
        self._buffer: List[Tuple[str, str, str, str, str]] = []
        self._ensure_file_with_header()

    # ---------- internal helpers ----------

    def _ensure_file_with_header(self) -> None:
        """Create file with header if it doesn't exist or is empty."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        need_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if need_header:
            # Write header atomically-ish under lock
            with self._lock:
                # Double-check in case of races
                need_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
                if need_header:
                    with open(self.path, "w", newline="", encoding="utf-8") as f:
                        if _HAVE_PORTALOCKER:
                            portalocker.lock(f, portalocker.LOCK_EX)  # type: ignore
                        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                        w.writerow(["ts", "session", "event", "detail", "elapsed_sec"])
                        if _HAVE_PORTALOCKER:
                            portalocker.unlock(f)  # type: ignore

    @staticmethod
    def _now_iso() -> str:
        """ISO 8601 w UTC z milisekundami (np. 2025-01-01T12:34:56.789Z)."""
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _coerce_detail(self, detail) -> str:
        """detail: str | dict | list | anything -> schludny, jednowierszowy string (JSON for complex types)."""
        try:
            if isinstance(detail, (dict, list, tuple)):
                s = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
            else:
                s = str(detail)
        except Exception:
            s = "<detail_unserializable>"

        # single-line for CSV
        s = " ".join(s.splitlines()).strip()
        if len(s) > self.max_detail_len:
            s = s[: self.max_detail_len - 3] + "..."
        return _sanitize_for_csv(s)

    def _maybe_rotate(self) -> None:
        """Rotate file when size exceeds max_bytes."""
        if not self.max_bytes:
            return
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) >= self.max_bytes:
                # Rotate: .N -> .N+1 (descending), base -> .1
                for i in range(self.backup_count - 1, 0, -1):
                    src = f"{self.path}.{i}"
                    dst = f"{self.path}.{i+1}"
                    if os.path.exists(src):
                        try:
                            os.replace(src, dst)
                        except Exception:
                            pass
                if self.backup_count > 0:
                    try:
                        os.replace(self.path, f"{self.path}.1")
                    except Exception:
                        # Fallback: rename may fail on some FS; try copy/remove
                        try:
                            with open(self.path, "rb") as r, open(f"{self.path}.1", "wb") as w:
                                w.write(r.read())
                            os.remove(self.path)
                        except Exception:
                            pass
                # New file with header
                self._ensure_file_with_header()
        except Exception:
            # non-fatal
            pass

    def _write_rows(self, rows: List[Tuple[str, str, str, str, str]]) -> None:
        """Write prepared rows (internal: assumes self._lock held)."""
        if not rows:
            return
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                if _HAVE_PORTALOCKER:
                    portalocker.lock(f, portalocker.LOCK_EX)  # type: ignore
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                w.writerows(rows)
                if _HAVE_PORTALOCKER:
                    portalocker.unlock(f)  # type: ignore
        except Exception:
            # On error keep rows buffered for next attempt
            self._buffer.extend(rows)
            raise

    # ---------- public API ----------

    def event(self, session: str, event: str, detail: object = "", elapsed_sec: float = 0.0) -> None:
        """
        Zaloguj zdarzenie.
        - session: identyfikator sesji/użytkownika
        - event: nazwa zdarzenia
        - detail: str/dict/list/… (automatycznie serializowane i czyszczone z nowych linii)
        - elapsed_sec: liczba sekund (float); gdy brak, podaj 0.0
        """
        if self.disabled:
            return

        ts = self._now_iso()
        # sanitize & coerce
        session_s = _sanitize_for_csv(str(session))
        event_s = _sanitize_for_csv(str(event))
        detail_str = self._coerce_detail(detail)
        try:
            elapsed_str = f"{float(elapsed_sec):.3f}"
        except Exception:
            elapsed_str = ""

        row = (ts, session_s, event_s, detail_str, elapsed_str)

        with self._lock:
            # rotate if needed
            self._maybe_rotate()

            # try flush buffer first (if any)
            pending = []
            if self._buffer:
                pending, self._buffer = self._buffer, []
            try:
                self._write_rows(pending + [row])
            except Exception:
                # swallow to not break the app
                pass

    # Prosty contextmanager do mierzenia czasu wykonywania
    def timeit(self, session: str, event: str, detail: object = ""):
        class _Timer:
            def __init__(self, outer: "Telemetry", session: str, event: str, detail: object):
                self._outer = outer
                self._session = session
                self._event = event
                self._detail = detail
                self._t0: Optional[float] = None

            def __enter__(self):
                self._t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                dt = (time.perf_counter() - self._t0) if self._t0 is not None else 0.0
                d = self._detail
                # If an exception occurred, include brief info in detail (same column)
                if exc_type is not None:
                    d = {"detail": self._detail, "error": getattr(exc_type, "__name__", str(exc_type)), "message": str(exc)}
                self._outer.event(self._session, self._event, d, dt)
                # don't suppress exceptions
                return False

        return _Timer(self, session, event, detail)

    # Decorator helper: logs elapsed time for the wrapped callable
    def timeit_deco(self, session: str, event: str, detail: object = ""):
        def _decorator(fn):
            def _wrapped(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    self.event(session, event, detail, time.perf_counter() - t0)
            return _wrapped
        return _decorator


# Default singleton (keeps backward-compat with existing imports)
telemetry = Telemetry()
