"""
Utility functions for The Most Important Variables v2.2
(robust decorators: rate_limit, retry_on_failure, time_execution, memoize_in_session)

Improvements vs 2.1:
- Safe Streamlit fallback (works outside Streamlit).
- rate_limit: optional key_fn/per_session/env toggle; precise retry-after.
- retry_on_failure: exponential backoff with jitter, optional retry filter.
- time_execution: resilient logging; stores last duration in session_state.
- memoize_in_session: thread-safe, optional TTL & max_entries; LRU eviction.
"""

from __future__ import annotations

import time
import functools
import hashlib
import json
import threading
from typing import Callable, Any, Dict, List, Optional

# =========================
# Streamlit-safe import
# =========================
try:
    import streamlit as st  # type: ignore
    if not hasattr(st, "session_state"):
        class _SS(dict): ...
        st.session_state = _SS()  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    class _DummySt:
        session_state: Dict[str, Any] = {}
        def info(self, *a, **k): print(*a)
        def warning(self, *a, **k): print(*a)
        def error(self, *a, **k): print(*a)
    st = _DummySt()  # type: ignore


# =========================
# Helpers / error classes
# =========================

class RateLimitError(Exception):
    """Raised when a rate limit is exceeded."""


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have failed."""


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    """
    Build a stable SHA1 hash of a (possibly nested) structure by using
    JSON with sorted keys and UTF-8 encoding.
    """
    try:
        s = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=repr)
    except Exception:
        # Absolute fallback – still deterministic enough for memo keys
        s = repr(payload)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _ensure_session_id() -> str:
    """
    Returns a per-Streamlit-session id (stable across reruns).
    """
    if "_util_session_id" not in st.session_state:
        st.session_state["_util_session_id"] = hashlib.sha1(
            f"{time.time_ns()}-{id(st.session_state)}".encode("utf-8")
        ).hexdigest()
    return st.session_state["_util_session_id"]


def _now() -> float:
    return time.time()


# ==================================
# 1) Rate limiting (thread-safe)
# ==================================

_rate_lock = threading.Lock()

def rate_limit(
    max_calls: int,
    time_window: int,
    *,
    key_fn: Optional[Callable[..., str]] = None,
    per_session: bool = True,
    env_toggle: str = "DISABLE_RATE_LIMIT"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Rate limiting decorator (per Streamlit session if available, else per-process).

    Args:
        max_calls: maximum number of calls allowed in the given window.
        time_window: sliding window in seconds.
        key_fn: optional function to compute a bucket key from args/kwargs; useful
                for per-user or per-resource limits. If None, the bucket key is
                derived from the function identity (and session if per_session=True).
        per_session: if True (default), bucket is scoped to Streamlit session id.
        env_toggle: if this environment variable is set to a truthy value,
                    the rate limiter is disabled (handy for local dev).

    Hard guarantees:
      - thread-safe
      - ignores timestamps older than `time_window`
    Usage:
        @rate_limit(max_calls=10, time_window=60)
        def api_call():
            pass
    """
    if max_calls <= 0 or time_window <= 0:
        # No-op: don't wrap needlessly
        def no_limit_decorator(func: Callable) -> Callable:
            return func
        return no_limit_decorator

    import os
    if str(os.getenv(env_toggle, "")).strip().lower() in {"1", "true", "yes", "on"}:
        def no_limit_decorator(func: Callable) -> Callable:
            return func
        return no_limit_decorator

    calls_key_prefix = "_rate_calls"

    def decorator(func: Callable) -> Callable:
        func_key = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            now = _now()
            with _rate_lock:
                # Prefer session_state to survive Streamlit reruns
                store: Dict[str, List[float]] = st.session_state.setdefault("_rate_limit_store", {})

                # Compose bucket id
                parts = [calls_key_prefix, func_key]
                if per_session:
                    parts.append(_ensure_session_id())
                if key_fn is not None:
                    try:
                        parts.append(str(key_fn(*args, **kwargs)))
                    except Exception:
                        # keep going with base bucket
                        parts.append("keyfn_error")
                bucket = "::".join(parts)

                timestamps = store.get(bucket, [])
                cutoff = now - time_window
                timestamps = [t for t in timestamps if t >= cutoff]

                if len(timestamps) >= max_calls:
                    # seconds until the window frees at the oldest hit
                    retry_after = max(0.0, time_window - (now - timestamps[0]))
                    raise RateLimitError(
                        f"Rate limit exceeded: {max_calls} calls per {time_window}s. "
                        f"Retry after ~{retry_after:.0f}s"
                    )

                # record call
                timestamps.append(now)
                store[bucket] = timestamps

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ==================================
# 2) Retry with backoff (+ jitter)
# ==================================

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    *,
    retry_if: Optional[Callable[[BaseException], bool]] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Retry decorator for functions that may fail.
    Implements exponential backoff with small jitter.

    Args:
        max_retries: total attempts (including the first). If 1 => no retries.
        delay: base delay in seconds (first backoff step).
               Real delay ~= delay * (2**attempt) + jitter
        retry_if: optional predicate taking the exception and returning True if we
                  should retry. Defaults to retry on any Exception.

    Usage:
        @retry_on_failure(max_retries=3, delay=1.0)
        def unstable_operation():
            pass
    """
    max_retries = max(1, int(max_retries))
    base_delay = max(0.0, float(delay))

    def should_retry(exc: BaseException) -> bool:
        try:
            return True if retry_if is None else bool(retry_if(exc))
        except Exception:
            return True

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1 and should_retry(e):
                        # exponential backoff (1, 2, 4, ...) * base_delay + jitter (0..base_delay*0.2)
                        factor = 2 ** attempt
                        jitter = ((time.time_ns() % 1_000_000) / 1_000_000.0) * (base_delay * 0.2)
                        time.sleep(base_delay * factor + jitter)
                        continue
                    break
            # Exhausted
            raise last_exc if last_exc is not None else RetryExhaustedError("Retries exhausted")

        return wrapper

    return decorator


# ==================================
# 3) Execution timing
# ==================================

def time_execution(func: Callable) -> Callable:
    """
    Decorator to measure execution time (perf_counter for precision).
    Writes to Streamlit if available, otherwise prints to stdout.

    Usage:
        @time_execution
        def slow_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            try:
                # show once per call; store last duration for optional later use
                st.session_state["_last_exec_times"] = st.session_state.get("_last_exec_times", {})
                st.session_state["_last_exec_times"][func.__name__] = dt
                st.info(f"⏱️ {func.__name__} executed in {dt:.3f}s")
            except Exception:
                print(f"{func.__name__} executed in {dt:.3f}s")

    return wrapper


# ==================================
# 4) Memoization in Streamlit session
# ==================================

_memo_lock = threading.Lock()

def memoize_in_session(
    key_prefix: str = "",
    *,
    ttl_seconds: Optional[float] = None,
    max_entries: int = 256
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Memoization decorator using Streamlit session_state.
    - Builds a stable hash key from function name + args/kwargs (JSON→SHA1).
    - Stores results in st.session_state["_memo_cache"] with a simple LRU eviction.
    - Optional TTL (seconds): entries older than TTL are recomputed.
    - Safe for reruns; per-session only.

    Usage:
        @memoize_in_session(key_prefix="my_func")
        def expensive_operation(x):
            return x ** 2
    """
    PREFIX = key_prefix or "memo"
    CACHE_KEY = "_memo_cache"
    ORDER_KEY = "_memo_cache_order"
    META_KEY = "_memo_cache_meta"  # per key: {"ts": float}

    def _ensure_cache():
        st.session_state.setdefault(CACHE_KEY, {})
        st.session_state.setdefault(ORDER_KEY, [])
        st.session_state.setdefault(META_KEY, {})

    def _touch_lru(k: str):
        order: List[str] = st.session_state[ORDER_KEY]
        if k in order:
            order.remove(k)
        order.append(k)
        # trim
        while len(order) > max_entries:
            old = order.pop(0)
            st.session_state[CACHE_KEY].pop(old, None)
            st.session_state[META_KEY].pop(old, None)
        st.session_state[ORDER_KEY] = order

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            _ensure_cache()

            # Build a stable hash from name+args/kwargs (repr-capped)
            try:
                args_ser = [repr(a)[:512] for a in args]
                kwargs_ser = {k: repr(kwargs[k])[:512] for k in sorted(kwargs)}
            except Exception:
                args_ser, kwargs_ser = [str(a) for a in args], {k: str(v) for k, v in kwargs.items()}

            key_payload = {
                "prefix": PREFIX,
                "func": f"{func.__module__}.{func.__qualname__}",
                "args": args_ser,
                "kwargs": kwargs_ser,
            }
            h = _stable_json_hash(key_payload)
            cache_key = f"{PREFIX}::{h}"

            with _memo_lock:
                cache: Dict[str, Any] = st.session_state[CACHE_KEY]
                meta: Dict[str, Dict[str, Any]] = st.session_state[META_KEY]

                # TTL check
                if cache_key in cache:
                    if ttl_seconds is not None:
                        ts = float(meta.get(cache_key, {}).get("ts", 0.0))
                        if _now() - ts > float(ttl_seconds):
                            # expired – drop and recompute
                            cache.pop(cache_key, None)
                            meta.pop(cache_key, None)
                        else:
                            _touch_lru(cache_key)
                            return cache[cache_key]
                    else:
                        _touch_lru(cache_key)
                        return cache[cache_key]

                # Compute and store
                result = func(*args, **kwargs)
                cache[cache_key] = result
                meta[cache_key] = {"ts": _now()}
                st.session_state[CACHE_KEY] = cache
                st.session_state[META_KEY] = meta
                _touch_lru(cache_key)
                return result

        return wrapper

    return decorator


__all__ = [
    "RateLimitError",
    "RetryExhaustedError",
    "rate_limit",
    "retry_on_failure",
    "time_execution",
    "memoize_in_session",
]
