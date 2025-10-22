# backend/async_ml_trainer.py
"""
Asynchronous trainer / job runner for TMIV.

Lightweight, dependency-free orchestration for long-running tasks (e.g., model training)
with:
- background execution (thread),
- progress reporting (0..1 with message),
- structured job states,
- cancellation,
- optional soft timeout,
- safe result/exception capture.

Intended usage (from Streamlit pages or services):
--------------------------------------------------
    from backend.async_ml_trainer import JobManager, Progress

    jm = JobManager.get()  # singleton-like manager

    def my_long_task(progress: Progress, *, data, plan):
        # User code can call progress.update(...)
        progress.update(0.05, "Preparing data")
        # ... do work ...
        progress.update(0.50, "Training models")
        # ... more work ...
        return {"metrics": {"auc": 0.87}, "artifacts_path": "exports/run_001"}

    job_id = jm.start(target=my_long_task, kwargs={"data": df, "plan": plan})
    # In the UI, poll:
    info = jm.get(job_id)
    if info.done:
        st.write(info.result)  # or info.error

Threading notes:
- We avoid asyncio to keep integration trivial for Streamlit.
- Cancellation is cooperative: `progress.check_cancelled()` or `if progress.cancelled: ...`.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


# =========================
# Job state & dataclasses
# =========================


class JobState(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


@dataclass
class Progress:
    """
    Handle passed into target function to report progress and respond to cancel/timeout.
    """

    _job_id: str
    _manager_ref: "JobManager"
    _cancel_event: threading.Event
    _deadline_ts: Optional[float] = None

    def update(self, value: float, message: str | None = None) -> None:
        """Report progress in [0.0, 1.0]."""
        v = max(0.0, min(1.0, float(value)))
        self._manager_ref._set_progress(self._job_id, v, message or "")

    def check_cancelled(self) -> None:
        """Raise KeyboardInterrupt if cancelled/timeout reached (cooperative cancellation)."""
        if self._cancel_event.is_set():
            raise KeyboardInterrupt("Job cancelled")
        if self._deadline_ts is not None and time.time() > self._deadline_ts:
            raise TimeoutError("Job timed out")

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def remaining_seconds(self) -> Optional[float]:
        if self._deadline_ts is None:
            return None
        return max(0.0, self._deadline_ts - time.time())


@dataclass
class JobInfo:
    id: str
    name: str
    state: JobState = JobState.PENDING
    progress: float = 0.0
    message: str = ""
    started_ts: float | None = None
    finished_ts: float | None = None
    created_ts: float = field(default_factory=time.time)
    timeout_s: float | None = None

    # Results
    result: Any | None = None
    error: str | None = None

    # Internals (not for external mutation)
    _thread: threading.Thread | None = None
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


# =========================
# Job Manager
# =========================


class JobManager:
    """Singleton-like manager for background jobs."""

    _instance: "JobManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._jobs: Dict[str, JobInfo] = {}
        self._jobs_lock = threading.Lock()

    # --- Singleton access ---
    @classmethod
    def get(cls) -> "JobManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = JobManager()
            return cls._instance

    # --- Public API ---

    def start(
        self,
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        timeout_s: float | None = None,
        job_id: str | None = None,
    ) -> str:
        """
        Start a background job by calling `target(progress, *args, **kwargs)`.

        - `target` MUST accept first arg: Progress
        - `timeout_s` is a soft deadline enforced cooperatively (via Progress.check_cancelled)
        - returns `job_id`
        """
        kwargs = kwargs or {}
        job_id = job_id or _gen_job_id()
        job_name = name or getattr(target, "__name__", "job")

        info = JobInfo(id=job_id, name=job_name, timeout_s=timeout_s)
        with self._jobs_lock:
            if job_id in self._jobs:
                raise ValueError(f"Job id already exists: {job_id}")
            self._jobs[job_id] = info

        t = threading.Thread(
            target=self._runner_wrapper,
            args=(job_id, target, args, kwargs),
            name=f"tmiv-job-{job_name}-{job_id[:8]}",
            daemon=True,
        )
        info._thread = t
        t.start()
        return job_id

    def get(self, job_id: str) -> JobInfo:
        with self._jobs_lock:
            if job_id not in self._jobs:
                raise KeyError(f"No such job: {job_id}")
            return self._jobs[job_id]

    def list(self) -> list[JobInfo]:
        with self._jobs_lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> None:
        info = self.get(job_id)
        if info.state in {JobState.SUCCESS, JobState.ERROR, JobState.CANCELLED, JobState.TIMEOUT}:
            return
        info._cancel_event.set()
        self._set_state(job_id, JobState.CANCELLED, message="Cancellation requested")

    # --- Internal helpers ---

    def _runner_wrapper(
        self,
        job_id: str,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        info = self.get(job_id)
        self._set_state(job_id, JobState.RUNNING, message="Started")
        info.started_ts = time.time()
        deadline = info.started_ts + info.timeout_s if info.timeout_s else None

        progress = Progress(job_id, self, info._cancel_event, deadline)

        try:
            self._set_progress(job_id, 0.01, "Initializing")
            result = target(progress, *args, **kwargs)  # user code
            if info._cancel_event.is_set():
                self._set_state(job_id, JobState.CANCELLED, message="Cancelled")
            else:
                self._set_progress(job_id, 1.0, "Done")
                self._set_result(job_id, result)
                self._set_state(job_id, JobState.SUCCESS, message="Success")
        except TimeoutError as te:
            self._set_state(job_id, JobState.TIMEOUT, message=str(te))
            self._set_error(job_id, str(te))
        except KeyboardInterrupt as ki:
            self._set_state(job_id, JobState.CANCELLED, message=str(ki))
            self._set_error(job_id, str(ki))
        except Exception as e:  # noqa: BLE001
            self._set_state(job_id, JobState.ERROR, message=str(e))
            self._set_error(job_id, _safe_exc(e))
        finally:
            info.finished_ts = time.time()

    def _set_state(self, job_id: str, state: JobState, *, message: str | None = None) -> None:
        with self._jobs_lock:
            job = self._jobs[job_id]
            job.state = state
            if message:
                job.message = message

    def _set_progress(self, job_id: str, value: float, message: str) -> None:
        with self._jobs_lock:
            job = self._jobs[job_id]
            job.progress = max(0.0, min(1.0, float(value)))
            if message:
                job.message = message

    def _set_result(self, job_id: str, result: Any) -> None:
        with self._jobs_lock:
            self._jobs[job_id].result = result

    def _set_error(self, job_id: str, error: str) -> None:
        with self._jobs_lock:
            self._jobs[job_id].error = error


# =========================
# Utilities
# =========================


def _gen_job_id() -> str:
    t = int(time.time() * 1000)
    rnd = int(threading.get_ident()) & 0xFFFF
    return f"job-{t:x}-{rnd:x}"


def _safe_exc(e: Exception) -> str:
    txt = f"{type(e).__name__}: {e}"
    return txt[:2000]  # cap size


# =========================
# Example target (manual test)
# =========================

if __name__ == "__main__":
    # Simple CLI test: python -m backend.async_ml_trainer
    jm = JobManager.get()

    def demo_task(progress: Progress, seconds: float = 3.0) -> dict[str, Any]:
        steps = 10
        for i in range(steps):
            time.sleep(seconds / steps)
            progress.check_cancelled()
            progress.update((i + 1) / steps, f"Step {i+1}/{steps}")
        return {"ok": True, "message": "Demo finished"}

    jid = jm.start(target=demo_task, kwargs={"seconds": 2.5}, timeout_s=10)
    print("Started:", jid)

    while True:
        info = jm.get(jid)
        print(f"[{info.state}] {info.progress*100:.0f}% - {info.message}")
        if info.state in {JobState.SUCCESS, JobState.ERROR, JobState.CANCELLED, JobState.TIMEOUT}:
            print("Result:", info.result, "Error:", info.error)
            break
        time.sleep(0.25)
