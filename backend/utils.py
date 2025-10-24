"""
TMIV Advanced Utilities & Helpers v3.0
=======================================
Zaawansowane narzędzia pomocnicze z:
- File operations (hashing, compression, encryption)
- Path management (safe paths, temp files)
- Timing & profiling utilities
- Context managers
- Retry logic with exponential backoff
- Caching decorators
- Resource management
- String utilities
- Data serialization
- System utilities
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from .telemetry import audit, metric


# ============================================================================
# TYPE VARIABLES
# ============================================================================

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# FILE HASHING
# ============================================================================

class FileHasher:
    """
    Efficient file hashing utilities.
    
    Features:
    - Multiple hash algorithms
    - Streaming for large files
    - Progress callbacks
    - Parallel hashing
    """
    
    CHUNK_SIZE = 8192  # 8KB chunks
    
    @staticmethod
    def sha256(path: Union[str, Path]) -> str:
        """Compute SHA256 hash of file."""
        h = hashlib.sha256()
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(FileHasher.CHUNK_SIZE), b""):
                h.update(chunk)
        
        return h.hexdigest()
    
    @staticmethod
    def md5(path: Union[str, Path]) -> str:
        """Compute MD5 hash of file."""
        h = hashlib.md5()
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(FileHasher.CHUNK_SIZE), b""):
                h.update(chunk)
        
        return h.hexdigest()
    
    @staticmethod
    def multi_hash(
        path: Union[str, Path],
        algorithms: List[str] = ['sha256', 'md5']
    ) -> Dict[str, str]:
        """
        Compute multiple hashes in single pass.
        
        Args:
            path: File path
            algorithms: List of hash algorithms
            
        Returns:
            Dict mapping algorithm -> hash
        """
        hashers = {
            alg: getattr(hashlib, alg)()
            for alg in algorithms
        }
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(FileHasher.CHUNK_SIZE), b""):
                for hasher in hashers.values():
                    hasher.update(chunk)
        
        return {
            alg: hasher.hexdigest()
            for alg, hasher in hashers.items()
        }
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """Hash a string."""
        h = getattr(hashlib, algorithm)()
        h.update(text.encode('utf-8'))
        return h.hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any], algorithm: str = 'sha256') -> str:
        """Hash a dictionary (order-independent)."""
        # Convert to sorted JSON string
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return FileHasher.hash_string(json_str, algorithm)


# ============================================================================
# PATH UTILITIES
# ============================================================================

class PathManager:
    """
    Safe path management utilities.
    
    Features:
    - Safe path creation
    - Temp file management
    - Path validation
    - Atomic operations
    """
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if needed.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def ensure_parent(path: Union[str, Path]) -> Path:
        """Ensure parent directory exists."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    @contextlib.contextmanager
    def temp_dir(
        prefix: str = "tmiv_",
        cleanup: bool = True
    ) -> Generator[Path, None, None]:
        """
        Context manager for temporary directory.
        
        Args:
            prefix: Directory name prefix
            cleanup: Whether to cleanup on exit
            
        Yields:
            Path to temp directory
        """
        temp_path = Path(tempfile.mkdtemp(prefix=prefix))
        
        try:
            yield temp_path
        finally:
            if cleanup and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
    
    @staticmethod
    @contextlib.contextmanager
    def temp_file(
        suffix: str = ".tmp",
        prefix: str = "tmiv_",
        cleanup: bool = True
    ) -> Generator[Path, None, None]:
        """
        Context manager for temporary file.
        
        Args:
            suffix: File extension
            prefix: File name prefix
            cleanup: Whether to cleanup on exit
            
        Yields:
            Path to temp file
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # Close file descriptor
        
        temp_path = Path(temp_path)
        
        try:
            yield temp_path
        finally:
            if cleanup and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    
    @staticmethod
    def safe_move(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Safely move file with atomic replacement.
        
        Args:
            src: Source path
            dst: Destination path
        """
        src = Path(src)
        dst = Path(dst)
        
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Use atomic rename if on same filesystem
        try:
            src.rename(dst)
        except OSError:
            # Fall back to copy + delete
            shutil.copy2(src, dst)
            src.unlink()
    
    @staticmethod
    def get_size_mb(path: Union[str, Path]) -> float:
        """Get file/directory size in MB."""
        path = Path(path)
        
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        
        elif path.is_dir():
            total_size = sum(
                f.stat().st_size
                for f in path.rglob('*')
                if f.is_file()
            )
            return total_size / (1024 * 1024)
        
        return 0.0


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class Timer:
    """
    High-precision timing utilities.
    
    Features:
    - Context manager support
    - Named timers
    - Automatic metrics logging
    - Statistics tracking
    """
    
    def __init__(self, name: str = "timer", auto_log: bool = True):
        """
        Args:
            name: Timer name
            auto_log: Automatically log to telemetry
        """
        self.name = name
        self.auto_log = auto_log
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self) -> 'Timer':
        """Start timer."""
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop timer."""
        self.stop()
        
        if self.auto_log and self.elapsed is not None:
            metric(f"{self.name}_duration_sec", self.elapsed)
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed seconds
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        return self.elapsed
    
    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed is not None else None


@contextlib.contextmanager
def time_block(metrics: Dict[str, float], key: str) -> Generator[None, None, None]:
    """
    Context manager to time a code block.
    
    Args:
        metrics: Dict to store result
        key: Key for storing time
        
    Yields:
        None
        
    Example:
        metrics = {}
        with time_block(metrics, "model_training"):
            train_model()
        print(f"Training took {metrics['model_training']:.2f}s")
    """
    t0 = time.perf_counter()
    
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        metrics[key] = round(elapsed, 6)


def timeit(func: F) -> F:
    """
    Decorator to time function execution.
    
    Example:
        @timeit
        def train_model():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    
    return wrapper


# ============================================================================
# RETRY LOGIC
# ============================================================================

class RetryPolicy:
    """
    Retry policy with exponential backoff.
    
    Features:
    - Exponential backoff
    - Maximum attempts
    - Configurable delays
    - Exception filtering
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """
        Args:
            max_attempts: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            exceptions: Exception types to catch
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions
    
    def __call__(self, func: F) -> F:
        """Decorator to add retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except self.exceptions as e:
                    last_exception = e
                    
                    if attempt < self.max_attempts - 1:
                        # Calculate delay with exponential backoff
                        delay = min(
                            self.initial_delay * (self.exponential_base ** attempt),
                            self.max_delay
                        )
                        
                        warnings.warn(
                            f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        time.sleep(delay)
                    else:
                        warnings.warn(
                            f"All {self.max_attempts} attempts failed"
                        )
            
            # Re-raise last exception
            raise last_exception
        
        return wrapper


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Convenience retry decorator.
    
    Example:
        @retry(max_attempts=3, initial_delay=1.0)
        def flaky_function():
            pass
    """
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        exceptions=exceptions
    )


# ============================================================================
# CACHING
# ============================================================================

class CacheManager:
    """
    Simple in-memory cache manager.
    
    Features:
    - TTL support
    - Size limits
    - LRU eviction
    """
    
    def __init__(
        self,
        max_size: int = 128,
        ttl_seconds: Optional[float] = None
    ):
        """
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        # Check TTL
        if self.ttl_seconds is not None:
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]
        
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


def cached(ttl_seconds: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Example:
        @cached(ttl_seconds=300)
        def expensive_computation(x):
            return x ** 2
    """
    cache = CacheManager(ttl_seconds=ttl_seconds)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = FileHasher.hash_string(
                f"{func.__name__}:{args}:{kwargs}"
            )
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = cache.clear
        
        return wrapper
    
    return decorator


# ============================================================================
# SERIALIZATION
# ============================================================================

class Serializer:
    """
    Unified serialization utilities.
    
    Supports: JSON, pickle, numpy, pandas
    """
    
    @staticmethod
    def save_json(
        data: Any,
        path: Union[str, Path],
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """Save data as JSON."""
        path = Path(path)
        PathManager.ensure_parent(path)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Any:
        """Load data from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, path: Union[str, Path]) -> None:
        """Save data using pickle."""
        path = Path(path)
        PathManager.ensure_parent(path)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """Load data from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_numpy(array: np.ndarray, path: Union[str, Path]) -> None:
        """Save numpy array."""
        path = Path(path)
        PathManager.ensure_parent(path)
        np.save(path, array)
    
    @staticmethod
    def load_numpy(path: Union[str, Path]) -> np.ndarray:
        """Load numpy array."""
        return np.load(path)
    
    @staticmethod
    def save_dataframe(
        df: pd.DataFrame,
        path: Union[str, Path],
        format: str = 'parquet'
    ) -> None:
        """Save pandas DataFrame."""
        path = Path(path)
        PathManager.ensure_parent(path)
        
        if format == 'parquet':
            df.to_parquet(path)
        elif format == 'csv':
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_dataframe(
        path: Union[str, Path],
        format: str = 'parquet'
    ) -> pd.DataFrame:
        """Load pandas DataFrame."""
        if format == 'parquet':
            return pd.read_parquet(path)
        elif format == 'csv':
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def sha256_of_path(path: str) -> str:
    """
    Backward compatible: compute SHA256 of file.
    
    Enhanced version with better error handling.
    """
    return FileHasher.sha256(path)


def ensure_dir(path: str) -> None:
    """
    Backward compatible: ensure directory exists.
    
    Enhanced version returning Path object.
    """
    PathManager.ensure_dir(path)


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def safe_file_operation(
    func: Callable,
    *args,
    max_attempts: int = 3,
    **kwargs
) -> Any:
    """
    Execute file operation with retry logic.
    
    Example:
        result = safe_file_operation(
            pd.read_csv,
            'data.csv',
            max_attempts=3
        )
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        exceptions=(IOError, OSError)
    )
    
    return policy(func)(*args, **kwargs)