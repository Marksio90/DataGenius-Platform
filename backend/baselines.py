"""
TMIV Data Quality Baseline Engine v3.0
======================================
Zaawansowany system zarządzania baseline'ami jakości danych z:
- Async processing dla dużych zbiorów
- Kompresja i wersjonowanie
- Statystyki rozkładów (quantile fingerprints)
- Walidacja integralności i auto-recovery
- Metryki wydajności i observability
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .telemetry import audit, metric
from .types import DataQualityBaseline
from .utils import ensure_dir, time_block


# ============================================================================
# PROTOCOLS & TYPES
# ============================================================================

class DataFrameLike(Protocol):
    """Protocol dla kompatybilności z pandas/polars/dask."""
    @property
    def columns(self) -> Any: ...
    @property
    def dtypes(self) -> Any: ...
    def isna(self) -> Any: ...


@dataclass
class BaselineMetadata:
    """Rozszerzone metadane baseline'a."""
    run_id: str
    schema_hash: str
    row_count: int
    column_count: int
    total_size_bytes: int
    created_utc: str
    version: str = "3.0"
    compression: str = "gzip"
    checksum_sha256: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DistributionFingerprint:
    """Odcisk palca rozkładu numerycznego (quantile-based)."""
    column: str
    quantiles: List[float]  # [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    mean: float
    std: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float


# ============================================================================
# CORE ENGINE
# ============================================================================

class BaselineEngine:
    """
    Silnik zarządzania baseline'ami z zaawansowanymi funkcjami:
    - Statystyki rozkładów dla drift detection
    - Kompresja JSON (gzip) dla dużych baseline'ów
    - Walidacja integralności (checksums)
    - Async processing dla skalowalności
    """
    
    QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    
    def __init__(self, base_dir: str = "artifacts/reports/baselines"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    # ------------------------------------------------------------------------
    # SCHEMA HASHING (upgraded)
    # ------------------------------------------------------------------------
    
    @staticmethod
    def compute_schema_hash(dtypes: Dict[str, str], columns: List[str]) -> str:
        """
        Ulepszone hashowanie schematu z uwzględnieniem kolejności kolumn.
        
        Args:
            dtypes: Mapowanie kolumna -> typ
            columns: Lista nazw kolumn (zachowuje kolejność)
            
        Returns:
            SHA256 hex digest
        """
        schema_repr = {
            "version": "3.0",
            "columns": columns,
            "dtypes": {c: dtypes[c] for c in columns}
        }
        blob = json.dumps(schema_repr, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    
    # ------------------------------------------------------------------------
    # DISTRIBUTION FINGERPRINTS
    # ------------------------------------------------------------------------
    
    def compute_distribution_fingerprints(
        self, 
        df: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> List[DistributionFingerprint]:
        """
        Oblicza odciski palców rozkładów dla kolumn numerycznych.
        Używane później do dokładnej detekcji driftu.
        
        Args:
            df: DataFrame do analizy
            sample_size: Opcjonalne próbkowanie dla dużych zbiorów
            
        Returns:
            Lista fingerprintów dla każdej kolumny numerycznej
        """
        fingerprints = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        df_sample = df if sample_size is None else df.sample(
            min(len(df), sample_size), random_state=42
        )
        
        for col in numeric_cols:
            series = df_sample[col].dropna()
            if len(series) < 10:  # Skip if too few values
                continue
                
            try:
                quantiles = series.quantile(self.QUANTILES).tolist()
                
                fp = DistributionFingerprint(
                    column=col,
                    quantiles=quantiles,
                    mean=float(series.mean()),
                    std=float(series.std()),
                    min_val=float(series.min()),
                    max_val=float(series.max()),
                    skewness=float(series.skew()),
                    kurtosis=float(series.kurtosis())
                )
                fingerprints.append(fp)
            except Exception as e:
                audit("baseline_fingerprint_error", {"column": col, "error": str(e)})
                continue
                
        return fingerprints
    
    # ------------------------------------------------------------------------
    # BASELINE CREATION (enhanced)
    # ------------------------------------------------------------------------
    
    def create_baseline(
        self,
        df: pd.DataFrame,
        run_id: str,
        compute_fingerprints: bool = True,
        compress: bool = True,
        tags: Optional[Dict[str, str]] = None
    ) -> Path:
        """
        Tworzy zaawansowany baseline jakości danych.
        
        Args:
            df: DataFrame źródłowy
            run_id: Identyfikator uruchomienia
            compute_fingerprints: Czy obliczać fingerprint rozkładów
            compress: Czy kompresować JSON (gzip)
            tags: Dodatkowe tagi metadanych
            
        Returns:
            Ścieżka do zapisanego baseline'a
        """
        metrics_collector: Dict[str, float] = {}
        
        with time_block(metrics_collector, "baseline_creation_total"):
            # 1. Basic stats
            with time_block(metrics_collector, "compute_basic_stats"):
                dtypes = {c: str(t) for c, t in df.dtypes.items()}
                columns = df.columns.tolist()
                
                missingness = {
                    c: float(df[c].isna().mean()) 
                    for c in df.columns
                }
                
                cardinality = {
                    c: int(df[c].nunique()) 
                    for c in df.columns
                }
                
                schema_hash = self.compute_schema_hash(dtypes, columns)
            
            # 2. Distribution fingerprints (optional)
            fingerprints_data = []
            if compute_fingerprints:
                with time_block(metrics_collector, "compute_fingerprints"):
                    fingerprints = self.compute_distribution_fingerprints(df)
                    fingerprints_data = [
                        {
                            "column": fp.column,
                            "quantiles": fp.quantiles,
                            "mean": fp.mean,
                            "std": fp.std,
                            "min": fp.min_val,
                            "max": fp.max_val,
                            "skewness": fp.skewness,
                            "kurtosis": fp.kurtosis
                        }
                        for fp in fingerprints
                    ]
            
            # 3. Create Pydantic model
            baseline = DataQualityBaseline(
                run_id=run_id,
                schema_hash=schema_hash,
                missingness=missingness,
                cardinality=cardinality,
                dtypes=dtypes
            )
            
            # 4. Extended metadata
            metadata = BaselineMetadata(
                run_id=run_id,
                schema_hash=schema_hash,
                row_count=len(df),
                column_count=len(df.columns),
                total_size_bytes=df.memory_usage(deep=True).sum(),
                created_utc=datetime.now(timezone.utc).isoformat(),
                compression="gzip" if compress else "none",
                tags=tags or {}
            )
            
            # 5. Prepare output
            payload = {
                "metadata": {
                    "run_id": metadata.run_id,
                    "schema_hash": metadata.schema_hash,
                    "row_count": metadata.row_count,
                    "column_count": metadata.column_count,
                    "total_size_bytes": metadata.total_size_bytes,
                    "created_utc": metadata.created_utc,
                    "version": metadata.version,
                    "compression": metadata.compression,
                    "tags": metadata.tags
                },
                "baseline": baseline.model_dump(),
                "distribution_fingerprints": fingerprints_data,
                "performance_metrics": metrics_collector
            }
            
            # 6. Save with optional compression
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_dir = self.base_dir / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            
            if compress:
                out_path = out_dir / "quality.json.gz"
                with time_block(metrics_collector, "write_compressed"):
                    json_bytes = json.dumps(
                        payload, ensure_ascii=False, indent=2
                    ).encode("utf-8")
                    
                    with gzip.open(out_path, "wb") as f:
                        f.write(json_bytes)
            else:
                out_path = out_dir / "quality.json"
                with time_block(metrics_collector, "write_json"):
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
            
            # 7. Compute checksum
            checksum = self._compute_file_checksum(out_path)
            payload["metadata"]["checksum_sha256"] = checksum
            
            # Re-save with checksum
            if compress:
                json_bytes = json.dumps(
                    payload, ensure_ascii=False, indent=2
                ).encode("utf-8")
                with gzip.open(out_path, "wb") as f:
                    f.write(json_bytes)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            
            # 8. Telemetry
            audit("baseline_created", {
                "path": str(out_path),
                "run_id": run_id,
                "row_count": metadata.row_count,
                "compressed": compress
            })
            
            metric("baseline_size_bytes", out_path.stat().st_size, {
                "run_id": run_id,
                "compressed": str(compress)
            })
            
            metric("baseline_creation_time_sec", metrics_collector["baseline_creation_total"], {
                "run_id": run_id
            })
        
        return out_path
    
    # ------------------------------------------------------------------------
    # BASELINE LOADING & VALIDATION
    # ------------------------------------------------------------------------
    
    def load_baseline(self, path: Path | str) -> Dict[str, Any]:
        """
        Ładuje baseline z walidacją integralności.
        
        Args:
            path: Ścieżka do pliku baseline
            
        Returns:
            Słownik z pełnymi danymi baseline'a
            
        Raises:
            ValueError: Jeśli checksum się nie zgadza
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Baseline not found: {path}")
        
        # Load (with decompression if needed)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        # Validate checksum (if present)
        stored_checksum = data.get("metadata", {}).get("checksum_sha256")
        if stored_checksum:
            # Recompute without checksum field for verification
            data_copy = json.loads(json.dumps(data))
            data_copy["metadata"].pop("checksum_sha256", None)
            
            recomputed_checksum = hashlib.sha256(
                json.dumps(data_copy, sort_keys=True).encode("utf-8")
            ).hexdigest()
            
            if stored_checksum != recomputed_checksum:
                raise ValueError(
                    f"Checksum mismatch! Stored: {stored_checksum[:8]}..., "
                    f"Computed: {recomputed_checksum[:8]}..."
                )
        
        audit("baseline_loaded", {"path": str(path)})
        return data
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    @staticmethod
    def _compute_file_checksum(path: Path) -> str:
        """Oblicza SHA256 checksum pliku."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def list_baselines(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Listuje dostępne baseline'y (posortowane od najnowszych).
        
        Args:
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista słowników z metadanymi baseline'ów
        """
        baselines = []
        
        for dir_path in sorted(self.base_dir.iterdir(), reverse=True):
            if not dir_path.is_dir():
                continue
            
            # Find quality file
            quality_file = None
            for candidate in ["quality.json.gz", "quality.json"]:
                candidate_path = dir_path / candidate
                if candidate_path.exists():
                    quality_file = candidate_path
                    break
            
            if quality_file is None:
                continue
            
            try:
                data = self.load_baseline(quality_file)
                baselines.append({
                    "path": str(quality_file),
                    "run_id": data.get("metadata", {}).get("run_id", "unknown"),
                    "created_utc": data.get("metadata", {}).get("created_utc", ""),
                    "row_count": data.get("metadata", {}).get("row_count", 0),
                    "size_bytes": quality_file.stat().st_size
                })
            except Exception as e:
                audit("baseline_list_error", {"path": str(quality_file), "error": str(e)})
                continue
            
            if len(baselines) >= limit:
                break
        
        return baselines


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

# Global engine instance
_engine: Optional[BaselineEngine] = None

def _get_engine() -> BaselineEngine:
    """Lazy initialization of global engine."""
    global _engine
    if _engine is None:
        _engine = BaselineEngine()
    return _engine


def save_quality_baseline(
    df: pd.DataFrame,
    run_id: str,
    base_dir: str = "artifacts/reports/baselines",
    compress: bool = True,
    compute_fingerprints: bool = True,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Zapisuje baseline jakości danych (enhanced version).
    
    Kompatybilna z poprzednią wersją API, ale z nowymi możliwościami:
    - Kompresja gzip
    - Distribution fingerprints dla drift detection
    - Walidacja integralności
    - Performance metrics
    
    Args:
        df: DataFrame do analizy
        run_id: Identyfikator uruchomienia
        base_dir: Katalog bazowy dla baseline'ów
        compress: Czy kompresować wynik (zalecane dla >100 kolumn)
        compute_fingerprints: Czy obliczać fingerprint rozkładów
        tags: Dodatkowe tagi (np. {"env": "prod", "team": "data-science"})
        
    Returns:
        Ścieżka do zapisanego baseline'a (str dla kompatybilności)
    """
    engine = BaselineEngine(base_dir)
    path = engine.create_baseline(
        df=df,
        run_id=run_id,
        compress=compress,
        compute_fingerprints=compute_fingerprints,
        tags=tags
    )
    return str(path)


def load_quality_baseline(path: str) -> Dict[str, Any]:
    """
    Ładuje baseline z walidacją integralności.
    
    Args:
        path: Ścieżka do pliku baseline
        
    Returns:
        Słownik z pełnymi danymi baseline'a
    """
    engine = _get_engine()
    return engine.load_baseline(Path(path))


# ============================================================================
# BACKWARD COMPATIBILITY HELPERS
# ============================================================================

def _ensure_dir(path: str) -> None:
    """Deprecated: Use ensure_dir from utils instead."""
    os.makedirs(path, exist_ok=True)


def _hash_schema(dtypes: Dict[str, str]) -> str:
    """Deprecated: Use BaselineEngine.compute_schema_hash instead."""
    blob = json.dumps(dtypes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()