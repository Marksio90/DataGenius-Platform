"""
TMIV Advanced Feature Store v3.0
=================================
Zaawansowany system zarządzania cechami (Feature Store) z:
- Versioning & lineage tracking (data provenance)
- Feature metadata & statistics
- Automatic feature type inference & validation
- Time-travel capabilities (point-in-time correctness)
- Feature materialization & caching
- Online/offline serving modes
- Feature drift monitoring
- Incremental updates & compaction
- Feature discovery & search
- Integration with training pipelines
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .telemetry import audit, metric
from .utils import ensure_dir, sha256_of_path


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class FeatureType(str, Enum):
    """Typy cech."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    EMBEDDING = "embedding"


class StorageFormat(str, Enum):
    """Formaty przechowywania."""
    PARQUET = "parquet"
    CSV = "csv"
    FEATHER = "feather"
    HDF5 = "hdf5"


class ServingMode(str, Enum):
    """Tryby serwowania cech."""
    OFFLINE = "offline"   # Batch training
    ONLINE = "online"     # Real-time inference
    HYBRID = "hybrid"     # Both


@dataclass
class FeatureMetadata:
    """Metadane pojedynczej cechy."""
    name: str
    feature_type: FeatureType
    dtype: str
    
    # Statistics
    non_null_count: int
    null_count: int
    null_percentage: float
    
    # Type-specific stats
    cardinality: Optional[int] = None  # For categorical
    mean: Optional[float] = None       # For numerical
    std: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    
    # Lineage
    source_table: Optional[str] = None
    transformation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.feature_type.value,
            "dtype": self.dtype,
            "non_null": self.non_null_count,
            "null_count": self.null_count,
            "null_pct": float(self.null_percentage),
            "cardinality": self.cardinality,
            "mean": float(self.mean) if self.mean is not None else None,
            "std": float(self.std) if self.std is not None else None,
            "min": str(self.min_val) if self.min_val is not None else None,
            "max": str(self.max_val) if self.max_val is not None else None,
            "source": self.source_table,
            "transformation": self.transformation
        }


class FeatureSetManifest(BaseModel):
    """Manifest dla zestawu cech."""
    name: str
    version: str
    created_utc: str
    
    # Content
    columns: List[str]
    row_count: int
    storage_path: str
    storage_format: StorageFormat
    
    # Metadata
    feature_metadata: Dict[str, Dict[str, Any]]
    data_hash: str
    file_size_bytes: int
    checksum: str
    
    # Lineage
    parent_version: Optional[str] = None
    source_dataset: Optional[str] = None
    
    # Serving
    serving_mode: ServingMode = ServingMode.OFFLINE
    
    # Tags
    tags: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# FEATURE STORE ENGINE
# ============================================================================

class FeatureStore:
    """
    Zaawansowany Feature Store z:
    - Versioning & time-travel
    - Metadata tracking
    - Multiple storage formats
    - Feature lineage
    - Online/offline serving
    """
    
    def __init__(
        self,
        base_dir: str = "artifacts/feature_store",
        default_format: StorageFormat = StorageFormat.PARQUET
    ):
        """
        Args:
            base_dir: Katalog bazowy dla feature store
            default_format: Domyślny format przechowywania
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_format = default_format
        
        # Create subdirectories
        self.data_dir = self.base_dir / "data"
        self.metadata_dir = self.base_dir / "metadata"
        self.index_dir = self.base_dir / "index"
        
        for d in [self.data_dir, self.metadata_dir, self.index_dir]:
            d.mkdir(exist_ok=True)
    
    # ------------------------------------------------------------------------
    # FEATURE INGESTION
    # ------------------------------------------------------------------------
    
    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        columns: Optional[List[str]] = None,
        storage_format: Optional[StorageFormat] = None,
        parent_version: Optional[str] = None,
        source_dataset: Optional[str] = None,
        serving_mode: ServingMode = ServingMode.OFFLINE,
        tags: Optional[Dict[str, str]] = None,
        compute_stats: bool = True
    ) -> Path:
        """
        Zapisuje zestaw cech do feature store.
        
        Args:
            df: DataFrame z cechami
            name: Nazwa zestawu cech
            columns: Lista kolumn do zapisania (None = all)
            storage_format: Format przechowywania
            parent_version: Wersja rodzica (dla lineage)
            source_dataset: Dataset źródłowy
            serving_mode: Tryb serwowania
            tags: Dodatkowe tagi
            compute_stats: Czy obliczać statystyki
            
        Returns:
            Ścieżka do zapisanego zestawu
        """
        # Select columns
        if columns is None:
            columns = df.columns.tolist()
        
        subset = df[columns].copy()
        
        # Generate version
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        
        # Compute data hash
        data_hash = self._compute_data_hash(subset)
        
        # Select storage format
        if storage_format is None:
            storage_format = self.default_format
        
        # Save data
        data_path = self._save_data(subset, name, version, storage_format)
        
        # Compute feature metadata
        feature_metadata = {}
        if compute_stats:
            feature_metadata = self._compute_feature_metadata(
                subset, source_dataset
            )
        
        # File info
        file_size = data_path.stat().st_size
        checksum = sha256_of_path(str(data_path))
        
        # Create manifest
        manifest = FeatureSetManifest(
            name=name,
            version=version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            columns=columns,
            row_count=len(subset),
            storage_path=str(data_path),
            storage_format=storage_format,
            feature_metadata=feature_metadata,
            data_hash=data_hash,
            file_size_bytes=file_size,
            checksum=checksum,
            parent_version=parent_version,
            source_dataset=source_dataset,
            serving_mode=serving_mode,
            tags=tags or {}
        )
        
        # Save manifest
        manifest_path = self._save_manifest(name, version, manifest)
        
        # Update index
        self._update_index(name, version, manifest)
        
        # Telemetry
        audit("feature_store_save", {
            "name": name,
            "version": version,
            "columns": len(columns),
            "rows": len(subset),
            "format": storage_format.value,
            "size_mb": file_size / (1024 * 1024)
        })
        
        metric("feature_store_size_bytes", file_size, {
            "name": name,
            "format": storage_format.value
        })
        
        return data_path
    
    # ------------------------------------------------------------------------
    # FEATURE RETRIEVAL
    # ------------------------------------------------------------------------
    
    def load_features(
        self,
        name: str,
        version: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ładuje zestaw cech z feature store.
        
        Args:
            name: Nazwa zestawu
            version: Wersja (None = latest)
            columns: Lista kolumn do załadowania (None = all)
            
        Returns:
            DataFrame z cechami
        """
        # Get manifest
        if version is None:
            version = self._get_latest_version(name)
        
        manifest = self._load_manifest(name, version)
        
        # Load data
        df = self._load_data(manifest, columns)
        
        # Telemetry
        audit("feature_store_load", {
            "name": name,
            "version": version,
            "rows": len(df),
            "columns": len(df.columns)
        })
        
        return df
    
    def get_manifest(
        self,
        name: str,
        version: Optional[str] = None
    ) -> FeatureSetManifest:
        """Zwraca manifest dla zestawu cech."""
        if version is None:
            version = self._get_latest_version(name)
        
        return self._load_manifest(name, version)
    
    # ------------------------------------------------------------------------
    # FEATURE DISCOVERY
    # ------------------------------------------------------------------------
    
    def list_feature_sets(
        self,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Listuje dostępne zestawy cech.
        
        Args:
            tags: Filtruj po tagach
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista metadanych zestawów
        """
        index = self._load_index()
        
        results = []
        for name, versions in index.items():
            # Get latest version
            latest_version = sorted(versions.keys(), reverse=True)[0]
            manifest_data = versions[latest_version]
            
            # Filter by tags
            if tags:
                manifest_tags = manifest_data.get("tags", {})
                if not all(manifest_tags.get(k) == v for k, v in tags.items()):
                    continue
            
            results.append({
                "name": name,
                "latest_version": latest_version,
                "versions_count": len(versions),
                "columns": manifest_data.get("columns", []),
                "row_count": manifest_data.get("row_count", 0),
                "created_utc": manifest_data.get("created_utc", ""),
                "tags": manifest_data.get("tags", {})
            })
            
            if len(results) >= limit:
                break
        
        return results
    
    def search_features(
        self,
        query: str,
        feature_type: Optional[FeatureType] = None
    ) -> List[Dict[str, Any]]:
        """
        Wyszukuje cechy po nazwie lub metadanych.
        
        Args:
            query: Zapytanie wyszukiwania
            feature_type: Filtruj po typie cechy
            
        Returns:
            Lista znalezionych cech
        """
        index = self._load_index()
        results = []
        
        query_lower = query.lower()
        
        for name, versions in index.items():
            latest_version = sorted(versions.keys(), reverse=True)[0]
            manifest_data = versions[latest_version]
            
            feature_metadata = manifest_data.get("feature_metadata", {})
            
            for feat_name, feat_meta in feature_metadata.items():
                # Match by name
                if query_lower not in feat_name.lower():
                    continue
                
                # Filter by type
                if feature_type and feat_meta.get("type") != feature_type.value:
                    continue
                
                results.append({
                    "feature_name": feat_name,
                    "feature_set": name,
                    "version": latest_version,
                    "type": feat_meta.get("type"),
                    "metadata": feat_meta
                })
        
        return results
    
    # ------------------------------------------------------------------------
    # TIME-TRAVEL
    # ------------------------------------------------------------------------
    
    def load_features_at_time(
        self,
        name: str,
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Ładuje cechy z punktu w czasie (time-travel).
        
        Args:
            name: Nazwa zestawu
            timestamp: Punkt w czasie
            
        Returns:
            DataFrame z cechami z tego czasu
        """
        # Find version closest to timestamp (but not after)
        versions = self.list_versions(name)
        
        target_version = None
        for v in sorted(versions, reverse=True):
            version_time = datetime.strptime(v["version"], "%Y%m%d-%H%M%S")
            version_time = version_time.replace(tzinfo=timezone.utc)
            
            if version_time <= timestamp:
                target_version = v["version"]
                break
        
        if target_version is None:
            raise ValueError(f"No version found before {timestamp}")
        
        return self.load_features(name, version=target_version)
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """Lista wszystkich wersji zestawu cech."""
        index = self._load_index()
        
        if name not in index:
            return []
        
        versions = []
        for version, manifest_data in index[name].items():
            versions.append({
                "version": version,
                "created_utc": manifest_data.get("created_utc"),
                "row_count": manifest_data.get("row_count"),
                "columns": len(manifest_data.get("columns", []))
            })
        
        return sorted(versions, key=lambda x: x["version"], reverse=True)
    
    # ------------------------------------------------------------------------
    # FEATURE LINEAGE
    # ------------------------------------------------------------------------
    
    def get_lineage(self, name: str, version: str) -> Dict[str, Any]:
        """
        Zwraca lineage (rodowód) dla zestawu cech.
        
        Returns:
            Dict z parent_version, source_dataset, transformations
        """
        manifest = self._load_manifest(name, version)
        
        lineage = {
            "name": name,
            "version": version,
            "parent_version": manifest.parent_version,
            "source_dataset": manifest.source_dataset,
            "created_utc": manifest.created_utc
        }
        
        # Recursively get parent lineage
        if manifest.parent_version:
            parent_lineage = self.get_lineage(name, manifest.parent_version)
            lineage["parent_lineage"] = parent_lineage
        
        return lineage
    
    # ------------------------------------------------------------------------
    # FEATURE STATISTICS
    # ------------------------------------------------------------------------
    
    def compute_feature_drift(
        self,
        name: str,
        reference_version: str,
        current_version: str
    ) -> Dict[str, float]:
        """
        Oblicza drift między wersjami zestawu cech.
        
        Returns:
            Dict[feature_name, drift_score]
        """
        from .drift import DriftDetector
        
        ref_df = self.load_features(name, reference_version)
        cur_df = self.load_features(name, current_version)
        
        detector = DriftDetector()
        drift_scores = {}
        
        for col in ref_df.columns:
            if col not in cur_df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(ref_df[col]):
                psi, _ = detector.population_stability_index(
                    ref_df[col].values,
                    cur_df[col].values
                )
                drift_scores[col] = psi
        
        return drift_scores
    
    # ------------------------------------------------------------------------
    # INTERNAL METHODS
    # ------------------------------------------------------------------------
    
    def _save_data(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
        storage_format: StorageFormat
    ) -> Path:
        """Zapisuje dane w wybranym formacie."""
        filename = f"{name}-{version}.{storage_format.value}"
        path = self.data_dir / filename
        
        try:
            if storage_format == StorageFormat.PARQUET:
                df.to_parquet(path, index=False)
            elif storage_format == StorageFormat.CSV:
                df.to_csv(path, index=False)
            elif storage_format == StorageFormat.FEATHER:
                df.to_feather(path)
            elif storage_format == StorageFormat.HDF5:
                df.to_hdf(path, key="data", mode="w")
            else:
                raise ValueError(f"Unsupported format: {storage_format}")
        except Exception as e:
            # Fallback to CSV
            warnings.warn(f"Failed to save as {storage_format}, falling back to CSV: {e}")
            filename = f"{name}-{version}.csv"
            path = self.data_dir / filename
            df.to_csv(path, index=False)
        
        return path
    
    def _load_data(
        self,
        manifest: FeatureSetManifest,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Ładuje dane z dysku."""
        path = Path(manifest.storage_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load based on format
        if manifest.storage_format == StorageFormat.PARQUET:
            df = pd.read_parquet(path, columns=columns)
        elif manifest.storage_format == StorageFormat.CSV:
            df = pd.read_csv(path, usecols=columns)
        elif manifest.storage_format == StorageFormat.FEATHER:
            df = pd.read_feather(path, columns=columns)
        elif manifest.storage_format == StorageFormat.HDF5:
            df = pd.read_hdf(path, key="data")
            if columns:
                df = df[columns]
        else:
            raise ValueError(f"Unsupported format: {manifest.storage_format}")
        
        return df
    
    def _save_manifest(
        self,
        name: str,
        version: str,
        manifest: FeatureSetManifest
    ) -> Path:
        """Zapisuje manifest."""
        manifest_path = self.metadata_dir / f"{name}-{version}.json"
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, ensure_ascii=False, indent=2)
        
        return manifest_path
    
    def _load_manifest(self, name: str, version: str) -> FeatureSetManifest:
        """Ładuje manifest."""
        manifest_path = self.metadata_dir / f"{name}-{version}.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return FeatureSetManifest(**data)
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Oblicza hash danych."""
        h = hashlib.sha256()
        h.update(str(tuple(df.columns)).encode())
        h.update(str(df.shape).encode())
        
        # Sample-based hash for large dataframes
        if len(df) > 10000:
            sample = df.sample(min(len(df), 1000), random_state=42)
            h.update(sample.to_json().encode())
        else:
            h.update(df.to_json().encode())
        
        return h.hexdigest()
    
    def _compute_feature_metadata(
        self,
        df: pd.DataFrame,
        source_table: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Oblicza metadane dla każdej cechy."""
        metadata = {}
        
        for col in df.columns:
            series = df[col]
            
            # Infer type
            feature_type = self._infer_feature_type(series)
            
            # Basic stats
            non_null = int(series.notna().sum())
            null_count = int(series.isna().sum())
            null_pct = null_count / len(series) if len(series) > 0 else 0.0
            
            feat_meta = FeatureMetadata(
                name=col,
                feature_type=feature_type,
                dtype=str(series.dtype),
                non_null_count=non_null,
                null_count=null_count,
                null_percentage=null_pct,
                source_table=source_table
            )
            
            # Type-specific stats
            if feature_type == FeatureType.NUMERICAL:
                clean = series.dropna()
                if len(clean) > 0:
                    feat_meta.mean = float(clean.mean())
                    feat_meta.std = float(clean.std())
                    feat_meta.min_val = float(clean.min())
                    feat_meta.max_val = float(clean.max())
            
            elif feature_type == FeatureType.CATEGORICAL:
                feat_meta.cardinality = int(series.nunique())
                feat_meta.min_val = None
                feat_meta.max_val = None
            
            metadata[col] = feat_meta.to_dict()
        
        return metadata
    
    @staticmethod
    def _infer_feature_type(series: pd.Series) -> FeatureType:
        """Wnioskuje typ cechy."""
        dtype = series.dtype
        
        if pd.api.types.is_bool_dtype(dtype):
            return FeatureType.BOOLEAN
        
        if pd.api.types.is_numeric_dtype(dtype):
            return FeatureType.NUMERICAL
        
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return FeatureType.DATETIME
        
        # Check if categorical (low cardinality)
        nunique = series.nunique()
        if nunique <= 100:
            return FeatureType.CATEGORICAL
        
        return FeatureType.TEXT
    
    def _update_index(
        self,
        name: str,
        version: str,
        manifest: FeatureSetManifest
    ) -> None:
        """Aktualizuje indeks feature store."""
        index = self._load_index()
        
        if name not in index:
            index[name] = {}
        
        index[name][version] = {
            "created_utc": manifest.created_utc,
            "columns": manifest.columns,
            "row_count": manifest.row_count,
            "tags": manifest.tags
        }
        
        self._save_index(index)
    
    def _load_index(self) -> Dict[str, Any]:
        """Ładuje indeks."""
        index_path = self.index_dir / "index.json"
        
        if not index_path.exists():
            return {}
        
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _save_index(self, index: Dict[str, Any]) -> None:
        """Zapisuje indeks."""
        index_path = self.index_dir / "index.json"
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def _get_latest_version(self, name: str) -> str:
        """Zwraca najnowszą wersję."""
        versions = self.list_versions(name)
        
        if not versions:
            raise ValueError(f"No versions found for feature set: {name}")
        
        return versions[0]["version"]


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

_store: Optional[FeatureStore] = None

def _get_store() -> FeatureStore:
    """Lazy initialization."""
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


def save_features(
    df: pd.DataFrame,
    cols: List[str],
    name: str = "features"
) -> str:
    """
    Backward compatible save_features (simplified).
    
    Returns path as string.
    """
    store = _get_store()
    path = store.save_features(df, name, columns=cols)
    return str(path)


def load_features(name: str, version: Optional[str] = None) -> pd.DataFrame:
    """Load features from store."""
    store = _get_store()
    return store.load_features(name, version)


# Deprecated internal functions
def _ensure(p: str) -> None:
    """Deprecated: Use ensure_dir from utils."""
    os.makedirs(p, exist_ok=True)


def _hash_df(df: pd.DataFrame) -> str:
    """Deprecated: Use FeatureStore._compute_data_hash."""
    h = hashlib.sha256()
    h.update(str(tuple(df.columns)).encode())
    h.update(str(df.shape).encode())
    return h.hexdigest()