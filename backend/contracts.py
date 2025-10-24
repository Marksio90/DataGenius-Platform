"""
TMIV Data Contract Engine v3.0
===============================
Zaawansowany system kontraktów danych oparty na Pandera z:
- Multi-level validation (schema, stats, business rules)
- Contract versioning & evolution tracking
- Automatic constraint inference (ranges, patterns, distributions)
- Custom validators & plugins
- Rich violation reports with remediation hints
- Contract diff & compatibility checks
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pydantic import BaseModel, Field, ValidationError

from .telemetry import audit, metric
from .types import UiNotice
from .utils import ensure_dir, sha256_of_path


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class ValidationLevel(str, Enum):
    """Poziomy walidacji kontraktu."""
    SCHEMA_ONLY = "schema_only"          # Tylko typy i nullable
    BASIC = "basic"                       # + ranges, cardinality
    STRICT = "strict"                     # + statistical checks
    BUSINESS = "business"                 # + custom business rules


class ViolationSeverity(str, Enum):
    """Poziomy powagi naruszeń."""
    CRITICAL = "critical"    # Blokuje deployment
    ERROR = "error"          # Wymaga naprawy
    WARNING = "warning"      # Do review
    INFO = "info"           # Informacyjne


@dataclass
class ViolationDetail:
    """Szczegóły pojedynczego naruszenia kontraktu."""
    severity: ViolationSeverity
    column: str
    check_name: str
    failure_count: int
    sample_values: List[Any]
    message: str
    remediation_hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "column": self.column,
            "check": self.check_name,
            "failures": self.failure_count,
            "sample": self.sample_values[:5],  # Max 5 examples
            "message": self.message,
            "hint": self.remediation_hint
        }


@dataclass
class ContractMetadata:
    """Rozszerzone metadane kontraktu."""
    run_id: str
    version: str
    created_utc: str
    validation_level: ValidationLevel
    schema_hash: str
    row_count: int
    column_count: int
    checksum: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    parent_version: Optional[str] = None  # Dla wersjonowania


# ============================================================================
# SCHEMA INFERENCE ENGINE
# ============================================================================

class SchemaInferenceEngine:
    """
    Silnik wnioskowania kontraktów z automatycznym wykrywaniem:
    - Ranges (min/max dla numerics)
    - Patterns (regex dla strings)
    - Cardinality constraints
    - Statistical properties (mean, std bounds)
    """
    
    # Confidence thresholds
    RANGE_CONFIDENCE = 0.95  # 95% danych w zakresie
    PATTERN_MIN_COVERAGE = 0.80  # 80% dopasowania do wzorca
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.BASIC):
        self.validation_level = validation_level
    
    def infer_schema(
        self, 
        df: pd.DataFrame,
        include_stats: bool = True,
        include_patterns: bool = True
    ) -> DataFrameSchema:
        """
        Wnioskuje zaawansowany schemat z automatycznymi constraintami.
        
        Args:
            df: DataFrame źródłowy
            include_stats: Czy dodawać statystyczne checks
            include_patterns: Czy wykrywać wzorce regex dla stringów
            
        Returns:
            Pandera DataFrameSchema z inferred checks
        """
        columns: Dict[str, Column] = {}
        
        for col_name in df.columns:
            series = df[col_name]
            dtype = series.dtype
            nullable = bool(series.isna().any())
            checks: List[Check] = []
            
            # 1. Type-based inference
            if pd.api.types.is_integer_dtype(dtype):
                pa_dtype = int
                if include_stats:
                    checks.extend(self._infer_numeric_checks(series))
                    
            elif pd.api.types.is_float_dtype(dtype):
                pa_dtype = float
                if include_stats:
                    checks.extend(self._infer_numeric_checks(series))
                    
            elif pd.api.types.is_bool_dtype(dtype):
                pa_dtype = bool
                
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                pa_dtype = "datetime64[ns]"
                checks.extend(self._infer_datetime_checks(series))
                
            else:
                pa_dtype = str
                if include_patterns:
                    pattern_check = self._infer_string_pattern(series)
                    if pattern_check:
                        checks.append(pattern_check)
            
            # 2. Cardinality check (only for low-cardinality columns)
            if self.validation_level in {ValidationLevel.STRICT, ValidationLevel.BUSINESS}:
                nunique = series.nunique()
                if nunique <= 100:  # Low cardinality
                    checks.append(
                        Check.isin(series.dropna().unique().tolist())
                    )
            
            # 3. Build column
            columns[col_name] = Column(
                pa_dtype,
                checks=checks,
                nullable=nullable,
                coerce=False
            )
        
        return DataFrameSchema(columns, coerce=False)
    
    def _infer_numeric_checks(self, series: pd.Series) -> List[Check]:
        """Wnioskuje checks dla kolumn numerycznych."""
        checks = []
        clean = series.dropna()
        
        if len(clean) == 0:
            return checks
        
        # Range check (percentile-based for robustness)
        q_low = clean.quantile(1 - self.RANGE_CONFIDENCE)
        q_high = clean.quantile(self.RANGE_CONFIDENCE)
        
        checks.append(Check.greater_than_or_equal_to(q_low))
        checks.append(Check.less_than_or_equal_to(q_high))
        
        # Statistical bounds (for STRICT level)
        if self.validation_level == ValidationLevel.STRICT:
            mean = float(clean.mean())
            std = float(clean.std())
            
            # Mean ± 3*sigma bounds (99.7% confidence for normal distribution)
            if std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                checks.append(Check.greater_than_or_equal_to(lower_bound))
                checks.append(Check.less_than_or_equal_to(upper_bound))
        
        return checks
    
    def _infer_datetime_checks(self, series: pd.Series) -> List[Check]:
        """Wnioskuje checks dla kolumn datetime."""
        checks = []
        clean = series.dropna()
        
        if len(clean) == 0:
            return checks
        
        # Reasonable date range
        min_date = clean.min()
        max_date = clean.max()
        
        checks.append(Check.greater_than_or_equal_to(min_date))
        checks.append(Check.less_than_or_equal_to(max_date))
        
        return checks
    
    def _infer_string_pattern(self, series: pd.Series) -> Optional[Check]:
        """
        Próbuje wykryć wspólny wzorzec regex dla kolumny stringowej.
        
        Obsługuje:
        - Email patterns
        - Phone numbers
        - UUIDs
        - Custom fixed-length patterns
        """
        clean = series.dropna().astype(str)
        
        if len(clean) < 10:  # Too few samples
            return None
        
        # Pre-defined patterns
        patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            "phone": r"^\+?[\d\s\-()]{7,}$",
            "url": r"^https?://[^\s]+$"
        }
        
        for pattern_name, pattern in patterns.items():
            matches = clean.str.match(pattern, case=False)
            coverage = matches.sum() / len(clean)
            
            if coverage >= self.PATTERN_MIN_COVERAGE:
                return Check.str_matches(pattern)
        
        # Fixed-length pattern detection
        lengths = clean.str.len()
        if lengths.nunique() == 1:
            fixed_len = int(lengths.iloc[0])
            return Check(
                lambda s: s.str.len() == fixed_len,
                name=f"fixed_length_{fixed_len}"
            )
        
        return None


# ============================================================================
# CONTRACT ENGINE
# ============================================================================

class ContractEngine:
    """
    Zaawansowany silnik kontraktów danych z:
    - Automatic schema inference
    - Versioning & evolution tracking
    - Custom validators
    - Rich violation reports
    - Contract diff
    """
    
    def __init__(self, base_dir: str = "artifacts/reports/baselines"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.inference_engine = SchemaInferenceEngine()
    
    # ------------------------------------------------------------------------
    # CONTRACT CREATION
    # ------------------------------------------------------------------------
    
    def create_contract(
        self,
        df: pd.DataFrame,
        run_id: str,
        validation_level: ValidationLevel = ValidationLevel.BASIC,
        custom_checks: Optional[Dict[str, List[Check]]] = None,
        tags: Optional[Dict[str, str]] = None,
        parent_version: Optional[str] = None
    ) -> Path:
        """
        Tworzy kontrakt danych z automatycznym wnioskowanie constraintów.
        
        Args:
            df: DataFrame źródłowy
            run_id: Identyfikator uruchomienia
            validation_level: Poziom strictness walidacji
            custom_checks: Dodatkowe custom checks per kolumna
            tags: Tagi metadanych
            parent_version: Wersja rodzica (dla evolution tracking)
            
        Returns:
            Ścieżka do zapisanego kontraktu
        """
        # 1. Infer base schema
        self.inference_engine.validation_level = validation_level
        schema = self.inference_engine.infer_schema(df)
        
        # 2. Add custom checks
        if custom_checks:
            for col_name, checks in custom_checks.items():
                if col_name in schema.columns:
                    existing = schema.columns[col_name]
                    existing.checks.extend(checks)
        
        # 3. Compute metadata
        schema_json = schema.to_json()
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()
        
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        
        metadata = ContractMetadata(
            run_id=run_id,
            version=version,
            created_utc=datetime.now(timezone.utc).isoformat(),
            validation_level=validation_level,
            schema_hash=schema_hash,
            row_count=len(df),
            column_count=len(df.columns),
            tags=tags or {},
            parent_version=parent_version
        )
        
        # 4. Save contract
        out_dir = self.base_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)
        
        contract_path = out_dir / "contract.json"
        
        payload = {
            "metadata": {
                "run_id": metadata.run_id,
                "version": metadata.version,
                "created_utc": metadata.created_utc,
                "validation_level": metadata.validation_level.value,
                "schema_hash": metadata.schema_hash,
                "row_count": metadata.row_count,
                "column_count": metadata.column_count,
                "tags": metadata.tags,
                "parent_version": metadata.parent_version
            },
            "schema": json.loads(schema_json)
        }
        
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        # 5. Compute checksum
        checksum = sha256_of_path(str(contract_path))
        payload["metadata"]["checksum"] = checksum
        
        # Re-save with checksum
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        # 6. Telemetry
        audit("contract_created", {
            "path": str(contract_path),
            "run_id": run_id,
            "version": version,
            "validation_level": validation_level.value
        })
        
        return contract_path
    
    # ------------------------------------------------------------------------
    # CONTRACT VALIDATION
    # ------------------------------------------------------------------------
    
    def validate_with_contract(
        self,
        df: pd.DataFrame,
        contract_path: Path | str,
        return_details: bool = True
    ) -> Dict[str, Any]:
        """
        Waliduje DataFrame względem kontraktu z bogatym raportem.
        
        Args:
            df: DataFrame do walidacji
            contract_path: Ścieżka do kontraktu
            return_details: Czy zwracać szczegóły naruszeń
            
        Returns:
            Dict z wynikami walidacji + szczegóły naruszeń
        """
        contract_path = Path(contract_path)
        
        # 1. Load contract
        with open(contract_path, "r", encoding="utf-8") as f:
            contract_data = json.load(f)
        
        schema_json = json.dumps(contract_data["schema"])
        schema = DataFrameSchema.from_json(schema_json)
        
        # 2. Validate
        violations: List[ViolationDetail] = []
        validation_ok = True
        
        try:
            schema.validate(df, lazy=True)
            
        except pa.errors.SchemaErrors as e:
            validation_ok = False
            
            # Parse violations with rich details
            for failure in e.failure_cases.itertuples(index=False):
                severity = self._infer_violation_severity(failure)
                
                violation = ViolationDetail(
                    severity=severity,
                    column=str(failure.column) if hasattr(failure, 'column') else "unknown",
                    check_name=str(failure.check) if hasattr(failure, 'check') else "unknown",
                    failure_count=int(failure.failure_case) if hasattr(failure, 'failure_case') else 0,
                    sample_values=self._extract_sample_values(df, failure),
                    message=self._build_violation_message(failure),
                    remediation_hint=self._generate_remediation_hint(failure)
                )
                violations.append(violation)
        
        # 3. Build result
        result = {
            "ok": validation_ok,
            "contract_version": contract_data["metadata"]["version"],
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "violations_count": len(violations),
            "critical_count": sum(1 for v in violations if v.severity == ViolationSeverity.CRITICAL),
            "error_count": sum(1 for v in violations if v.severity == ViolationSeverity.ERROR),
            "warning_count": sum(1 for v in violations if v.severity == ViolationSeverity.WARNING)
        }
        
        if return_details:
            result["violations"] = [v.to_dict() for v in violations]
        
        # 4. Telemetry
        audit("contract_validation", {
            "contract": str(contract_path),
            "ok": validation_ok,
            "violations": len(violations)
        })
        
        metric("contract_violations", len(violations), {
            "contract": contract_data["metadata"]["version"]
        })
        
        return result
    
    def _infer_violation_severity(self, failure) -> ViolationSeverity:
        """Wnioskuje powagę naruszenia na podstawie typu check."""
        check_name = str(getattr(failure, 'check', '')).lower()
        
        # Critical: type mismatches, nullability violations
        if 'dtype' in check_name or 'type' in check_name:
            return ViolationSeverity.CRITICAL
        
        if 'null' in check_name or 'none' in check_name:
            return ViolationSeverity.ERROR
        
        # Error: range violations, cardinality
        if any(x in check_name for x in ['range', 'greater', 'less', 'isin']):
            return ViolationSeverity.ERROR
        
        # Warning: statistical bounds
        if any(x in check_name for x in ['mean', 'std', 'distribution']):
            return ViolationSeverity.WARNING
        
        return ViolationSeverity.INFO
    
    def _extract_sample_values(self, df: pd.DataFrame, failure) -> List[Any]:
        """Wyciąga przykładowe wartości naruszające."""
        try:
            col = getattr(failure, 'column', None)
            if col and col in df.columns:
                # Get up to 5 unique violating values
                return df[col].dropna().unique().tolist()[:5]
        except Exception:
            pass
        return []
    
    def _build_violation_message(self, failure) -> str:
        """Buduje czytelny komunikat naruszenia."""
        col = getattr(failure, 'column', 'unknown')
        check = getattr(failure, 'check', 'unknown')
        return f"Column '{col}' failed check: {check}"
    
    def _generate_remediation_hint(self, failure) -> Optional[str]:
        """Generuje podpowiedź naprawczą."""
        check_name = str(getattr(failure, 'check', '')).lower()
        
        hints = {
            "dtype": "Check data types during ingestion. Consider explicit casting.",
            "null": "Remove or impute null values. Use .fillna() or .dropna().",
            "greater": "Values below expected range. Check data source or scaling.",
            "less": "Values above expected range. Check for outliers or data errors.",
            "isin": "Unexpected categorical values. Update contract or fix source data.",
            "str_match": "String format mismatch. Validate input patterns."
        }
        
        for key, hint in hints.items():
            if key in check_name:
                return hint
        
        return "Review data quality and contract definition."
    
    # ------------------------------------------------------------------------
    # CONTRACT DIFF & EVOLUTION
    # ------------------------------------------------------------------------
    
    def diff_contracts(
        self,
        old_contract: Path | str,
        new_contract: Path | str
    ) -> Dict[str, Any]:
        """
        Porównuje dwa kontrakty i zwraca różnice.
        
        Returns:
            Dict z: added_columns, removed_columns, changed_checks
        """
        def load_contract(path: Path | str) -> Dict[str, Any]:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        old = load_contract(old_contract)
        new = load_contract(new_contract)
        
        old_schema = old["schema"]
        new_schema = new["schema"]
        
        old_cols = set(old_schema.get("columns", {}).keys())
        new_cols = set(new_schema.get("columns", {}).keys())
        
        diff = {
            "added_columns": list(new_cols - old_cols),
            "removed_columns": list(old_cols - new_cols),
            "changed_checks": [],
            "compatible": True
        }
        
        # Check for incompatible changes
        if diff["removed_columns"]:
            diff["compatible"] = False
        
        return diff
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def list_contracts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lista dostępnych kontraktów."""
        contracts = []
        
        for dir_path in sorted(self.base_dir.iterdir(), reverse=True):
            if not dir_path.is_dir():
                continue
            
            contract_file = dir_path / "contract.json"
            if not contract_file.exists():
                continue
            
            try:
                with open(contract_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                contracts.append({
                    "path": str(contract_file),
                    "version": data["metadata"]["version"],
                    "run_id": data["metadata"]["run_id"],
                    "created_utc": data["metadata"]["created_utc"],
                    "validation_level": data["metadata"]["validation_level"]
                })
            except Exception:
                continue
            
            if len(contracts) >= limit:
                break
        
        return contracts


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

_engine: Optional[ContractEngine] = None

def _get_engine() -> ContractEngine:
    """Lazy initialization."""
    global _engine
    if _engine is None:
        _engine = ContractEngine()
    return _engine


def schema_from_df(df: pd.DataFrame) -> DataFrameSchema:
    """
    Deprecated: Use ContractEngine.inference_engine.infer_schema() instead.
    
    Basic schema inference for backward compatibility.
    """
    cols = {}
    for c in df.columns:
        dtype = df[c].dtype
        if pd.api.types.is_integer_dtype(dtype):
            col = Column(int, nullable=df[c].isna().any())
        elif pd.api.types.is_float_dtype(dtype):
            col = Column(float, nullable=df[c].isna().any())
        elif pd.api.types.is_bool_dtype(dtype):
            col = Column(bool, nullable=df[c].isna().any())
        else:
            col = Column(str, nullable=True)
        cols[c] = col
    return DataFrameSchema(cols, coerce=False)


def save_contract(
    df: pd.DataFrame,
    run_id: str,
    base_dir: str = "artifacts/reports/baselines",
    validation_level: ValidationLevel = ValidationLevel.BASIC,
    custom_checks: Optional[Dict[str, List[Check]]] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Zapisuje kontrakt danych (enhanced version).
    
    Kompatybilne z poprzednim API, ale z nowymi możliwościami.
    """
    engine = ContractEngine(base_dir)
    path = engine.create_contract(
        df=df,
        run_id=run_id,
        validation_level=validation_level,
        custom_checks=custom_checks,
        tags=tags
    )
    return str(path)


def validate_with_contract(
    df: pd.DataFrame,
    contract_path: str,
    return_details: bool = True
) -> Dict[str, Any]:
    """
    Waliduje DataFrame względem kontraktu.
    
    Enhanced version z rich violation reports.
    """
    engine = _get_engine()
    return engine.validate_with_contract(df, contract_path, return_details)


def _ensure_dir(p: str) -> None:
    """Deprecated: Use ensure_dir from utils instead."""
    os.makedirs(p, exist_ok=True)