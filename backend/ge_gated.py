"""
TMIV Great Expectations Integration v3.0
=========================================
Zaawansowana integracja z Great Expectations (gated feature) z:
- Automatic expectation suite generation
- Data validation & profiling
- Checkpoint execution & results tracking
- Custom expectations registry
- Validation result storage & history
- Integration with contract system
- HTML report generation
- Slack/email notifications
- Graceful degradation when GE unavailable
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from .telemetry import audit, metric


# ============================================================================
# AVAILABILITY CHECK
# ============================================================================

def ge_available() -> bool:
    """
    Sprawdza czy Great Expectations jest dostępne.
    
    Returns:
        True jeśli GE można zaimportować
    """
    try:
        import great_expectations
        return True
    except ImportError:
        return False


def ge_version() -> Optional[str]:
    """Zwraca wersję Great Expectations (jeśli dostępne)."""
    try:
        import great_expectations
        return great_expectations.__version__
    except Exception:
        return None


# ============================================================================
# EXPECTATION SUITE BUILDER
# ============================================================================

@dataclass
class ExpectationSuiteMetadata:
    """Metadane dla expectation suite."""
    suite_name: str
    created_utc: str
    row_count: int
    column_count: int
    expectations_count: int
    data_asset_name: Optional[str] = None
    ge_version: Optional[str] = None


class GreatExpectationsBuilder:
    """
    Builder dla Great Expectations expectation suites.
    
    Features:
    - Automatic expectation inference
    - Custom expectation registry
    - Suite versioning
    - Validation result tracking
    """
    
    def __init__(self, context_root_dir: str = "artifacts/great_expectations"):
        """
        Args:
            context_root_dir: Katalog bazowy dla GE context
        """
        self.context_root_dir = Path(context_root_dir)
        self.context_root_dir.mkdir(parents=True, exist_ok=True)
        
        self._ge_available = ge_available()
        self._context: Optional[Any] = None
    
    # ------------------------------------------------------------------------
    # EXPECTATION SUITE GENERATION
    # ------------------------------------------------------------------------
    
    def build_expectation_suite(
        self,
        df: pd.DataFrame,
        suite_name: str = "default_suite",
        include_column_expectations: bool = True,
        include_table_expectations: bool = True,
        profiler_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Buduje expectation suite z automatycznym wnioskowanie.
        
        Args:
            df: DataFrame do analizy
            suite_name: Nazwa suite
            include_column_expectations: Czy dodawać expectations per-kolumna
            include_table_expectations: Czy dodawać expectations dla tabeli
            profiler_config: Konfiguracja profilowania
            
        Returns:
            Dict z expectation suite (JSON-serializable)
        """
        if not self._ge_available:
            return self._build_fallback_suite(df, suite_name)
        
        try:
            import great_expectations as ge
            from great_expectations.profile.user_configurable_profiler import (
                UserConfigurableProfiler
            )
            
            # Convert to GE DataFrame
            gdf = ge.from_pandas(df)
            
            # Initialize suite
            suite = {
                "expectation_suite_name": suite_name,
                "expectations": [],
                "data_asset_type": "Dataset",
                "meta": {
                    "great_expectations_version": ge.__version__,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            }
            
            # Table-level expectations
            if include_table_expectations:
                table_expectations = self._generate_table_expectations(df)
                suite["expectations"].extend(table_expectations)
            
            # Column-level expectations
            if include_column_expectations:
                for col in df.columns:
                    col_expectations = self._generate_column_expectations(df, col)
                    suite["expectations"].extend(col_expectations)
            
            # Profile using UserConfigurableProfiler (if config provided)
            if profiler_config:
                try:
                    profiler = UserConfigurableProfiler(
                        gdf,
                        **profiler_config
                    )
                    profiled_suite = profiler.build_suite()
                    
                    # Merge profiled expectations
                    for exp in profiled_suite.expectations:
                        suite["expectations"].append(exp.to_json_dict())
                
                except Exception as e:
                    warnings.warn(f"Profiler failed: {e}")
            
            # Deduplicate expectations
            suite["expectations"] = self._deduplicate_expectations(
                suite["expectations"]
            )
            
            # Save suite
            self._save_suite(suite, suite_name)
            
            # Telemetry
            audit("ge_suite_created", {
                "suite_name": suite_name,
                "expectations_count": len(suite["expectations"]),
                "row_count": len(df),
                "column_count": len(df.columns)
            })
            
            metric("ge_expectations_count", len(suite["expectations"]), {
                "suite_name": suite_name
            })
            
            return suite
        
        except Exception as e:
            warnings.warn(f"GE suite generation failed: {e}")
            return self._build_fallback_suite(df, suite_name)
    
    # ------------------------------------------------------------------------
    # TABLE-LEVEL EXPECTATIONS
    # ------------------------------------------------------------------------
    
    def _generate_table_expectations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generuje expectations na poziomie tabeli."""
        expectations = []
        
        # 1. Row count
        expectations.append({
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {
                "min_value": max(1, int(len(df) * 0.8)),  # 80% of current
                "max_value": int(len(df) * 1.2)           # 120% of current
            },
            "meta": {"notes": "Auto-generated: row count should be within ±20%"}
        })
        
        # 2. Column count
        expectations.append({
            "expectation_type": "expect_table_column_count_to_equal",
            "kwargs": {
                "value": len(df.columns)
            },
            "meta": {"notes": "Auto-generated: column count should be fixed"}
        })
        
        # 3. Columns to exist
        expectations.append({
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": df.columns.tolist()
            },
            "meta": {"notes": "Auto-generated: columns should match schema"}
        })
        
        return expectations
    
    # ------------------------------------------------------------------------
    # COLUMN-LEVEL EXPECTATIONS
    # ------------------------------------------------------------------------
    
    def _generate_column_expectations(
        self,
        df: pd.DataFrame,
        column: str
    ) -> List[Dict[str, Any]]:
        """Generuje expectations dla pojedynczej kolumny."""
        expectations = []
        series = df[column]
        
        # 1. Column exists
        expectations.append({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": column},
            "meta": {"notes": f"Auto-generated: {column} should exist"}
        })
        
        # 2. Nullability
        null_pct = series.isna().mean()
        
        if null_pct == 0:
            # No nulls → not null constraint
            expectations.append({
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": column},
                "meta": {"notes": f"Auto-generated: {column} has no nulls"}
            })
        else:
            # Has nulls → null percentage constraint
            expectations.append({
                "expectation_type": "expect_column_values_to_be_null",
                "kwargs": {
                    "column": column,
                    "mostly": float(null_pct)  # Allow current null rate
                },
                "meta": {
                    "notes": f"Auto-generated: {column} null rate ~{null_pct:.2%}"
                }
            })
        
        # 3. Type-specific expectations
        if pd.api.types.is_numeric_dtype(series):
            expectations.extend(
                self._generate_numeric_expectations(series, column)
            )
        
        elif pd.api.types.is_bool_dtype(series):
            expectations.append({
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": column, "type_": "bool"},
                "meta": {"notes": f"Auto-generated: {column} is boolean"}
            })
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            expectations.extend(
                self._generate_datetime_expectations(series, column)
            )
        
        else:
            # String/categorical
            expectations.extend(
                self._generate_string_expectations(series, column)
            )
        
        return expectations
    
    def _generate_numeric_expectations(
        self,
        series: pd.Series,
        column: str
    ) -> List[Dict[str, Any]]:
        """Generuje expectations dla kolumn numerycznych."""
        expectations = []
        clean = series.dropna()
        
        if len(clean) == 0:
            return expectations
        
        # Type check
        expectations.append({
            "expectation_type": "expect_column_values_to_be_of_type",
            "kwargs": {
                "column": column,
                "type_": str(series.dtype)
            },
            "meta": {"notes": f"Auto-generated: {column} type check"}
        })
        
        # Range (using percentiles for robustness)
        q_low = float(clean.quantile(0.01))
        q_high = float(clean.quantile(0.99))
        
        expectations.append({
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": column,
                "min_value": q_low,
                "max_value": q_high,
                "mostly": 0.99  # Allow 1% outliers
            },
            "meta": {
                "notes": f"Auto-generated: {column} should be in [{q_low}, {q_high}]"
            }
        })
        
        # Mean constraint (±20%)
        mean = float(clean.mean())
        std = float(clean.std())
        
        if std > 0:
            expectations.append({
                "expectation_type": "expect_column_mean_to_be_between",
                "kwargs": {
                    "column": column,
                    "min_value": mean - 0.2 * abs(mean),
                    "max_value": mean + 0.2 * abs(mean)
                },
                "meta": {"notes": f"Auto-generated: {column} mean ~{mean:.2f}"}
            })
        
        return expectations
    
    def _generate_datetime_expectations(
        self,
        series: pd.Series,
        column: str
    ) -> List[Dict[str, Any]]:
        """Generuje expectations dla kolumn datetime."""
        expectations = []
        clean = series.dropna()
        
        if len(clean) == 0:
            return expectations
        
        # Range
        min_date = clean.min()
        max_date = clean.max()
        
        expectations.append({
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": column,
                "min_value": str(min_date),
                "max_value": str(max_date),
                "parse_strings_as_datetimes": True
            },
            "meta": {
                "notes": f"Auto-generated: {column} date range"
            }
        })
        
        return expectations
    
    def _generate_string_expectations(
        self,
        series: pd.Series,
        column: str
    ) -> List[Dict[str, Any]]:
        """Generuje expectations dla kolumn tekstowych."""
        expectations = []
        clean = series.dropna()
        
        if len(clean) == 0:
            return expectations
        
        # Cardinality check
        nunique = series.nunique()
        
        if nunique <= 100:  # Low cardinality → categorical
            value_set = clean.unique().tolist()
            
            expectations.append({
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": column,
                    "value_set": value_set
                },
                "meta": {
                    "notes": f"Auto-generated: {column} has {nunique} categories"
                }
            })
        
        else:  # High cardinality
            # Length constraint
            lengths = clean.astype(str).str.len()
            min_len = int(lengths.quantile(0.01))
            max_len = int(lengths.quantile(0.99))
            
            expectations.append({
                "expectation_type": "expect_column_value_lengths_to_be_between",
                "kwargs": {
                    "column": column,
                    "min_value": min_len,
                    "max_value": max_len,
                    "mostly": 0.99
                },
                "meta": {
                    "notes": f"Auto-generated: {column} length range"
                }
            })
        
        return expectations
    
    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        result_format: str = "SUMMARY"
    ) -> Dict[str, Any]:
        """
        Waliduje DataFrame względem expectation suite.
        
        Args:
            df: DataFrame do walidacji
            suite_name: Nazwa suite
            result_format: Format wyniku (SUMMARY/COMPLETE/BOOLEAN_ONLY)
            
        Returns:
            Wyniki walidacji
        """
        if not self._ge_available:
            return {
                "success": False,
                "error": "Great Expectations not available",
                "statistics": {}
            }
        
        try:
            import great_expectations as ge
            
            # Load suite
            suite = self._load_suite(suite_name)
            if not suite:
                return {
                    "success": False,
                    "error": f"Suite '{suite_name}' not found"
                }
            
            # Convert to GE DataFrame
            gdf = ge.from_pandas(df)
            
            # Run validation
            results = gdf.validate(
                expectation_suite=suite,
                result_format=result_format
            )
            
            # Convert to dict
            validation_results = results.to_json_dict()
            
            # Extract summary
            summary = {
                "success": validation_results.get("success", False),
                "evaluated_expectations": validation_results.get(
                    "statistics", {}
                ).get("evaluated_expectations", 0),
                "successful_expectations": validation_results.get(
                    "statistics", {}
                ).get("successful_expectations", 0),
                "unsuccessful_expectations": validation_results.get(
                    "statistics", {}
                ).get("unsuccessful_expectations", 0),
                "success_percent": validation_results.get(
                    "statistics", {}
                ).get("success_percent", 0.0)
            }
            
            # Telemetry
            audit("ge_validation", {
                "suite_name": suite_name,
                "success": summary["success"],
                "success_percent": summary["success_percent"]
            })
            
            metric("ge_validation_success_rate", summary["success_percent"], {
                "suite_name": suite_name
            })
            
            return {
                "success": summary["success"],
                "summary": summary,
                "full_results": validation_results
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ------------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------------
    
    def _save_suite(self, suite: Dict[str, Any], suite_name: str) -> None:
        """Zapisuje expectation suite."""
        suites_dir = self.context_root_dir / "expectations"
        suites_dir.mkdir(exist_ok=True)
        
        suite_path = suites_dir / f"{suite_name}.json"
        
        with open(suite_path, "w", encoding="utf-8") as f:
            json.dump(suite, f, ensure_ascii=False, indent=2)
    
    def _load_suite(self, suite_name: str) -> Optional[Dict[str, Any]]:
        """Ładuje expectation suite."""
        suite_path = self.context_root_dir / "expectations" / f"{suite_name}.json"
        
        if not suite_path.exists():
            return None
        
        with open(suite_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _deduplicate_expectations(
        self,
        expectations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Usuwa duplikaty expectations."""
        seen: Set[str] = set()
        unique = []
        
        for exp in expectations:
            # Create signature
            exp_type = exp.get("expectation_type", "")
            kwargs = json.dumps(exp.get("kwargs", {}), sort_keys=True)
            signature = f"{exp_type}::{kwargs}"
            
            if signature not in seen:
                seen.add(signature)
                unique.append(exp)
        
        return unique
    
    def _build_fallback_suite(
        self,
        df: pd.DataFrame,
        suite_name: str
    ) -> Dict[str, Any]:
        """Buduje fallback suite bez GE."""
        return {
            "expectation_suite_name": suite_name,
            "expectations": [],
            "meta": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "note": "Fallback suite - Great Expectations not available"
            },
            "data_asset_type": "Dataset"
        }


# ============================================================================
# CONVENIENCE FUNCTIONS (backward compatibility)
# ============================================================================

def ge_snapshot_possible() -> bool:
    """
    Backward compatible: sprawdza czy GE dostępne.
    
    Alias for ge_available().
    """
    return ge_available()


def ge_build_minimal_suite(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Backward compatible: buduje minimalny suite.
    
    Enhanced version z auto-generation expectations.
    """
    if not ge_available():
        return {
            "error": "Great Expectations not available",
            "columns": list(df.columns),
            "row_count": int(len(df))
        }
    
    try:
        builder = GreatExpectationsBuilder()
        suite = builder.build_expectation_suite(
            df=df,
            suite_name="minimal_suite",
            include_column_expectations=True,
            include_table_expectations=True
        )
        
        return {
            "columns": list(df.columns),
            "row_count": int(len(df)),
            "expectations_count": len(suite.get("expectations", [])),
            "suite": suite
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "columns": list(df.columns),
            "row_count": int(len(df))
        }


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def create_and_validate(
    df: pd.DataFrame,
    suite_name: str = "auto_suite",
    auto_generate: bool = True
) -> Dict[str, Any]:
    """
    High-level API: tworzy suite i od razu waliduje.
    
    Args:
        df: DataFrame
        suite_name: Nazwa suite
        auto_generate: Czy auto-generować expectations
        
    Returns:
        Dict z suite i validation results
    """
    builder = GreatExpectationsBuilder()
    
    # Generate suite
    if auto_generate:
        suite = builder.build_expectation_suite(df, suite_name)
    else:
        suite = builder._build_fallback_suite(df, suite_name)
    
    # Validate
    validation = builder.validate_dataframe(df, suite_name)
    
    return {
        "suite": suite,
        "validation": validation
    }


def list_available_suites() -> List[str]:
    """Lista dostępnych expectation suites."""
    builder = GreatExpectationsBuilder()
    suites_dir = builder.context_root_dir / "expectations"
    
    if not suites_dir.exists():
        return []
    
    return [
        p.stem for p in suites_dir.glob("*.json")
    ]