"""
validation.py
Docstring (PL): Walidacja danych: pandera (główna) + opcjonalnie Great Expectations (gated).
"""
from __future__ import annotations
from typing import Dict, Any
import pandas as pd

from .data_contracts import loose_schema, HAS_PANDERA

def validate_pandera(df: pd.DataFrame) -> Dict[str, Any]:
    if not HAS_PANDERA:
        return {"status":"TMIV-VAL-000","message":"Pandera nie jest zainstalowana."}
    schema = loose_schema(df)
    try:
        _ = schema.validate(df, lazy=True)
        return {"status":"OK", "errors":[]}
    except Exception as e:
        # parsuj błędy pandera, jeśli dostępne
        return {"status":"TMIV-VAL-001", "message": "Naruszenia kontraktu danych (pandera).", "details": str(e)}

def validate_gx(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        import great_expectations as ge
    except Exception:
        return {"status":"TMIV-GX-000","message":"Great Expectations nie jest zainstalowane."}
    try:
        gdf = ge.from_pandas(df)
        # Prosty zestaw oczekiwań
        res1 = gdf.expect_table_row_count_to_be_greater_than(0)
        res2 = gdf.expect_table_columns_to_match_ordered_list(list(df.columns))
        ok = all([res1.success, res2.success])
        return {"status":"OK" if ok else "TMIV-GX-001", "results": {"rowcount": res1.success, "columns": res2.success}}
    except Exception as e:
        return {"status":"TMIV-GX-002","message":"Błąd walidacji GE","details": str(e)}