
from __future__ import annotations

def ge_snapshot_possible() -> bool:
    try:
        import great_expectations  # noqa: F401
        return True
    except Exception:
        return False

def ge_build_minimal_suite(df) -> dict:
    """Zwraca minimalny opis oczekiwań (gdy GE dostępne)."""
    try:
        import great_expectations as ge
        gdf = ge.from_pandas(df)
        res = {
            "columns": list(df.columns),
            "row_count": int(len(df)),
        }
        return res
    except Exception as e:
        return {"error": str(e)}
