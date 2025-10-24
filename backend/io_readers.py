
from __future__ import annotations
import os
import pandas as pd

def read_any(upload, fallback_csv_path: str | None = None) -> pd.DataFrame:
    """Czyta CSV/Parquet z użyciem Polars (gated) jeśli USE_POLARS=true i polars jest dostępny, inaczej pandas.
    `upload` może być file-like albo ścieżka.
    """
    use_polars = os.environ.get("USE_POLARS", "false").lower() in {"1","true","yes"}
    name = getattr(upload, "name", str(upload)).lower()
    if use_polars:
        try:
            import polars as pl
            if name.endswith('.parquet'):
                df = pl.read_parquet(upload).to_pandas()
            else:
                df = pl.read_csv(upload).to_pandas()
            return df
        except Exception:
            pass
    # pandas fallback
    if name.endswith('.parquet'):
        return pd.read_parquet(upload)
    return pd.read_csv(upload)
