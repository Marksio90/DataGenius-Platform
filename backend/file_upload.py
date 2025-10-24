# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
#  Shimy i fallbacki (bezpieczne importy)
# =============================================================================
try:
    # Lokalnie, jeśli struktura pakietu jest dostępna
    from .security_manager import rate_limiter, data_validator  # type: ignore
except Exception:
    # Fallback, gdy relative import nie działa
    class _RateLimiter:
        def enforce(self, key: str):
            return True

    rate_limiter = _RateLimiter()

    class _DataValidator:
        def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    data_validator = _DataValidator()

# --- SmartErrorHandler shim ---------------------------------------------------
# Zapewnia dekorator .data_processing nawet jeśli backend.error_handler go nie ma.
try:
    from .error_handler import SmartErrorHandler as _SEH  # type: ignore
except Exception:
    _SEH = None  # brak modułu lub klasy

def _mk_data_processing_decorator():
    """Tworzy dekorator, który pokazuje błąd w Streamlit (jeśli dostępny) i zwraca None."""
    import functools, traceback

    # Próba leniwego importu streamlit (może nie istnieć w środowisku unit-testów)
    try:
        import streamlit as st  # type: ignore
    except Exception:
        st = None

    def data_processing(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if st is not None:
                    st.error(f"❌ Błąd przetwarzania danych: {e}")
                    with st.expander("🐛 Szczegóły", expanded=False):
                        st.code(traceback.format_exc())
                # Zwracamy None, aby nie wywalać całej aplikacji (zachowanie zgodne z dotychczasowym fallbackiem)
                return None
        return wrapper
    return data_processing

if _SEH is None or not hasattr(_SEH, "data_processing"):
    class SmartErrorHandler:  # type: ignore
        data_processing = staticmethod(_mk_data_processing_decorator())
else:
    SmartErrorHandler = _SEH  # type: ignore


# =============================================================================
#  Klasa FileUploadHandler
# =============================================================================
class FileUploadHandler:
    """Obsługa wczytywania plików oraz przykładowych zbiorów danych (FULL)."""

    supported_formats = ["csv", "json", "xlsx", "parquet"]
    max_file_size = 200 * 1024 * 1024  # 200 MB

    def __init__(self) -> None:
        pass

    # -------------------------- helpers ---------------------------------------
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Porządki pod UI/EDA:
        - kolumny category -> string,
        - nazwy kolumn -> str.
        """
        dfc = df.copy()
        for c in dfc.columns:
            if str(dfc[c].dtype).startswith("category"):
                dfc[c] = dfc[c].astype("string")
        # (opcjonalnie) nullable string -> string (spójnie)
        str_cols = dfc.select_dtypes(include=["string"]).columns
        for c in str_cols:
            dfc[c] = dfc[c].astype("string")
        # kolumny jako str (na wypadek kolumn typu int/tuple)
        dfc.columns = [str(x) for x in dfc.columns]
        return dfc

    def _read_csv_with_auto_sep_and_encoding(self, bio: io.BytesIO) -> pd.DataFrame:
        """Próbuje różne separatory i kodowania."""
        seps = [",", ";", "|", "\t"]
        encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]

        last_err: Optional[Exception] = None

        for enc in encodings:
            for sep in seps:
                bio.seek(0)
                try:
                    df = pd.read_csv(bio, sep=sep, encoding=enc)
                    # heurystyka: jeśli parser złapał 1 kolumnę i sep != ',', spróbuj dalej
                    if df.shape[1] > 1 or sep == ",":
                        return df
                except Exception as e:
                    last_err = e
                    continue

        # Ostatnia próba — domyślnie
        bio.seek(0)
        if last_err:
            # spróbuj bez parametrów, aby oddać sensowny wyjątek Pandas gdyby się nie udało
            return pd.read_csv(bio)
        return pd.read_csv(bio)

    # ------------------------ główne metody -----------------------------------
    @SmartErrorHandler.data_processing
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Wczytuje CSV/JSON/XLSX/Parquet z auto-separatorem, próbą kodowania i normalizacją."""
        if uploaded_file is None:
            raise ValueError("Nie wybrano pliku.")

        # Limit rozmiaru
        size = getattr(uploaded_file, "size", None)
        if size and size > self.max_file_size:
            raise ValueError("Plik jest za duży (limit 200MB).")

        name = getattr(uploaded_file, "name", "data.csv").lower()

        # Pozyskaj bajty i opakuj w BytesIO
        if hasattr(uploaded_file, "read"):
            raw = uploaded_file.read()
        else:
            raw = uploaded_file  # już bytes/bytearray/str
        if isinstance(raw, (bytes, bytearray)):
            bio = io.BytesIO(raw)
        else:
            bio = io.BytesIO(str(raw).encode("utf-8"))

        # Rate limit
        rate_limiter.enforce("file_uploads")

        # Rozpoznanie formatu
        if name.endswith(".csv"):
            df = self._read_csv_with_auto_sep_and_encoding(bio)

        elif name.endswith(".json"):
            # Próby różnych wariantów JSON
            for orient in (None,):
                for enc in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
                    try:
                        bio.seek(0)
                        return self._normalize_df(pd.read_json(bio, lines=False, encoding=enc))
                    except Exception:
                        continue
            # ostatnia próba bez encoding param
            bio.seek(0)
            df = pd.read_json(bio, lines=False)

        elif name.endswith(".xlsx"):
            bio.seek(0)
            # wymagany openpyxl
            import openpyxl  # noqa: F401
            df = pd.read_excel(bio)

        elif name.endswith(".parquet"):
            bio.seek(0)
            # preferujemy pyarrow
            import pyarrow  # noqa: F401
            df = pd.read_parquet(bio)

        else:
            raise ValueError(f"Nieobsługiwany format pliku (dozwolone: {', '.join(self.supported_formats)}).")

        df = self._normalize_df(df)
        return data_validator.validate_dataframe(df)

    @SmartErrorHandler.data_processing
    def load_sample_dataset(self, name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Ładuje przykładowy zbiór: avocado/iris/wine/california lub generuje fallback."""
        name = (name or "iris").lower()

        # 1) Spróbuj lokalnego avocado
        # root repo = dwa poziomy wyżej od tego pliku
        roots = [Path(__file__).resolve().parents[1]]
        for rel in ("data/avocado.csv", "datasets/avocado.csv", "avocado.csv"):
            for root in roots:
                f = root / rel
                if name in ("avocado", "avo") and f.exists():
                    df = pd.read_csv(f)
                    return self._normalize_df(df)

        # 2) scikit-learn datasets
        try:
            from sklearn.datasets import load_iris, load_wine, fetch_california_housing  # type: ignore
        except Exception:
            load_iris = load_wine = fetch_california_housing = None  # type: ignore

        if name in ("iris", "default") and load_iris:
            data = load_iris(as_frame=True)
            df = data.frame.copy()
            # niektóre wersje i tak mają 'target' w .frame, ale dodatkowe przypisanie nie szkodzi
            df["target"] = data.target
            return self._normalize_df(df)

        if name in ("wine", "wine_quality", "vinos") and load_wine:
            data = load_wine(as_frame=True)
            df = data.frame.copy()
            df["target"] = data.target
            return self._normalize_df(df)

        if name in ("california", "housing") and fetch_california_housing:
            data = fetch_california_housing(as_frame=True)
            df = data.frame.copy()
            df["MedHouseVal"] = data.target
            return self._normalize_df(df)

        # 3) Fallback syntetyczny (klasyfikacja binarna)
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "feature_a": rng.normal(size=300),
                "feature_b": rng.integers(0, 5, size=300).astype("int64"),
                "category": pd.Series(rng.choice(["A", "B", "C"], size=300), dtype="string"),
                "target": rng.integers(0, 2, size=300).astype("int64"),
            }
        )
        return self._normalize_df(df)
