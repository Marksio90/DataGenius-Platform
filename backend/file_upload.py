from __future__ import annotations
# -*- coding: utf-8 -*-

from backend.safe_utils import truthy_df_safe

import io
from pathlib import Path
from typing import Optional, List, Sequence, Union

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
    """Tworzy dekorator, który pokazuje błąd w Streamlit (jeśli dostępny) i zwraca None (bez rerun pętli)."""
    import functools, traceback
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
                    try:
                        st.error(f"❌ Błąd przetwarzania danych: {e}")
                        with st.expander("🐛 Szczegóły", expanded=False):
                            st.code(traceback.format_exc())
                    except Exception:
                        # nie wywołuj st.experimental_rerun tutaj – to mogłoby wywoływać pętlę refresh
                        pass
                return None
        return wrapper
    return data_processing


# Ekspozycja spójnego API: SmartErrorHandler.data_processing
if _SEH is not None and hasattr(_SEH, "data_processing_handler"):
    class SmartErrorHandler:  # type: ignore
        data_processing = staticmethod(_SEH.data_processing_handler)
elif _SEH is not None and hasattr(_SEH, "data_processing"):
    class SmartErrorHandler:  # type: ignore
        data_processing = staticmethod(_SEH.data_processing)
else:
    class SmartErrorHandler:  # type: ignore
        data_processing = staticmethod(_mk_data_processing_decorator())


# =============================================================================
#  Pomocnicza sanizacja CSV/DF (użyteczna w wielu miejscach)
# =============================================================================
def _dedupe_columns(cols: Sequence[object]) -> List[str]:
    """Zapewnia unikalne nagłówki (pandas zwykle to robi, ale dla spójności)."""
    seen = {}
    out: List[str] = []
    for c in cols:
        base = str(c)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}.{seen[base]}")
    return out


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    # usuń śmieciowe kolumny i całe puste
    drop_cols = [c for c in df.columns if str(c).strip().lower().startswith(("unnamed", "index"))]
    if truthy_df_safe(drop_cols):
        df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna(axis=1, how="all")

    # ujednolicenia dtype'ów „kłopotliwych” dla modeli/wykresów
    try:
        from pandas.api.types import is_categorical_dtype
    except Exception:
        def is_categorical_dtype(x):  # type: ignore
            return str(getattr(x, "dtype", "")) == "category"

    for c in df.columns:
        # category -> string (stabilniej w UI i przy eksportach)
        try:
            if is_categorical_dtype(df[c]):
                df[c] = df[c].astype("string").fillna("NA")
        except Exception:
            pass

    # spójne nagłówki
    df.columns = _dedupe_columns(df.columns)
    return df


def _read_csv_smart(src, *, try_index_col_first: bool = True) -> pd.DataFrame:
    """CSV z heurystykami: sniff sep/encoding, bez pętli rerun."""
    # 1) Spróbuj automatyczny sniffer (engine='python'), a potem fallbacki
    def _try_read(**kw):
        return pd.read_csv(src, **kw)

    # Najpierw sniffer
    kw_base = dict(engine="python")
    if try_index_col_first:
        try:
            df = _try_read(sep=None, index_col=0, **kw_base)
            if df.shape[1] >= 1:
                return _sanitize_df(df)
        except Exception:
            pass
    try:
        df = _try_read(sep=None, **kw_base)
        if df.shape[1] >= 1:
            return _sanitize_df(df)
    except Exception:
        pass

    # 2) Matrix prób: encodings x separators
    seps = [",", ";", "|", "\t"]
    encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(src, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1 or sep == ",":
                    return _sanitize_df(df)
            except Exception as e:
                last_err = e
                continue

    # 3) Ostatnia próba — niech pandas zgłosi klarowny wyjątek
    if truthy_df_safe(last_err):
        df = pd.read_csv(src)
    else:
        df = pd.read_csv(src)
    return _sanitize_df(df)


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
        - nazwy kolumn -> unikalne str.
        """
        dfc = df.copy()

        # category -> string / string -> string (nullable)
        try:
            from pandas.api.types import is_categorical_dtype
            for c in dfc.columns:
                if is_categorical_dtype(dfc[c]):
                    dfc[c] = dfc[c].astype("string")
        except Exception:
            pass

        str_cols = dfc.select_dtypes(include=["string"]).columns
        for c in str_cols:
            dfc[c] = dfc[c].astype("string")

        dfc.columns = _dedupe_columns(dfc.columns)
        return dfc

    def _read_csv_with_auto_sep_and_encoding(self, bio: io.BytesIO) -> pd.DataFrame:
        """Próbuje różne separatory i kodowania, zwraca pierwszy sensowny wynik."""
        # pozwól pandasowi posniffować najpierw
        try:
            bio.seek(0)
            df = _read_csv_smart(bio, try_index_col_first=True)
            return self._normalize_df(_sanitize_df(df))
        except Exception:
            pass

        seps = [",", ";", "|", "\t"]
        encodings = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
        last_err: Optional[Exception] = None

        for enc in encodings:
            for sep in seps:
                bio.seek(0)
                try:
                    df = pd.read_csv(bio, sep=sep, encoding=enc, engine="python")
                    if df.shape[1] > 1 or sep == ",":
                        return self._normalize_df(_sanitize_df(df))
                except Exception as e:
                    last_err = e
                    continue

        # Ostatnia próba — domyślnie
        bio.seek(0)
        if truthy_df_safe(last_err):
            df = pd.read_csv(bio)  # niech pandas zgłosi sensowny wyjątek jeśli padnie
        else:
            df = pd.read_csv(bio)
        return self._normalize_df(_sanitize_df(df))

    def _bytes_io_from_input(self, uploaded_file: Union[bytes, bytearray, io.BytesIO, io.BufferedReader, str, Path, object]) -> io.BytesIO:
        """Ujednolica różne typy wejścia (Streamlit UploadedFile / bytes / ścieżka)."""
        # Streamlit UploadedFile
        if hasattr(uploaded_file, "read"):
            raw = uploaded_file.read()
            return io.BytesIO(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
        # BytesIO / file-like
        if isinstance(uploaded_file, (io.BytesIO, io.BufferedReader)):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            data = uploaded_file.read()
            return io.BytesIO(data)
        # Ścieżka plikowa
        if isinstance(uploaded_file, (str, Path)):
            p = Path(uploaded_file)
            with p.open("rb") as f:
                return io.BytesIO(f.read())
        # surowe bytes
        if isinstance(uploaded_file, (bytes, bytearray)):
            return io.BytesIO(uploaded_file)
        # fallback: zrób z repr
        return io.BytesIO(str(uploaded_file).encode("utf-8"))

    # ------------------------ główne metody -----------------------------------
    @SmartErrorHandler.data_processing
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Wczytuje CSV/JSON/XLSX/Parquet z auto-separatorem, próbą kodowania i normalizacją."""
        if uploaded_file is None:
            raise ValueError("Nie wybrano pliku.")

        # Limit rozmiaru (jeśli znany)
        size = getattr(uploaded_file, "size", None)
        if truthy_df_safe(size) and size > self.max_file_size:
            raise ValueError("Plik jest za duży (limit 200MB).")

        # Nazwa/rozszerzenie (gdy brak — domyślnie 'data.csv')
        name = (getattr(uploaded_file, "name", None) or str(uploaded_file) or "data.csv").lower()

        # BytesIO z wejścia (stabilne „rewind” przy wielokrotnych próbach)
        bio = self._bytes_io_from_input(uploaded_file)

        # Rate limit
        rate_limiter.enforce("file_uploads")

        # Rozpoznanie formatu (proste po rozszerzeniu; dla nietypowych można próbować heurystyk)
        if name.endswith(".csv") or ".csv" in name:
            df = self._read_csv_with_auto_sep_and_encoding(bio)

        elif name.endswith(".json") or ".jsonl" in name:
            # Spróbuj kilka wariantów: lines=True/False i różne enkodowania
            tried: List[str] = []
            for lines_flag in (True, False):
                for enc in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
                    try:
                        bio.seek(0)
                        text = bio.read().decode(enc, errors="replace")
                        df = pd.read_json(io.StringIO(text), lines=lines_flag)
                        return self._normalize_df(_sanitize_df(df))
                    except Exception:
                        tried.append(f"lines={lines_flag}, enc={enc}")
                        continue
            # ostatnia próba na surowych bajtach
            bio.seek(0)
            df = pd.read_json(bio, lines=False)
            df = self._normalize_df(_sanitize_df(df))

        elif name.endswith(".xlsx") or name.endswith(".xls"):
            bio.seek(0)
            # wymagany openpyxl (xls → pandas i tak spróbuje przekonwertować)
            try:
                import openpyxl  # noqa: F401
            except Exception as e:
                raise ValueError(f"Brak silnika Excel (zainstaluj openpyxl): {e}")
            # domyślnie pierwszy arkusz
            df = pd.read_excel(bio, sheet_name=0)
            df = self._normalize_df(_sanitize_df(df))

        elif name.endswith(".parquet") or name.endswith(".pq"):
            bio.seek(0)
            # preferujemy pyarrow, ale jeśli brak, spróbuj fastparquet
            try:
                import pyarrow  # noqa: F401
                df = pd.read_parquet(bio)
            except Exception:
                try:
                    import fastparquet  # noqa: F401
                    df = pd.read_parquet(bio, engine="fastparquet")
                except Exception as e:
                    raise ValueError(f"Brak silnika Parquet (zainstaluj pyarrow lub fastparquet): {e}")
            df = self._normalize_df(_sanitize_df(df))

        else:
            # Jeżeli rozszerzenie nieznane — spróbuj CSV jako najbardziej popularne
            try:
                df = self._read_csv_with_auto_sep_and_encoding(bio)
            except Exception:
                raise ValueError(f"Nieobsługiwany format pliku (dozwolone: {', '.join(self.supported_formats)}).")

        # Walidacja końcowa (może modyfikować df)
        return data_validator.validate_dataframe(df)

    @SmartErrorHandler.data_processing
    def load_sample_dataset(self, name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Ładuje przykładowy zbiór: avocado/iris/wine/california lub generuje fallback."""
        name = (name or "iris").lower()

        # 1) Spróbuj lokalnego avocado
        try:
            roots = [Path(__file__).resolve().parents[1]]
        except Exception:
            roots = [Path.cwd()]
        for rel in ("data/avocado.csv", "datasets/avocado.csv", "avocado.csv"):
            for root in roots:
                f = root / rel
                if name in ("avocado", "avo") and f.exists():
                    df = pd.read_csv(f)
                    return self._normalize_df(_sanitize_df(df))

        # 2) scikit-learn datasets
        try:
            from sklearn.datasets import load_iris, load_wine, fetch_california_housing  # type: ignore
        except Exception:
            load_iris = load_wine = fetch_california_housing = None  # type: ignore

        if name in ("iris", "default") and load_iris:
            data = load_iris(as_frame=True)
            df = data.frame.copy()
            # pewne wersje zostawiają target w data.frame; ujednól
            if "target" not in df.columns:
                df["target"] = data.target
            return self._normalize_df(_sanitize_df(df))

        if name in ("wine", "wine_quality", "vinos") and load_wine:
            data = load_wine(as_frame=True)
            df = data.frame.copy()
            if "target" not in df.columns:
                df["target"] = data.target
            return self._normalize_df(_sanitize_df(df))

        if name in ("california", "housing") and fetch_california_housing:
            data = fetch_california_housing(as_frame=True)
            df = data.frame.copy()
            # nazwa targetu zgodna z popularnymi przykładami
            df["MedHouseVal"] = data.target
            return self._normalize_df(_sanitize_df(df))

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
        return self._normalize_df(_sanitize_df(df))
