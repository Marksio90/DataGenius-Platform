# core/services/data_service.py
"""
DataService – warstwa dostępu do danych dla TMIV – Advanced ML Platform.

Zakres:
- Wczytywanie plików (CSV/XLSX/JSON/Parquet) oraz przykładowego zbioru.
- Normalizacja nazw kolumn (snake_case, unikalne).
- Odcisk danych (fingerprint) do cache/eksportów.
- Budowa raportu EDA HTML (ydata-profiling / pandas-profiling – jeśli dostępne).

Implementuje minimalny kontrakt `IDataService` z `core/interfaces.py`.
Nie posiada twardych zależności na Streamlit/UI.

Uwaga:
- Normalizacja nazw kolumn jest wykonywana zaraz po wczytaniu (spójność w całej aplikacji).
- Jeżeli profiling nie jest dostępny (brak pakietu), `profile_html` zwraca None.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import pandas as pd

# backend loader + kolumny
try:
    from backend.file_upload import load_any_file, load_example_dataset, normalize_columns
except Exception as e:  # pragma: no cover
    raise ImportError("backend.file_upload is required for DataService.") from e

# opcjonalny profiler EDA (łagodny fallback)
try:
    from backend.profiling_eda import build_profile_html as _build_profile_html  # type: ignore
except Exception:  # pragma: no cover
    _build_profile_html = None  # type: ignore

# fingerprint/cache – korzystamy z lekkiej fasady
from .cache_service import CacheService


class DataService:
    """Prosta implementacja IDataService."""

    def __init__(self) -> None:
        self._cache = CacheService()

    # --------------------------
    # Public API
    # --------------------------

    def load_any(self, src: Any) -> Tuple[pd.DataFrame, str]:
        """
        Wczytaj plik dowolnego wspieranego typu i zwróć (df, nazwa_pliku).
        Dodatkowo normalizuje nazwy kolumn (snake_case, unikalne).
        """
        df, name = load_any_file(src)
        if df is None or df.empty:
            raise ValueError(f"Pusty lub niepoprawny plik danych: {name}")
        df = self._sanitize(df)
        return df, name

    def load_example(self) -> pd.DataFrame:
        """Załaduj przykładowy dataset repo i znormalizuj kolumny."""
        df = load_example_dataset()
        if df is None or df.empty:
            raise ValueError("Przykładowy dataset jest pusty lub niedostępny.")
        return self._sanitize(df)

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unikalne, snake_case nazwy kolumn."""
        return normalize_columns(df)

    def fingerprint(self, df: pd.DataFrame) -> str:
        """Stabilny odcisk danych (hash struktury + próbki zawartości)."""
        return self._cache.df_fingerprint(df)

    def profile_html(self, df: pd.DataFrame, *, title: str = "TMIV – EDA Profile") -> Optional[str]:
        """
        Zbuduj raport EDA HTML; zwraca absolutną ścieżkę do pliku lub None,
        gdy profiling nie jest zainstalowany.
        """
        if df is None or df.empty:
            return None
        if _build_profile_html is None:
            return None
        # Profiling i tak sam robi próbkowanie i cache; tu tylko delegacja:
        try:
            return _build_profile_html(df, title=title)
        except Exception:
            # Utrzymujemy defensywny charakter: w UI pokażemy info, że profil nie powstał
            return None

    # --------------------------
    # Helpers
    # --------------------------

    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimalna sanityzacja:
        - normalizacja nazw
        - konwersja kategorii do string (zachowanie spójności z loaderem)
        """
        out = normalize_columns(df)
        try:
            # Zamień kategorię -> str (bez utraty informacji o brakach)
            for c in out.select_dtypes(include=["category"]).columns:
                out[c] = out[c].astype("string")
        except Exception:
            pass
        return out


__all__ = ["DataService"]
