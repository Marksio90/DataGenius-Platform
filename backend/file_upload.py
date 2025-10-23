"""
Moduł obsługi wczytywania plików różnych formatów.

Wspierane formaty:
- CSV (.csv)
- Excel (.xlsx, .xls)
- Parquet (.parquet)
- JSON (.json)

Funkcjonalności:
- Automatyczna detekcja separatora CSV
- Obsługa różnych encodingów
- Walidacja struktury danych
- Raportowanie błędów
"""

import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from backend.error_handler import DataLoadException, handle_errors

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def detect_csv_separator(file_content: str, sample_size: int = 5) -> str:
    """
    Automatycznie wykrywa separator w pliku CSV.

    Args:
        file_content: Zawartość pliku jako string
        sample_size: Liczba linii do analizy

    Returns:
        str: Wykryty separator (domyślnie ',')

    Example:
        >>> content = "a;b;c\\n1;2;3"
        >>> detect_csv_separator(content)
        ';'
    """
    lines = file_content.split('\n')[:sample_size]
    
    # Kandydaci na separator
    separators = [',', ';', '\t', '|']
    separator_counts = {}
    
    for sep in separators:
        counts = [line.count(sep) for line in lines if line.strip()]
        if counts and len(set(counts)) == 1 and counts[0] > 0:
            separator_counts[sep] = counts[0]
    
    if separator_counts:
        return max(separator_counts, key=separator_counts.get)
    
    return ','


@handle_errors(show_in_ui=False)
def try_read_csv_with_encoding(
    file_obj: Union[BytesIO, StringIO],
    encoding: str,
    separator: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Próbuje wczytać CSV z podanym encodingiem.

    Args:
        file_obj: Obiekt pliku
        encoding: Encoding do próby
        separator: Separator (None = auto-detect)

    Returns:
        Optional[pd.DataFrame]: DataFrame lub None przy błędzie
    """
    try:
        file_obj.seek(0)
        
        # Jeśli separator nie podany, wykryj automatycznie
        if separator is None:
            if isinstance(file_obj, BytesIO):
                content = file_obj.read().decode(encoding, errors='ignore')
                file_obj.seek(0)
            else:
                content = file_obj.getvalue()
            separator = detect_csv_separator(content)
        
        df = pd.read_csv(file_obj, encoding=encoding, sep=separator, low_memory=False)
        logger.info(f"Pomyślnie wczytano CSV z encodingiem {encoding} i separatorem '{separator}'")
        return df
    except Exception as e:
        logger.debug(f"Nie udało się wczytać z encodingiem {encoding}: {e}")
        return None


def load_file(
    uploaded_file,
    file_extension: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Główna funkcja wczytująca plik różnych formatów.

    Args:
        uploaded_file: Obiekt pliku z Streamlit
        file_extension: Rozszerzenie (opcjonalne, wykrywane automatycznie)

    Returns:
        Tuple[pd.DataFrame, dict]: (DataFrame, metadane)

    Raises:
        DataLoadException: Gdy nie można wczytać pliku

    Example:
        >>> # W kontekście Streamlit
        >>> df, metadata = load_file(uploaded_file)
    """
    if uploaded_file is None:
        raise DataLoadException("Nie wybrano pliku do wczytania")
    
    # Wykryj rozszerzenie
    if file_extension is None:
        file_extension = Path(uploaded_file.name).suffix.lower()
    
    metadata = {
        "filename": uploaded_file.name,
        "extension": file_extension,
        "size_bytes": uploaded_file.size if hasattr(uploaded_file, 'size') else 0,
    }
    
    logger.info(f"Rozpoczęcie wczytywania pliku: {uploaded_file.name} ({file_extension})")
    
    try:
        # CSV
        if file_extension == '.csv':
            df = _load_csv(uploaded_file)
        
        # Excel
        elif file_extension in ['.xlsx', '.xls']:
            df = _load_excel(uploaded_file)
        
        # Parquet
        elif file_extension == '.parquet':
            df = _load_parquet(uploaded_file)
        
        # JSON
        elif file_extension == '.json':
            df = _load_json(uploaded_file)
        
        else:
            raise DataLoadException(
                f"Nieobsługiwany format pliku: {file_extension}. "
                f"Wspierane: .csv, .xlsx, .xls, .parquet, .json"
            )
        
        # Walidacja podstawowa
        if df.empty:
            raise DataLoadException("Wczytany plik jest pusty")
        
        metadata.update({
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
        })
        
        logger.info(f"Pomyślnie wczytano: {len(df)} wierszy, {len(df.columns)} kolumn")
        return df, metadata
    
    except DataLoadException:
        raise
    except Exception as e:
        raise DataLoadException(f"Błąd wczytywania pliku: {str(e)}")


def _load_csv(uploaded_file) -> pd.DataFrame:
    """
    Wczytuje plik CSV z automatyczną detekcją encodingu i separatora.

    Args:
        uploaded_file: Obiekt pliku

    Returns:
        pd.DataFrame: Wczytany DataFrame

    Raises:
        DataLoadException: Gdy nie można wczytać pliku
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    file_obj = BytesIO(uploaded_file.read())
    
    for encoding in encodings:
        df = try_read_csv_with_encoding(file_obj, encoding)
        if df is not None:
            return df
    
    raise DataLoadException(
        "Nie udało się wczytać pliku CSV. Sprawdź encoding i format pliku."
    )


def _load_excel(uploaded_file) -> pd.DataFrame:
    """
    Wczytuje plik Excel (xlsx/xls).

    Args:
        uploaded_file: Obiekt pliku

    Returns:
        pd.DataFrame: Wczytany DataFrame

    Raises:
        DataLoadException: Gdy nie można wczytać pliku
    """
    try:
        # Próba wczytania pierwszego arkusza
        df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
        logger.info(f"Wczytano arkusz Excel: {len(df)} wierszy")
        return df
    except Exception as e:
        raise DataLoadException(f"Błąd wczytywania Excel: {str(e)}")


def _load_parquet(uploaded_file) -> pd.DataFrame:
    """
    Wczytuje plik Parquet.

    Args:
        uploaded_file: Obiekt pliku

    Returns:
        pd.DataFrame: Wczytany DataFrame

    Raises:
        DataLoadException: Gdy nie można wczytać pliku
    """
    try:
        df = pd.read_parquet(BytesIO(uploaded_file.read()))
        logger.info(f"Wczytano Parquet: {len(df)} wierszy")
        return df
    except Exception as e:
        raise DataLoadException(f"Błąd wczytywania Parquet: {str(e)}")


def _load_json(uploaded_file) -> pd.DataFrame:
    """
    Wczytuje plik JSON.

    Args:
        uploaded_file: Obiekt pliku

    Returns:
        pd.DataFrame: Wczytany DataFrame

    Raises:
        DataLoadException: Gdy nie można wczytać pliku
    """
    try:
        content = uploaded_file.read().decode('utf-8')
        df = pd.read_json(StringIO(content))
        logger.info(f"Wczytano JSON: {len(df)} wierszy")
        return df
    except Exception as e:
        raise DataLoadException(f"Błąd wczytywania JSON: {str(e)}")


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Waliduje podstawową strukturę DataFrame.

    Args:
        df: DataFrame do walidacji

    Returns:
        Tuple[bool, list]: (czy_poprawny, lista_ostrzeżeń)

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> is_valid, warnings = validate_dataframe(df)
        >>> is_valid
        True
    """
    warnings = []
    
    # Sprawdź minimalny rozmiar
    if len(df) < 10:
        warnings.append("⚠️ Dataset ma mniej niż 10 wierszy - wyniki mogą być niemiarodajne")
    
    # Sprawdź duplikaty nazw kolumn
    if len(df.columns) != len(set(df.columns)):
        warnings.append("⚠️ Wykryto duplikaty nazw kolumn - zostaną unikalizowane")
    
    # Sprawdź kolumny bez nazw
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        warnings.append(f"⚠️ Wykryto {len(unnamed_cols)} kolumn bez nazw")
    
    # Sprawdź brakujące wartości
    missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 50:
        warnings.append(f"⚠️ Ponad 50% wartości to braki ({missing_pct:.1f}%)")
    
    # Sprawdź constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        warnings.append(f"⚠️ Wykryto {len(constant_cols)} kolumn stałych (jedna wartość)")
    
    is_valid = len(warnings) == 0 or all('⚠️' in w for w in warnings)
    
    return is_valid, warnings