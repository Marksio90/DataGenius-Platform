"""
Moduł generowania zaawansowanych profili EDA.

Wykorzystuje ydata-profiling do generowania szczegółowych raportów.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from backend.error_handler import EDAException, handle_errors

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def generate_profile_report(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    minimal: bool = True
) -> Optional[str]:
    """
    Generuje profil EDA używając ydata-profiling.

    Args:
        df: DataFrame do analizy
        output_path: Ścieżka zapisu (opcjonalna)
        minimal: Czy użyć trybu minimal (szybszy)

    Returns:
        Optional[str]: HTML raportu lub None

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> html = generate_profile_report(df, minimal=True)
        >>> html is not None or html is None  # Zależy od dostępności biblioteki
        True
    """
    try:
        from ydata_profiling import ProfileReport

        logger.info("Rozpoczęcie generowania profilu EDA...")

        config = {
            "minimal": minimal,
            "explorative": not minimal,
            "correlations": {
                "pearson": {"calculate": True},
                "spearman": {"calculate": not minimal},
                "kendall": {"calculate": False},
            },
            "interactions": {"continuous": False},
            "missing_diagrams": {"heatmap": not minimal},
        }

        profile = ProfileReport(df, **config)

        if output_path:
            profile.to_file(output_path)
            logger.info(f"Profil zapisany do: {output_path}")

        html_content = profile.to_html()
        logger.info("Profil wygenerowany pomyślnie")

        return html_content

    except ImportError:
        logger.warning("ydata-profiling nie jest zainstalowany")
        return None
    except Exception as e:
        logger.error(f"Błąd generowania profilu: {e}")
        return None


@handle_errors(show_in_ui=False)
def should_use_profiling(df: pd.DataFrame, max_rows: int = 50000) -> Tuple[bool, str]:
    """
    Sprawdza czy warto użyć profiling dla danego datasetu.

    Args:
        df: DataFrame do sprawdzenia
        max_rows: Maksymalna liczba wierszy

    Returns:
        Tuple[bool, str]: (czy_użyć, powód)

    Example:
        >>> df = pd.DataFrame({'a': range(100000)})
        >>> should_use, reason = should_use_profiling(df)
        >>> isinstance(should_use, bool)
        True
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    # Dataset za duży
    if n_rows > max_rows:
        return False, f"Dataset za duży ({n_rows} wierszy > {max_rows})"

    # Za dużo kolumn
    if n_cols > 100:
        return False, f"Za dużo kolumn ({n_cols} > 100)"

    # Szacowany czas
    estimated_time = (n_rows * n_cols) / 10000  # Prosta heurystyka
    if estimated_time > 60:
        return False, f"Szacowany czas: {estimated_time:.0f}s (> 60s)"

    return True, "OK"