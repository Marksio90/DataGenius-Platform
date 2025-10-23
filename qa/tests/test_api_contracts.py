"""
Testy kontraktów API modułów backend.

Sprawdzają:
- Signatury funkcji
- Typy zwracane
- Obsługa błędów
"""

import pandas as pd
import pytest

from backend.dtype_sanitizer import sanitize_dataframe
from backend.eda_integration import compute_basic_statistics, perform_eda
from backend.file_upload import load_file
from backend.utils import get_column_stats_summary, hash_dataframe
from backend.utils_target import detect_problem_type, detect_target_column


class TestUtilsContracts:
    """Testy kontraktów utils."""

    def test_hash_dataframe_returns_string(self):
        """Test czy hash_dataframe zwraca string."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = hash_dataframe(df)

        assert isinstance(result, str)
        assert len(result) == 64  # SHA256

    def test_get_column_stats_returns_dict(self):
        """Test czy get_column_stats_summary zwraca dict."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        result = get_column_stats_summary(df)

        assert isinstance(result, dict)
        assert 'n_rows' in result
        assert 'n_columns' in result
        assert result['n_rows'] == 3


class TestSanitizerContracts:
    """Testy kontraktów sanitizer."""

    def test_sanitize_returns_tuple(self):
        """Test czy sanitize_dataframe zwraca tuple."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = sanitize_dataframe(df)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)

    def test_sanitize_handles_empty_df(self):
        """Test obsługi pustego DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(Exception):  # Powinien rzucić wyjątek
            sanitize_dataframe(df)


class TestEDAContracts:
    """Testy kontraktów EDA."""

    def test_compute_basic_statistics_returns_dict(self):
        """Test czy compute_basic_statistics zwraca dict."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        result = compute_basic_statistics(df)

        assert isinstance(result, dict)
        assert 'n_rows' in result
        assert 'n_columns' in result

    def test_perform_eda_returns_dict(self):
        """Test czy perform_eda zwraca dict."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        result = perform_eda(df, include_profiling=False)

        assert isinstance(result, dict)
        assert 'basic_stats' in result
        assert 'numeric_stats' in result


class TestTargetDetectionContracts:
    """Testy kontraktów detekcji target."""

    def test_detect_target_returns_string_or_none(self):
        """Test czy detect_target_column zwraca string lub None."""
        df = pd.DataFrame({'feature': [1, 2, 3], 'target': [0, 1, 0]})
        result = detect_target_column(df)

        assert result is None or isinstance(result, str)

    def test_detect_problem_type_returns_tuple(self):
        """Test czy detect_problem_type zwraca tuple."""
        df = pd.DataFrame({'target': [0, 1, 0, 1]})
        result = detect_problem_type(df, 'target')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

    def test_detect_problem_type_classification(self):
        """Test wykrywania klasyfikacji."""
        df = pd.DataFrame({'target': [0, 1, 0, 1, 0, 1]})
        problem_type, metadata = detect_problem_type(df, 'target')

        assert problem_type == 'binary_classification'
        assert 'classes' in metadata
        assert len(metadata['classes']) == 2

    def test_detect_problem_type_regression(self):
        """Test wykrywania regresji."""
        df = pd.DataFrame({'target': [1.5, 2.3, 3.7, 4.2, 5.1]})
        problem_type, metadata = detect_problem_type(df, 'target')

        assert problem_type == 'regression'
        assert 'min' in metadata
        assert 'max' in metadata


class TestExportContracts:
    """Testy kontraktów eksportu."""

    def test_export_creates_files(self):
        """Test czy eksport tworzy pliki."""
        from backend.export_everything import ArtifactExporter

        exporter = ArtifactExporter()

        # Minimalne dane
        results = []
        pipeline_results = {
            'problem_type': 'classification',
            'n_models_trained': 0,
            'n_features': 5,
        }

        # Powinien utworzyć plik
        zip_path = exporter.export_to_zip(results, pipeline_results)

        assert zip_path.exists()
        assert zip_path.suffix == '.zip'

        # Cleanup
        zip_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])