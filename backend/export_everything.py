"""
Eksport wszystkich artefaktów do ZIP.

Funkcjonalności:
- Eksport modeli (joblib)
- Eksport metryk (JSON/CSV)
- Eksport wykresów (PNG)
- Eksport konfiguracji
- README z instrukcjami
"""

import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZipFile

import joblib
import pandas as pd

from backend.error_handler import ExportException, handle_errors
from backend.utils import get_timestamp, sanitize_filename
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ArtifactExporter:
    """
    Eksporter artefaktów ML do ZIP.
    
    Eksportuje:
    - Modele (.joblib)
    - Metryki (.json, .csv)
    - Wykresy (.png)
    - Konfiguracje (.yml)
    - README.txt
    """

    def __init__(self, export_dir: Optional[Path] = None):
        """
        Inicjalizacja exportera.

        Args:
            export_dir: Katalog eksportu (None = artifacts/)
        """
        self.export_dir = export_dir or settings.artifacts_dir
        self.export_dir.mkdir(exist_ok=True)

    @handle_errors(show_in_ui=False)
    def export_to_zip(
        self,
        results: List[Dict],
        pipeline_results: Dict,
        plots: Optional[Dict[str, BytesIO]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Eksportuje wszystkie artefakty do ZIP.

        Args:
            results: Lista wyników modeli
            pipeline_results: Wyniki pipeline
            plots: Słownik z wykresami
            filename: Nazwa pliku ZIP (None = auto)

        Returns:
            Path: Ścieżka do utworzonego ZIP

        Raises:
            ExportException: Gdy eksport się nie powiedzie

        Example:
            >>> exporter = ArtifactExporter()
            >>> results = [{'model_name': 'M1', 'model': None, 'test_metrics': {}}]
            >>> pipeline = {'problem_type': 'classification'}
            >>> # zip_path = exporter.export_to_zip(results, pipeline)
        """
        if filename is None:
            timestamp = get_timestamp()
            filename = f"tmiv_artifacts_{timestamp}.zip"

        filename = sanitize_filename(filename)
        zip_path = self.export_dir / filename

        logger.info(f"Tworzenie archiwum ZIP: {zip_path}")

        try:
            with ZipFile(zip_path, 'w') as zipf:
                # 1. Eksportuj modele
                self._export_models(zipf, results)

                # 2. Eksportuj metryki
                self._export_metrics(zipf, results, pipeline_results)

                # 3. Eksportuj wykresy
                if plots:
                    self._export_plots(zipf, plots)

                # 4. Eksportuj konfigurację
                self._export_config(zipf, pipeline_results)

                # 5. Utwórz README
                self._create_readme(zipf, pipeline_results)

            logger.info(f"Archiwum ZIP utworzone: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
            return zip_path

        except Exception as e:
            raise ExportException(f"Błąd tworzenia archiwum ZIP: {e}")

    def _export_models(self, zipf: ZipFile, results: List[Dict]) -> None:
        """Eksportuje modele do ZIP."""
        logger.info("Eksport modeli...")

        for result in results:
            model_name = result['model_name']
            model = result.get('model')

            if model is None:
                logger.warning(f"Model {model_name} jest None - pomijam")
                continue

            try:
                # Serializuj model do BytesIO
                model_bytes = BytesIO()
                joblib.dump(model, model_bytes)
                model_bytes.seek(0)

                # Dodaj do ZIP
                filename = f"models/{sanitize_filename(model_name)}.joblib"
                zipf.writestr(filename, model_bytes.read())
                logger.debug(f"Wyeksportowano model: {filename}")

            except Exception as e:
                logger.warning(f"Nie udało się wyeksportować modelu {model_name}: {e}")

    def _export_metrics(
        self,
        zipf: ZipFile,
        results: List[Dict],
        pipeline_results: Dict
    ) -> None:
        """Eksportuje metryki do ZIP."""
        logger.info("Eksport metryk...")

        # 1. JSON z pełnymi wynikami
        try:
            # Przygotuj dane (usuń modele - nie są serializowalne)
            exportable_results = []
            for result in results:
                result_copy = result.copy()
                result_copy.pop('model', None)
                exportable_results.append(result_copy)

            metrics_json = json.dumps(exportable_results, indent=2, default=str)
            zipf.writestr("metrics/results.json", metrics_json)
            logger.debug("Wyeksportowano results.json")

        except Exception as e:
            logger.warning(f"Nie udało się wyeksportować JSON: {e}")

        # 2. CSV z metrykami
        try:
            rows = []
            for result in results:
                row = {'Model': result['model_name']}
                row.update(result.get('test_metrics', {}))
                rows.append(row)

            df_metrics = pd.DataFrame(rows)
            csv_bytes = df_metrics.to_csv(index=False)
            zipf.writestr("metrics/metrics.csv", csv_bytes)
            logger.debug("Wyeksportowano metrics.csv")

        except Exception as e:
            logger.warning(f"Nie udało się wyeksportować CSV: {e}")

        # 3. Podsumowanie pipeline
        try:
            summary = {
                "problem_type": pipeline_results.get('problem_type'),
                "n_models_trained": pipeline_results.get('n_models_trained'),
                "n_features": pipeline_results.get('n_features'),
                "best_model": pipeline_results.get('best_model_name'),
                "export_timestamp": datetime.now().isoformat(),
            }

            summary_json = json.dumps(summary, indent=2)
            zipf.writestr("summary.json", summary_json)
            logger.debug("Wyeksportowano summary.json")

        except Exception as e:
            logger.warning(f"Nie udało się wyeksportować podsumowania: {e}")

    def _export_plots(self, zipf: ZipFile, plots: Dict[str, BytesIO]) -> None:
        """Eksportuje wykresy do ZIP."""
        logger.info(f"Eksport wykresów: {len(plots)} plików...")

        for plot_name, plot_bytes in plots.items():
            try:
                if plot_bytes is None:
                    continue

                plot_bytes.seek(0)
                filename = f"plots/{sanitize_filename(plot_name)}.png"
                zipf.writestr(filename, plot_bytes.read())
                logger.debug(f"Wyeksportowano wykres: {filename}")

            except Exception as e:
                logger.warning(f"Nie udało się wyeksportować wykresu {plot_name}: {e}")

    def _export_config(self, zipf: ZipFile, pipeline_results: Dict) -> None:
        """Eksportuje konfigurację do ZIP."""
        logger.info("Eksport konfiguracji...")

        try:
            config = {
                "training_plan": pipeline_results.get('training_plan'),
                "feature_names": pipeline_results.get('feature_names'),
                "class_names": pipeline_results.get('class_names'),
            }

            config_json = json.dumps(config, indent=2, default=str)
            zipf.writestr("config/training_config.json", config_json)
            logger.debug("Wyeksportowano training_config.json")

        except Exception as e:
            logger.warning(f"Nie udało się wyeksportować konfiguracji: {e}")

    def _create_readme(self, zipf: ZipFile, pipeline_results: Dict) -> None:
        """Tworzy README w ZIP."""
        logger.info("Tworzenie README...")

        readme_content = f"""
TMIV - The Most Important Variables
Eksport Artefaktów ML
=====================================

Data eksportu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Wersja aplikacji: {settings.app_version}

STRUKTURA ARCHIWUM
------------------
models/          - Wytrenowane modele (.joblib)
metrics/         - Metryki i wyniki (JSON, CSV)
plots/           - Wykresy i wizualizacje (PNG)
config/          - Konfiguracja treningu
README.txt       - Ten plik
summary.json     - Podsumowanie pipeline

INFORMACJE O PIPELINE
---------------------
Typ problemu: {pipeline_results.get('problem_type', 'N/A')}
Liczba modeli: {pipeline_results.get('n_models_trained', 'N/A')}
Liczba features: {pipeline_results.get('n_features', 'N/A')}
Najlepszy model: {pipeline_results.get('best_model_name', 'N/A')}

JAK UŻYĆ MODELI
---------------
1. Zainstaluj wymagane biblioteki:
   pip install scikit-learn xgboost lightgbm catboost joblib

2. Wczytaj model:
   import joblib
   model = joblib.load('models/nazwa_modelu.joblib')

3. Wykonaj predykcję:
   predictions = model.predict(X_test)
   # Dla klasyfikacji z prawdopodobieństwami:
   probabilities = model.predict_proba(X_test)

UWAGI
-----
- Modele wymagają identycznego preprocessingu jak podczas treningu
- Sprawdź feature_names w config/training_config.json
- Dla klasyfikacji sprawdź class_names

WSPARCIE
--------
Dokumentacja: docs/README.md
Issues: https://github.com/your-repo/issues
"""

        zipf.writestr("README.txt", readme_content.strip())
        logger.debug("Utworzono README.txt")


@handle_errors(show_in_ui=False)
def create_sample_export() -> Path:
    """
    Tworzy przykładowy eksport (do testów).

    Returns:
        Path: Ścieżka do utworzonego ZIP
    """
    exporter = ArtifactExporter()

    # Przykładowe dane
    results = [
        {
            'model_name': 'SampleModel',
            'model': None,
            'test_metrics': {'accuracy': 0.85},
            'training_time': 10.5
        }
    ]

    pipeline_results = {
        'problem_type': 'binary_classification',
        'n_models_trained': 1,
        'n_features': 10,
        'best_model_name': 'SampleModel',
    }

    return exporter.export_to_zip(results, pipeline_results)