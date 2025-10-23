"""
MLflow Integration - experiment tracking i model registry.

Funkcjonalności:
- Automatyczne logowanie eksperymentów
- Tracking metryk i parametrów
- Artifact logging (modele, wykresy)
- Model registry
- Model versioning
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    import mlflow.catboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MLflow nie jest zainstalowane")

from backend.error_handler import handle_errors
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MLflowTracker:
    """
    MLflow experiment tracker.
    
    Automatycznie loguje:
    - Parametry modelu
    - Metryki (train/val/test)
    - Model artifacts
    - Wykresy
    - Dataset info
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Inicjalizacja MLflow tracker.

        Args:
            experiment_name: Nazwa eksperymentu
            tracking_uri: URI MLflow tracking server (None = local)
            artifact_location: Lokalizacja artifacts
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow nie jest zainstalowane: pip install mlflow")

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri or "file:./mlruns"
        self.artifact_location = artifact_location

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                self.experiment = mlflow.get_experiment(experiment_id)
        except Exception as e:
            logger.warning(f"Błąd tworzenia eksperymentu: {e}")
            self.experiment = None

        self.current_run = None
        self.run_id = None

        logger.info(f"MLflow Tracker zainicjalizowany (experiment={experiment_name})")

    @handle_errors(show_in_ui=False)
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None
    ):
        """
        Rozpoczyna nowy run.

        Args:
            run_name: Nazwa run
            tags: Tagi
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id if self.experiment else None,
            run_name=run_name,
            tags=tags
        )

        self.run_id = self.current_run.info.run_id

        logger.info(f"MLflow run rozpoczęty: {run_name} (id={self.run_id})")

    def end_run(self):
        """Kończy aktualny run."""
        if self.current_run:
            mlflow.end_run()
            logger.info(f"MLflow run zakończony: {self.run_id}")
            self.current_run = None
            self.run_id = None

    def log_params(self, params: Dict):
        """
        Loguje parametry.

        Args:
            params: Słownik z parametrami
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - parametry nie zostaną zapisane")
            return

        # Flatten nested params
        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Nie udało się zalogować parametru {key}: {e}")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Loguje metryki.

        Args:
            metrics: Słownik z metrykami
            step: Krok (dla time series metryk)
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - metryki nie zostaną zapisane")
            return

        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float, np.number)):
                    mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.warning(f"Nie udało się zalogować metryki {key}: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None
    ):
        """
        Loguje model.

        Args:
            model: Model do zalogowania
            artifact_path: Ścieżka w artifacts
            registered_model_name: Nazwa w registry (None = nie rejestruj)
            signature: Model signature
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - model nie zostanie zapisany")
            return

        model_class_name = model.__class__.__name__

        try:
            # Detect model type and use appropriate logger
            if 'XGB' in model_class_name:
                mlflow.xgboost.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature
                )
            elif 'LGBM' in model_class_name or 'LightGBM' in model_class_name:
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature
                )
            elif 'CatBoost' in model_class_name:
                mlflow.catboost.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature
                )
            else:
                # Sklearn or other
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature
                )

            logger.info(f"Model zalogowany: {artifact_path}")

        except Exception as e:
            logger.warning(f"Nie udało się zalogować modelu: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Loguje artifact (plik).

        Args:
            local_path: Ścieżka lokalna do pliku
            artifact_path: Ścieżka w artifacts (None = root)
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - artifact nie zostanie zapisany")
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Artifact zalogowany: {local_path}")
        except Exception as e:
            logger.warning(f"Nie udało się zalogować artifact: {e}")

    def log_figure(self, figure: Any, filename: str):
        """
        Loguje wykres matplotlib/plotly.

        Args:
            figure: Figure object
            filename: Nazwa pliku
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - wykres nie zostanie zapisany")
            return

        try:
            mlflow.log_figure(figure, filename)
            logger.debug(f"Wykres zalogowany: {filename}")
        except Exception as e:
            logger.warning(f"Nie udało się zalogować wykresu: {e}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Loguje słownik jako JSON.

        Args:
            dictionary: Słownik
            filename: Nazwa pliku
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - słownik nie zostanie zapisany")
            return

        try:
            mlflow.log_dict(dictionary, filename)
            logger.debug(f"Dict zalogowany: {filename}")
        except Exception as e:
            logger.warning(f"Nie udało się zalogować dict: {e}")

    def log_text(self, text: str, filename: str):
        """
        Loguje tekst.

        Args:
            text: Tekst
            filename: Nazwa pliku
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - tekst nie zostanie zapisany")
            return

        try:
            mlflow.log_text(text, filename)
            logger.debug(f"Tekst zalogowany: {filename}")
        except Exception as e:
            logger.warning(f"Nie udało się zalogować tekstu: {e}")

    def set_tags(self, tags: Dict):
        """
        Ustawia tagi dla run.

        Args:
            tags: Słownik z tagami
        """
        if not self.current_run:
            logger.warning("Brak aktywnego run - tagi nie zostaną zapisane")
            return

        for key, value in tags.items():
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Nie udało się ustawić tagu {key}: {e}")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Spłaszcza zagnieżdżony słownik.

        Args:
            d: Słownik
            parent_key: Klucz rodzica
            sep: Separator

        Returns:
            Dict: Spłaszczony słownik
        """
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def get_run_info(self) -> Dict:
        """
        Zwraca informacje o aktualnym run.

        Returns:
            Dict: Informacje o run
        """
        if not self.current_run:
            return {}

        return {
            'run_id': self.run_id,
            'experiment_id': self.current_run.info.experiment_id,
            'status': self.current_run.info.status,
            'start_time': self.current_run.info.start_time,
            'artifact_uri': self.current_run.info.artifact_uri
        }

    @staticmethod
    def search_runs(
        experiment_names: Optional[List[str]] = None,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[Dict]:
        """
        Wyszukuje runs.

        Args:
            experiment_names: Lista nazw eksperymentów
            filter_string: Filtr (MLflow syntax)
            max_results: Maksymalna liczba wyników

        Returns:
            List[Dict]: Lista runs
        """
        if not MLFLOW_AVAILABLE:
            return []

        try:
            runs = mlflow.search_runs(
                experiment_names=experiment_names,
                filter_string=filter_string,
                max_results=max_results
            )

            return runs.to_dict('records')

        except Exception as e:
            logger.warning(f"Błąd wyszukiwania runs: {e}")
            return []

    @staticmethod
    def load_model(model_uri: str) -> Any:
        """
        Ładuje model z MLflow.

        Args:
            model_uri: URI modelu (np. "runs:/<run_id>/model")

        Returns:
            Any: Model
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow nie jest zainstalowane")

        try:
            # Try different flavors
            try:
                return mlflow.sklearn.load_model(model_uri)
            except:
                pass

            try:
                return mlflow.xgboost.load_model(model_uri)
            except:
                pass

            try:
                return mlflow.lightgbm.load_model(model_uri)
            except:
                pass

            try:
                return mlflow.catboost.load_model(model_uri)
            except:
                pass

            # Generic pyfunc
            return mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            logger.error(f"Błąd ładowania modelu: {e}")
            raise


# Context manager dla automatycznego zarządzania runs
class MLflowRunContext:
    """Context manager dla MLflow runs."""

    def __init__(
        self,
        tracker: MLflowTracker,
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None
    ):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags

    def __enter__(self):
        self.tracker.start_run(run_name=self.run_name, tags=self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_run()