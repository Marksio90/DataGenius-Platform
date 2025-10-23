"""
Auto Retrainer - automatyczny retraining modeli.

Funkcjonalności:
- Scheduled retraining
- Drift-triggered retraining
- Performance-triggered retraining
- Automatic model deployment
- Rollback mechanism
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import schedule
import time

from backend.error_handler import handle_errors
from backend.mlops.drift_detector import DriftDetector
from backend.mlops.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """
    Automatyczny retraining system.
    
    Triggery:
    - Time-based (co X dni)
    - Drift-based (wykrycie drift)
    - Performance-based (degradacja metryk)
    - Manual trigger
    """

    def __init__(
        self,
        model_name: str,
        training_function: Any,
        registry: ModelRegistry,
        retraining_frequency_days: Optional[int] = 7,
        drift_threshold: float = 0.2,
        performance_threshold: float = 0.05,
        auto_deploy: bool = False
    ):
        """
        Inicjalizacja auto retrainer.

        Args:
            model_name: Nazwa modelu
            training_function: Funkcja trenująca model (signature: fn(data) -> model)
            registry: Model registry
            retraining_frequency_days: Częstotliwość retreningu (dni)
            drift_threshold: Threshold dla drift trigger
            performance_threshold: Threshold dla performance trigger
            auto_deploy: Czy automatycznie deployować
        """
        self.model_name = model_name
        self.training_function = training_function
        self.registry = registry
        self.retraining_frequency_days = retraining_frequency_days
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.auto_deploy = auto_deploy

        # State
        self.last_retraining_date = None
        self.retraining_history = []
        self.current_model = None
        self.baseline_data = None
        self.drift_detector = None

        logger.info(f"Auto Retrainer zainicjalizowany dla {model_name}")

    def set_baseline(
        self,
        data: pd.DataFrame,
        target: Optional[np.ndarray] = None
    ):
        """
        Ustawia dane baseline dla drift detection.

        Args:
            data: Dane baseline
            target: Target baseline
        """
        self.baseline_data = data
        self.drift_detector = DriftDetector(
            reference_data=data,
            reference_target=target,
            threshold_psi=self.drift_threshold
        )

        logger.info("Baseline ustawiony dla drift detection")

    @handle_errors(show_in_ui=False)
    def check_retraining_needed(
        self,
        current_data: pd.DataFrame,
        current_target: Optional[np.ndarray] = None,
        current_model: Optional[Any] = None
    ) -> Tuple[bool, List[str]]:
        """
        Sprawdza czy retraining jest potrzebny.

        Args:
            current_data: Aktualne dane
            current_target: Aktualny target
            current_model: Aktualny model (dla concept drift)

        Returns:
            Tuple[bool, List[str]]: (czy_potrzebny, lista_powodów)
        """
        reasons = []

        # 1. Time-based trigger
        if self.retraining_frequency_days and self.last_retraining_date:
            days_since_last = (datetime.now() - self.last_retraining_date).days

            if days_since_last >= self.retraining_frequency_days:
                reasons.append(f"scheduled_retraining_{days_since_last}_days")

        # 2. Drift-based trigger
        if self.drift_detector and current_data is not None:
            try:
                drift_report = self.drift_detector.full_drift_report(
                    current_model,
                    current_data,
                    current_target
                )

                if drift_report['overall_drift_detected']:
                    for alert in drift_report['alerts']:
                        reasons.append(f"drift_{alert['type']}")

            except Exception as e:
                logger.warning(f"Błąd sprawdzania drift: {e}")

        # 3. Performance-based trigger (manual check needed)
        # User must provide performance metrics

        retraining_needed = len(reasons) > 0

        if retraining_needed:
            logger.info(f"Retraining potrzebny: {', '.join(reasons)}")

        return retraining_needed, reasons

    @handle_errors(show_in_ui=False)
    def retrain_model(
        self,
        training_data: pd.DataFrame,
        reason: str = 'manual',
        metadata: Optional[Dict] = None
    ) -> Any:
        """
        Retrenuje model.

        Args:
            training_data: Dane treningowe
            reason: Powód retreningu
            metadata: Dodatkowe metadata

        Returns:
            Any: Nowy model
        """
        logger.info(f"Rozpoczęcie retreningu: {self.model_name} (reason={reason})")

        try:
            # Train new model
            new_model = self.training_function(training_data)

            # Register in registry
            retraining_metadata = {
                'reason': reason,
                'retraining_date': datetime.now().isoformat(),
                'training_samples': len(training_data),
                **(metadata or {})
            }

            version = self.registry.register_model(
                new_model,
                self.model_name,
                stage='staging',  # Start in staging
                metadata=retraining_metadata
            )

            # Update state
            self.last_retraining_date = datetime.now()
            self.current_model = new_model

            self.retraining_history.append({
                'version': version,
                'date': datetime.now().isoformat(),
                'reason': reason,
                'metadata': retraining_metadata
            })

            logger.info(f"Retraining zakończony: {version}")

            # Auto deploy
            if self.auto_deploy:
                self.deploy_model(version)

            return new_model

        except Exception as e:
            logger.error(f"Błąd retreningu: {e}")
            raise

    def deploy_model(
        self,
        version: str,
        from_stage: str = 'staging',
        to_stage: str = 'production'
    ):
        """
        Deployuje model do production.

        Args:
            version: Wersja modelu
            from_stage: Stage źródłowy
            to_stage: Stage docelowy
        """
        logger.info(f"Deployment modelu: {self.model_name} {version} ({from_stage} -> {to_stage})")

        # Archive old production models
        production_models = self.registry.list_models(stage='production')

        for model_info in production_models:
            if model_info['name'] == self.model_name:
                self.registry.update_stage(
                    model_info['name'],
                    model_info['version'],
                    'archived'
                )

        # Promote new model
        self.registry.update_stage(self.model_name, version, to_stage)

        logger.info(f"Model {version} jest teraz w {to_stage}")

    def rollback(
        self,
        to_version: Optional[str] = None
    ):
        """
        Rollback do poprzedniej wersji.

        Args:
            to_version: Wersja docelowa (None = poprzednia production)
        """
        logger.info(f"Rollback modelu: {self.model_name}")

        if to_version is None:
            # Find previous production version
            archived = self.registry.list_models(stage='archived')
            archived = [m for m in archived if m['name'] == self.model_name]

            if not archived:
                logger.error("Brak modeli archived do rollback")
                return

            # Sort by date
            archived.sort(key=lambda x: x.get('stage_updated_at', ''), reverse=True)
            to_version = archived[0]['version']

        # Archive current production
        production_models = self.registry.list_models(stage='production')

        for model_info in production_models:
            if model_info['name'] == self.model_name:
                self.registry.update_stage(
                    model_info['name'],
                    model_info['version'],
                    'archived'
                )

        # Restore old version
        self.registry.update_stage(self.model_name, to_version, 'production')

        logger.info(f"Rollback zakończony: {to_version} jest teraz production")

    def schedule_retraining(
        self,
        training_data_loader: Any,
        check_interval_hours: int = 24
    ):
        """
        Ustawia scheduled retraining.

        Args:
            training_data_loader: Funkcja zwracająca dane treningowe
            check_interval_hours: Interwał sprawdzania (godziny)
        """
        def job():
            logger.info("Scheduled retraining check...")

            try:
                # Load current data
                current_data = training_data_loader()

                # Check if needed
                needed, reasons = self.check_retraining_needed(current_data)

                if needed:
                    self.retrain_model(
                        current_data,
                        reason=f"scheduled_{','.join(reasons)}"
                    )

            except Exception as e:
                logger.error(f"Błąd scheduled retraining: {e}")

        # Schedule
        schedule.every(check_interval_hours).hours.do(job)

        logger.info(f"Scheduled retraining co {check_interval_hours}h")

        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def get_retraining_history(self) -> List[Dict]:
        """Zwraca historię retreningów."""
        return self.retraining_history

    def get_status(self) -> Dict:
        """
        Zwraca status auto retrainer.

        Returns:
            Dict: Status
        """
        return {
            'model_name': self.model_name,
            'last_retraining_date': self.last_retraining_date.isoformat() if self.last_retraining_date else None,
            'n_retrainings': len(self.retraining_history),
            'auto_deploy': self.auto_deploy,
            'retraining_frequency_days': self.retraining_frequency_days,
            'has_baseline': self.baseline_data is not None
        }


# Helper function dla prostego retreningu
def simple_retraining_workflow(
    model_name: str,
    training_function: Any,
    training_data: pd.DataFrame,
    registry: ModelRegistry,
    deploy_to_production: bool = False
) -> str:
    """
    Prosty workflow retreningu modelu.

    Args:
        model_name: Nazwa modelu
        training_function: Funkcja trenująca
        training_data: Dane treningowe
        registry: Model registry
        deploy_to_production: Czy od razu deployować

    Returns:
        str: Wersja nowego modelu
    """
    logger.info(f"Prosty retraining workflow: {model_name}")

    # Train
    new_model = training_function(training_data)

    # Register
    version = registry.register_model(
        new_model,
        model_name,
        stage='staging',
        metadata={
            'retraining_date': datetime.now().isoformat(),
            'training_samples': len(training_data)
        }
    )

    # Deploy
    if deploy_to_production:
        # Archive old production
        production_models = registry.list_models(stage='production')

        for model_info in production_models:
            if model_info['name'] == model_name:
                registry.update_stage(
                    model_info['name'],
                    model_info['version'],
                    'archived'
                )

        # Promote new
        registry.update_stage(model_name, version, 'production')

    logger.info(f"Retraining zakończony: {version}")

    return version