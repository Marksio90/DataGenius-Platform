"""
Główna integracja ML - orchestrator treningu i ewaluacji.

Funkcjonalności:
- Orkiestracja pełnego pipeline ML
- Agregacja wyników
- Generowanie raportów
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.async_ml_trainer import AsyncMLTrainer
from backend.auto_prep import auto_preprocess
from backend.error_handler import MLTrainingException, handle_errors
from backend.training_plan import TrainingPlan
from backend.utils_target import detect_target_column, validate_target_column

logger = logging.getLogger(__name__)


class MLIntegration:
    """
    Główna klasa integracji ML.
    
    Zarządza pełnym pipeline:
    - Preprocessing
    - Training plan
    - Model training
    - Results aggregation
    """

    def __init__(self):
        """Inicjalizacja ML integration."""
        self.training_plan: Optional[Dict] = None
        self.trainer: Optional[AsyncMLTrainer] = None
        self.results: List[Dict] = []
        self.preprocessing_metadata: Optional[Dict] = None

    @handle_errors(show_in_ui=False)
    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        strategy: str = "balanced",
        test_size: float = 0.2,
        use_tuning: Optional[bool] = None,
        use_ensemble: Optional[bool] = None
    ) -> Dict:
        """
        Uruchamia pełny pipeline ML.

        Args:
            df: DataFrame z danymi
            target_col: Nazwa kolumny target (None = auto-detect)
            strategy: Strategia treningu
            test_size: Rozmiar zbioru testowego
            use_tuning: Czy użyć tuningu
            use_ensemble: Czy użyć ensemble

        Returns:
            Dict: Kompletne wyniki pipeline

        Raises:
            MLTrainingException: Gdy pipeline nie powiedzie się

        Example:
            >>> df = pd.DataFrame({
            ...     'feature1': range(100),
            ...     'feature2': range(100, 200),
            ...     'target': [0, 1] * 50
            ... })
            >>> ml = MLIntegration()
            >>> results = ml.run_full_pipeline(df, 'target', strategy='fast_small')
            >>> 'models' in results
            True
        """
        logger.info("=== Rozpoczęcie Full ML Pipeline ===")

        # 1. Wykryj target (jeśli nie podany)
        if target_col is None:
            target_col = detect_target_column(df)
            if target_col is None:
                raise MLTrainingException("Nie można automatycznie wykryć kolumny target")
            logger.info(f"Auto-wykryty target: {target_col}")

        # 2. Waliduj target
        is_valid, warnings = validate_target_column(df, target_col)
        if not is_valid:
            raise MLTrainingException(f"Niepoprawny target: {'; '.join(warnings)}")

        # 3. Preprocessing
        logger.info("Krok 1/3: Preprocessing...")
        X_train, X_test, y_train, y_test, preprocessor, prep_meta = auto_preprocess(
            df, target_col, test_size=test_size
        )
        self.preprocessing_metadata = prep_meta

        # 4. Training plan
        logger.info("Krok 2/3: Tworzenie planu treningu...")
        plan_creator = TrainingPlan(
            problem_type=prep_meta['problem_type'],
            n_samples=prep_meta['n_train'],
            n_features=prep_meta['n_features'],
            strategy=strategy
        )
        self.training_plan = plan_creator.create_plan(
            use_tuning=use_tuning,
            use_ensemble=use_ensemble
        )

        # 5. Trening modeli
        logger.info("Krok 3/3: Trening modeli...")
        self.trainer = AsyncMLTrainer(
            problem_type=prep_meta['problem_type'],
            training_plan=self.training_plan
        )

        self.results = self.trainer.train_all_models(
            X_train, y_train, X_test, y_test
        )

        # 6. Agreguj wyniki
        logger.info("Agregacja wyników...")
        pipeline_results = self._aggregate_results(X_test, y_test)

        logger.info("=== Pipeline zakończony pomyślnie ===")

        return pipeline_results

    def _aggregate_results(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Agreguje wyniki wszystkich modeli.

        Args:
            X_test: Features testowe
            y_test: Target testowy

        Returns:
            Dict: Zagregowane wyniki
        """
        best_name, best_model = self.trainer.get_best_model()

        aggregated = {
            'problem_type': self.preprocessing_metadata['problem_type'],
            'n_models_trained': len(self.results),
            'n_features': self.preprocessing_metadata['n_features'],
            'feature_names': self.preprocessing_metadata['feature_names'],
            'class_names': self.preprocessing_metadata.get('class_names'),
            'training_plan': self.training_plan,
            'models': self.results,
            'best_model_name': best_name,
            'best_model': best_model,
            'X_test': X_test,
            'y_test': y_test,
        }

        # Dodaj ranking modeli
        aggregated['model_ranking'] = self._create_model_ranking()

        return aggregated

    def _create_model_ranking(self) -> List[Dict]:
        """
        Tworzy ranking modeli według metryki głównej.

        Returns:
            List[Dict]: Lista modeli posortowana od najlepszego
        """
        primary_metric = self.training_plan.get('metrics', {}).get('primary', 'accuracy')

        ranking = []
        for result in self.results:
            score = result.get('test_metrics', {}).get(primary_metric)
            if score is None:
                score = result.get('cv_scores', {}).get(primary_metric, {}).get('mean')

            if score is not None:
                # Obsługa metryk negatywnych
                if primary_metric.startswith('neg_'):
                    score = -score

                ranking.append({
                    'model_name': result['model_name'],
                    'score': score,
                    'training_time': result['training_time'],
                })

        # Sortuj od najlepszego
        ranking.sort(key=lambda x: x['score'], reverse=True)

        return ranking

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Zwraca wyniki jako DataFrame.

        Returns:
            pd.DataFrame: DataFrame z wynikami

        Example:
            >>> ml = MLIntegration()
            >>> # Po uruchomieniu pipeline...
            >>> # df_results = ml.get_results_dataframe()
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        primary_metric = self.training_plan.get('metrics', {}).get('primary', 'accuracy')

        for result in self.results:
            row = {
                'Model': result['model_name'],
                'Training Time (s)': result['training_time'],
            }

            # Dodaj metryki testowe
            for metric, value in result.get('test_metrics', {}).items():
                row[f'Test {metric}'] = value

            # Dodaj CV scores
            for metric, scores in result.get('cv_scores', {}).items():
                row[f'CV {metric} (mean)'] = scores.get('mean')
                row[f'CV {metric} (std)'] = scores.get('std')

            rows.append(row)

        df_results = pd.DataFrame(rows)

        # Sortuj według metryki głównej
        test_col = f'Test {primary_metric}'
        if test_col in df_results.columns:
            df_results = df_results.sort_values(test_col, ascending=False)

        return df_results