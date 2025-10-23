"""
Smoke testy dla głównego pipeline ML.

Sprawdzają:
- Podstawowy przepływ danych
- Trening modeli
- Kształty predykcji
- Sumy prawdopodobieństw
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from backend.async_ml_trainer import AsyncMLTrainer
from backend.auto_prep import auto_preprocess
from backend.dtype_sanitizer import sanitize_dataframe
from backend.ml_integration import MLIntegration
from backend.training_plan import TrainingPlan


@pytest.fixture
def sample_classification_data():
    """Fixture z danymi klasyfikacji."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


@pytest.fixture
def sample_regression_data():
    """Fixture z danymi regresji."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


class TestDataSanitization:
    """Testy sanityzacji danych."""

    def test_sanitize_basic(self, sample_classification_data):
        """Test podstawowej sanityzacji."""
        df_clean, report = sanitize_dataframe(sample_classification_data)

        assert not df_clean.empty
        assert len(df_clean) == len(sample_classification_data)
        assert 'original_shape' in report
        assert 'final_shape' in report

    def test_sanitize_with_duplicates(self):
        """Test sanityzacji z duplikatami nazw kolumn."""
        df = pd.DataFrame({
            'col': [1, 2, 3],
            'col ': [4, 5, 6],  # Duplikat z spacją
            ' col': [7, 8, 9]   # Duplikat z wiodącą spacją
        })

        df_clean, report = sanitize_dataframe(df)

        # Sprawdź że nazwy są unikalne
        assert len(df_clean.columns) == len(set(df_clean.columns))
        assert df_clean.shape[1] == 3


class TestPreprocessing:
    """Testy preprocessingu."""

    def test_auto_preprocess_classification(self, sample_classification_data):
        """Test auto preprocessingu dla klasyfikacji."""
        X_train, X_test, y_train, y_test, preprocessor, metadata = auto_preprocess(
            sample_classification_data,
            'target',
            test_size=0.2
        )

        # Sprawdź kształty
        assert X_train.shape[0] == pytest.approx(160, abs=10)
        assert X_test.shape[0] == pytest.approx(40, abs=10)
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

        # Sprawdź metadata
        assert metadata['problem_type'] == 'binary_classification'
        assert metadata['n_features'] == X_train.shape[1]

    def test_auto_preprocess_regression(self, sample_regression_data):
        """Test auto preprocessingu dla regresji."""
        X_train, X_test, y_train, y_test, preprocessor, metadata = auto_preprocess(
            sample_regression_data,
            'target',
            test_size=0.2
        )

        # Sprawdź kształty
        assert X_train.shape[0] == pytest.approx(160, abs=10)
        assert X_test.shape[0] == pytest.approx(40, abs=10)

        # Sprawdź metadata
        assert metadata['problem_type'] == 'regression'


class TestTrainingPlan:
    """Testy planu trenowania."""

    def test_create_plan_balanced(self):
        """Test tworzenia planu balanced."""
        plan_creator = TrainingPlan(
            problem_type='binary_classification',
            n_samples=1000,
            n_features=10,
            strategy='balanced'
        )

        plan = plan_creator.create_plan()

        assert 'models' in plan
        assert len(plan['models']) >= 3
        assert 'cv_strategy' in plan
        assert 'metrics' in plan

    def test_create_plan_fast(self):
        """Test tworzenia planu fast."""
        plan_creator = TrainingPlan(
            problem_type='regression',
            n_samples=100,
            n_features=5,
            strategy='fast_small'
        )

        plan = plan_creator.create_plan()

        assert len(plan['models']) <= 5  # Mniej modeli dla fast


class TestMLTraining:
    """Testy trenowania modeli."""

    def test_train_single_model_classification(self, sample_classification_data):
        """Test treningu pojedynczego modelu klasyfikacji."""
        # Preprocessing
        X_train, X_test, y_train, y_test, _, metadata = auto_preprocess(
            sample_classification_data,
            'target',
            test_size=0.2
        )

        # Plan
        plan = {
            'random_state': 42,
            'n_jobs': 1,
            'metrics': {
                'primary': 'accuracy',
                'secondary': ['f1']
            }
        }

        # Trainer
        trainer = AsyncMLTrainer('binary_classification', plan)

        # Trening
        result = trainer.train_single_model(
            'LogisticRegression',
            X_train,
            y_train,
            X_test,
            y_test,
            cv_folds=3
        )

        # Sprawdź wynik
        assert 'model' in result
        assert 'model_name' in result
        assert result['model_name'] == 'LogisticRegression'
        assert 'test_metrics' in result
        assert 'training_time' in result

        # Sprawdź predykcje
        model = result['model']
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # CRITICAL: Sprawdź predict_proba dla klasyfikacji
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            assert y_proba.shape[0] == len(y_test)
            assert y_proba.shape[1] == 2  # Binary

            # CRITICAL: Sprawdź czy sumy prawdopodobieństw = 1
            row_sums = y_proba.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_train_single_model_regression(self, sample_regression_data):
        """Test treningu pojedynczego modelu regresji."""
        # Preprocessing
        X_train, X_test, y_train, y_test, _, metadata = auto_preprocess(
            sample_regression_data,
            'target',
            test_size=0.2
        )

        # Plan
        plan = {
            'random_state': 42,
            'n_jobs': 1,
            'metrics': {
                'primary': 'r2',
                'secondary': ['neg_mean_absolute_error']
            }
        }

        # Trainer
        trainer = AsyncMLTrainer('regression', plan)

        # Trening
        result = trainer.train_single_model(
            'LinearRegression',
            X_train,
            y_train,
            X_test,
            y_test,
            cv_folds=3
        )

        # Sprawdź wynik
        assert 'model' in result
        assert 'test_metrics' in result

        # Sprawdź predykcje
        model = result['model']
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Sprawdź czy metryki są liczbami
        assert isinstance(result['test_metrics'].get('r2', 0), (int, float))


class TestFullPipeline:
    """Testy pełnego pipeline."""

    def test_full_pipeline_classification(self, sample_classification_data):
        """Test pełnego pipeline dla klasyfikacji."""
        ml = MLIntegration()

        pipeline_results = ml.run_full_pipeline(
            df=sample_classification_data,
            target_col='target',
            strategy='fast_small',
            test_size=0.2
        )

        # Sprawdź wyniki
        assert 'problem_type' in pipeline_results
        assert pipeline_results['problem_type'] == 'binary_classification'
        assert 'models' in pipeline_results
        assert len(pipeline_results['models']) > 0
        assert 'best_model_name' in pipeline_results
        assert pipeline_results['best_model_name'] is not None

    def test_full_pipeline_regression(self, sample_regression_data):
        """Test pełnego pipeline dla regresji."""
        ml = MLIntegration()

        pipeline_results = ml.run_full_pipeline(
            df=sample_regression_data,
            target_col='target',
            strategy='fast_small',
            test_size=0.2
        )

        # Sprawdź wyniki
        assert pipeline_results['problem_type'] == 'regression'
        assert len(pipeline_results['models']) > 0
        assert pipeline_results['best_model_name'] is not None


class TestPredictionShapes:
    """Testy kształtów predykcji."""

    def test_binary_classification_proba_sums(self, sample_classification_data):
        """CRITICAL TEST: Sprawdź czy predict_proba sumuje się do 1."""
        ml = MLIntegration()

        pipeline_results = ml.run_full_pipeline(
            df=sample_classification_data,
            target_col='target',
            strategy='fast_small'
        )

        # Dla każdego modelu
        for result in pipeline_results['models']:
            model = result['model']

            if hasattr(model, 'predict_proba'):
                X_test = pipeline_results['X_test']
                y_proba = model.predict_proba(X_test)

                # Sprawdź kształt
                assert y_proba.shape[0] == len(X_test)
                assert y_proba.shape[1] == 2  # Binary

                # CRITICAL: Sprawdź sumy
                row_sums = y_proba.sum(axis=1)
                np.testing.assert_allclose(
                    row_sums,
                    1.0,
                    rtol=1e-5,
                    err_msg=f"Model {result['model_name']} predict_proba nie sumuje się do 1"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])