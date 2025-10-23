"""
Moduł tworzenia planu trenowania modeli ML.

Funkcjonalności:
- Dobór strategii (fast/balanced/accurate/advanced)
- Wybór metryk
- Konfiguracja CV
- Selekcja modeli
- Feature flags (tuning, ensemble, PyCaret)
"""

import logging
from typing import Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TrainingPlan:
    """
    Plan trenowania modeli ML.
    
    Określa:
    - Modele do trenowania
    - Strategię CV
    - Metryki
    - Hyperparametry
    """

    def __init__(
        self,
        problem_type: str,
        n_samples: int,
        n_features: int,
        strategy: str = "balanced"
    ):
        """
        Inicjalizacja planu trenowania.

        Args:
            problem_type: Typ problemu ML
            n_samples: Liczba próbek
            n_features: Liczba features
            strategy: Strategia ('fast_small', 'balanced', 'accurate', 'advanced')
        """
        self.problem_type = problem_type
        self.n_samples = n_samples
        self.n_features = n_features
        self.strategy = strategy
        self.plan: Dict = {}

    def create_plan(
        self,
        use_tuning: Optional[bool] = None,
        use_ensemble: Optional[bool] = None,
        use_pycaret: Optional[bool] = None
    ) -> Dict:
        """
        Tworzy plan trenowania.

        Args:
            use_tuning: Czy użyć tuningu (None = auto)
            use_ensemble: Czy użyć ensemble (None = auto)
            use_pycaret: Czy użyć PyCaret (None = settings)

        Returns:
            Dict: Słownik z planem trenowania

        Example:
            >>> plan_obj = TrainingPlan('binary_classification', 1000, 10, 'balanced')
            >>> plan = plan_obj.create_plan()
            >>> 'models' in plan
            True
        """
        # Auto-determine flags
        if use_tuning is None:
            use_tuning = self._should_use_tuning()
        
        if use_ensemble is None:
            use_ensemble = self._should_use_ensemble()
        
        if use_pycaret is None:
            use_pycaret = settings.enable_pycaret

        # Określ modele
        models = self._select_models()

        # Określ CV strategy
        cv_strategy = self._select_cv_strategy()

        # Określ metryki
        metrics = self._select_metrics()

        self.plan = {
            "problem_type": self.problem_type,
            "strategy": self.strategy,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "models": models,
            "cv_strategy": cv_strategy,
            "metrics": metrics,
            "use_tuning": use_tuning,
            "use_ensemble": use_ensemble,
            "use_pycaret": use_pycaret,
            "n_jobs": settings.n_jobs,
            "random_state": settings.random_state,
        }

        logger.info(f"Plan trenowania utworzony: strategia={self.strategy}, "
                    f"modele={len(models)}, tuning={use_tuning}, ensemble={use_ensemble}")

        return self.plan

    def _should_use_tuning(self) -> bool:
        """Określa czy użyć tuningu na podstawie rozmiaru danych."""
        if not settings.enable_auto_tuning:
            return False

        # Tuning dla większych datasetsów
        if self.n_samples >= 1000 and self.n_features <= 100:
            return True

        # Nie dla bardzo dużych lub małych
        if self.n_samples < 500 or self.n_samples > 50000:
            return False

        return self.strategy in ['accurate', 'advanced']

    def _should_use_ensemble(self) -> bool:
        """Określa czy użyć ensemble."""
        if not settings.enable_ensemble:
            return False

        # Ensemble dla odpowiednich rozmiarów
        if self.n_samples >= 500 and self.n_samples <= 50000:
            return True

        return self.strategy == 'advanced'

    def _select_models(self) -> List[str]:
        """
        Wybiera modele do trenowania.

        Returns:
            List[str]: Lista nazw modeli
        """
        if self.strategy == "fast_small":
            # Szybkie modele dla małych datasetsów
            if 'classification' in self.problem_type:
                return ['LogisticRegression', 'DecisionTree', 'RandomForest']
            else:
                return ['LinearRegression', 'DecisionTree', 'RandomForest']

        elif self.strategy == "balanced":
            # Zbalansowany zestaw
            if 'classification' in self.problem_type:
                return [
                    'LogisticRegression',
                    'RandomForest',
                    'GradientBoosting',
                    'XGBoost',
                    'LightGBM'
                ]
            else:
                return [
                    'LinearRegression',
                    'Ridge',
                    'RandomForest',
                    'GradientBoosting',
                    'XGBoost',
                    'LightGBM'
                ]

        elif self.strategy == "accurate":
            # Więcej modeli, lepsze wyniki
            if 'classification' in self.problem_type:
                return [
                    'LogisticRegression',
                    'RandomForest',
                    'GradientBoosting',
                    'XGBoost',
                    'LightGBM',
                    'CatBoost',
                    'SVC'
                ]
            else:
                return [
                    'LinearRegression',
                    'Ridge',
                    'Lasso',
                    'ElasticNet',
                    'RandomForest',
                    'GradientBoosting',
                    'XGBoost',
                    'LightGBM',
                    'CatBoost'
                ]

        elif self.strategy == "advanced":
            # Wszystkie dostępne modele + tuning
            if 'classification' in self.problem_type:
                return [
                    'LogisticRegression',
                    'KNN',
                    'DecisionTree',
                    'RandomForest',
                    'ExtraTrees',
                    'GradientBoosting',
                    'XGBoost',
                    'LightGBM',
                    'CatBoost',
                    'SVC'
                ]
            else:
                return [
                    'LinearRegression',
                    'Ridge',
                    'Lasso',
                    'ElasticNet',
                    'KNN',
                    'DecisionTree',
                    'RandomForest',
                    'ExtraTrees',
                    'GradientBoosting',
                    'XGBoost',
                    'LightGBM',
                    'CatBoost',
                    'SVR'
                ]

        # Default: balanced
        return self._select_models_for_strategy("balanced")

    def _select_models_for_strategy(self, strategy: str) -> List[str]:
        """Helper do wyboru modeli dla strategii."""
        temp_plan = TrainingPlan(self.problem_type, self.n_samples, self.n_features, strategy)
        return temp_plan._select_models()

    def _select_cv_strategy(self) -> Dict:
        """
        Wybiera strategię cross-validation.

        Returns:
            Dict: Konfiguracja CV
        """
        # Domyślnie: StratifiedKFold dla klasyfikacji, KFold dla regresji
        if 'classification' in self.problem_type:
            cv_type = 'StratifiedKFold'
        else:
            cv_type = 'KFold'

        # Liczba foldów zależna od rozmiaru danych
        if self.n_samples < 500:
            n_splits = 3
        elif self.n_samples < 2000:
            n_splits = 5
        else:
            n_splits = 5  # Standard

        return {
            "type": cv_type,
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": settings.random_state,
        }

    def _select_metrics(self) -> Dict:
        """
        Wybiera metryki do ewaluacji.

        Returns:
            Dict: Słownik z metrykami
        """
        if self.problem_type == 'binary_classification':
            return {
                "primary": "roc_auc",
                "secondary": ["accuracy", "f1", "precision", "recall", "average_precision"],
                "names": {
                    "roc_auc": "ROC AUC",
                    "accuracy": "Accuracy",
                    "f1": "F1 Score",
                    "precision": "Precision",
                    "recall": "Recall",
                    "average_precision": "Avg Precision",
                }
            }

        elif self.problem_type == 'multiclass_classification':
            return {
                "primary": "f1_weighted",
                "secondary": ["accuracy", "f1_macro", "f1_micro"],
                "names": {
                    "f1_weighted": "F1 Weighted",
                    "accuracy": "Accuracy",
                    "f1_macro": "F1 Macro",
                    "f1_micro": "F1 Micro",
                }
            }

        elif self.problem_type == 'regression':
            return {
                "primary": "neg_root_mean_squared_error",
                "secondary": ["neg_mean_absolute_error", "r2"],
                "names": {
                    "neg_root_mean_squared_error": "RMSE",
                    "neg_mean_absolute_error": "MAE",
                    "r2": "R²",
                }
            }

        # Default
        return {
            "primary": "accuracy",
            "secondary": [],
            "names": {"accuracy": "Accuracy"}
        }

    def get_summary(self) -> str:
        """
        Zwraca tekstowe podsumowanie planu.

        Returns:
            str: Podsumowanie planu

        Example:
            >>> plan_obj = TrainingPlan('regression', 1000, 10, 'balanced')
            >>> plan_obj.create_plan()
            >>> summary = plan_obj.get_summary()
            >>> 'Strategia' in summary
            True
        """
        if not self.plan:
            return "Plan nie został jeszcze utworzony"

        summary = f"""
📋 **Plan Trenowania ML**

**Problem:** {self.problem_type}
**Strategia:** {self.strategy}
**Dataset:** {self.n_samples} próbek, {self.n_features} features

**Modele:** {len(self.plan['models'])}
{', '.join(self.plan['models'])}

**Cross-Validation:** {self.plan['cv_strategy']['type']} ({self.plan['cv_strategy']['n_splits']} splits)

**Główna metryka:** {self.plan['metrics']['names'][self.plan['metrics']['primary']]}

**Opcje:**
- Tuning: {'✅' if self.plan['use_tuning'] else '❌'}
- Ensemble: {'✅' if self.plan['use_ensemble'] else '❌'}
- PyCaret: {'✅' if self.plan['use_pycaret'] else '❌'}
"""
        return summary.strip()