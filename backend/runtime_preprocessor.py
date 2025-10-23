"""
Preprocessing danych w runtime (przed treningiem).

Funkcjonalności:
- Podział train/test
- Imputacja brakujących wartości
- Encoding kategorii
- Feature scaling
- Kompatybilność z sklearn ≥1.5
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from backend.error_handler import DataValidationException, handle_errors
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RuntimePreprocessor:
    """
    Preprocessor danych dla ML pipeline.
    
    Obsługuje:
    - Split train/test
    - Imputację
    - Encoding
    - Scaling
    """

    def __init__(self, problem_type: str):
        """
        Inicjalizacja preprocessora.

        Args:
            problem_type: Typ problemu ('binary_classification', 'multiclass_classification', 'regression')
        """
        self.problem_type = problem_type
        self.label_encoder: Optional[LabelEncoder] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []

    @handle_errors(show_in_ui=False)
    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Dzieli dane na train/test.

        Args:
            df: DataFrame
            target_col: Nazwa kolumny target
            test_size: Rozmiar zbioru testowego
            stratify: Czy stratyfikować (dla klasyfikacji)

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)

        Example:
            >>> df = pd.DataFrame({'a': range(100), 'b': range(100), 'target': [0, 1] * 50})
            >>> prep = RuntimePreprocessor('binary_classification')
            >>> X_train, X_test, y_train, y_test = prep.split_data(df, 'target')
            >>> len(X_train) + len(X_test) == 100
            True
        """
        if target_col not in df.columns:
            raise DataValidationException(f"Kolumna target '{target_col}' nie istnieje")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Stratyfikacja tylko dla klasyfikacji
        stratify_param = y if (stratify and 'classification' in self.problem_type) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=settings.random_state,
            stratify=stratify_param
        )

        logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    @handle_errors(show_in_ui=False)
    def build_preprocessor(
        self,
        X: pd.DataFrame,
        numeric_strategy: str = 'mean',
        categorical_strategy: str = 'most_frequent'
    ) -> ColumnTransformer:
        """
        Buduje preprocessor dla features.

        Args:
            X: DataFrame z features
            numeric_strategy: Strategia imputacji dla numerycznych ('mean', 'median', 'constant')
            categorical_strategy: Strategia imputacji dla kategorycznych ('most_frequent', 'constant')

        Returns:
            ColumnTransformer: Zbudowany preprocessor

        Example:
            >>> df = pd.DataFrame({'num': [1, 2, np.nan], 'cat': ['a', 'b', 'a']})
            >>> prep = RuntimePreprocessor('binary_classification')
            >>> preprocessor = prep.build_preprocessor(df)
            >>> preprocessor is not None
            True
        """
        # Identyfikuj typy kolumn
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.info(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

        # Pipeline dla numerycznych
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', StandardScaler())
        ])

        # Pipeline dla kategorycznych - WAŻNE: sparse_output zamiast sparse
        # Kompatybilność z sklearn >= 1.5
        try:
            # Sprawdź wersję sklearn
            import sklearn
            sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
            
            if sklearn_version >= (1, 5):
                # Nowa wersja: używaj sparse_output
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            else:
                # Starsza wersja: używaj sparse (deprecated)
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
                ('onehot', ohe)
            ])
        except Exception as e:
            logger.warning(f"Błąd konfiguracji OneHotEncoder: {e}. Użycie domyślnej konfiguracji.")
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

        # Kombinuj transformery
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))

        if not transformers:
            raise DataValidationException("Brak features do przetworzenia")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        self.preprocessor = preprocessor
        logger.info("Preprocessor zbudowany pomyślnie")
        
        return preprocessor

    @handle_errors(show_in_ui=False)
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fituje preprocessor na train i transformuje train/test.

        Args:
            X_train: DataFrame treningowy
            X_test: DataFrame testowy

        Returns:
            Tuple: (X_train_transformed, X_test_transformed, feature_names)

        Example:
            >>> X_train = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
            >>> X_test = pd.DataFrame({'a': [4], 'b': ['x']})
            >>> prep = RuntimePreprocessor('regression')
            >>> prep.build_preprocessor(X_train)
            >>> X_tr, X_te, names = prep.fit_transform(X_train, X_test)
            >>> len(names) > 0
            True
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor nie został zbudowany. Wywołaj build_preprocessor() najpierw.")

        # Fit i transform
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        # Pobierz nazwy features po transformacji
        feature_names = self._get_feature_names()
        self.feature_names = feature_names

        logger.info(f"Preprocessing zakończony: {len(feature_names)} features po transformacji")

        return X_train_transformed, X_test_transformed, feature_names

    def _get_feature_names(self) -> List[str]:
        """
        Pobiera nazwy features po transformacji.

        Returns:
            List[str]: Lista nazw features
        """
        if self.preprocessor is None:
            return []

        feature_names = []

        try:
            # Dla każdego transformera w ColumnTransformer
            for name, transformer, columns in self.preprocessor.transformers_:
                if name == 'remainder':
                    continue

                if hasattr(transformer, 'get_feature_names_out'):
                    # Sklearn >= 1.0
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                elif name == 'num':
                    # Numeryczne - bez zmian nazw
                    feature_names.extend(columns)
                elif name == 'cat':
                    # Kategoryczne - onehot generuje wiele kolumn
                    try:
                        ohe = transformer.named_steps['onehot']
                        ohe_names = ohe.get_feature_names_out(columns)
                        feature_names.extend(ohe_names)
                    except:
                        # Fallback
                        feature_names.extend([f"{col}_enc" for col in columns])
                else:
                    feature_names.extend(columns)

        except Exception as e:
            logger.warning(f"Nie udało się pobrać nazw features: {e}")
            # Fallback: użyj indeksów
            n_features = self.preprocessor.transform(pd.DataFrame(columns=self.preprocessor.feature_names_in_)).shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]

        return feature_names

    @handle_errors(show_in_ui=False)
    def encode_target(
        self,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enkoduje target dla klasyfikacji.

        Args:
            y_train: Seria target treningowa
            y_test: Seria target testowa

        Returns:
            Tuple: (y_train_encoded, y_test_encoded)

        Example:
            >>> y_train = pd.Series(['cat', 'dog', 'cat'])
            >>> y_test = pd.Series(['dog'])
            >>> prep = RuntimePreprocessor('binary_classification')
            >>> y_tr, y_te = prep.encode_target(y_train, y_test)
            >>> len(y_tr) == 3
            True
        """
        if 'classification' not in self.problem_type:
            # Dla regresji: bez encodingu
            return y_train.values, y_test.values

        # Label encoding dla klasyfikacji
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        logger.info(f"Target encoded: {len(self.label_encoder.classes_)} klas")

        return y_train_encoded, y_test_encoded

    def get_class_names(self) -> Optional[List[str]]:
        """
        Zwraca nazwy klas (dla klasyfikacji).

        Returns:
            Optional[List[str]]: Lista nazw klas lub None
        """
        if self.label_encoder is not None:
            return self.label_encoder.classes_.tolist()
        return None