"""
TensorFlow/Keras Neural Network Trainer.

Funkcjonalności:
- Automatyczna architektura
- Callbacks (EarlyStopping, ReduceLROnPlateau)
- Model checkpointing
- TensorBoard logging
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow nie jest zainstalowane - funkcjonalność niedostępna")

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class TensorFlowTrainer:
    """
    Trainer dla TensorFlow/Keras neural networks.
    
    Wspiera:
    - Klasyfikację (binary, multiclass)
    - Regresję
    - Automatyczny dobór architektury
    - Callbacks
    """

    def __init__(
        self,
        problem_type: str,
        hidden_sizes: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        dropout_rate: float = 0.3,
        tensorboard_log_dir: Optional[Path] = None
    ):
        """
        Inicjalizacja trainera.

        Args:
            problem_type: Typ problemu
            hidden_sizes: Lista rozmiarów warstw ukrytych
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Maksymalna liczba epok
            early_stopping_patience: Patience dla early stopping
            dropout_rate: Dropout rate
            tensorboard_log_dir: Katalog logów TensorBoard
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow nie jest zainstalowane")

        self.problem_type = problem_type
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.tensorboard_log_dir = tensorboard_log_dir

        # Model i komponenty
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

        logger.info("TensorFlow Trainer zainicjalizowany")

    def _auto_architecture(self, input_size: int, output_size: int) -> List[int]:
        """Automatyczny dobór architektury."""
        start_size = min(input_size * 2, 512)
        end_size = max(output_size * 2, 32)

        if start_size <= end_size:
            return [start_size]

        middle_size = int(np.sqrt(start_size * end_size))
        return [start_size, middle_size, end_size]

    def _build_model(self, input_size: int, output_size: int) -> keras.Model:
        """
        Buduje model Keras.

        Args:
            input_size: Liczba features
            output_size: Liczba outputs/klas

        Returns:
            keras.Model: Model
        """
        # Determine architecture
        if self.hidden_sizes is None:
            self.hidden_sizes = self._auto_architecture(input_size, output_size)

        # Build model
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_size,)))

        # Hidden layers
        for hidden_size in self.hidden_sizes:
            model.add(layers.Dense(hidden_size, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))

        # Output layer
        if 'classification' in self.problem_type:
            if output_size == 2:
                # Binary classification
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy', keras.metrics.AUC(name='auc')]
            else:
                # Multiclass classification
                model.add(layers.Dense(output_size, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            # Regression
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae', keras.metrics.RootMeanSquaredError(name='rmse')]

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    @handle_errors(show_in_ui=False)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Trenuje model.

        Args:
            X: Features
            y: Target
            X_val: Validation features (opcjonalne)
            y_val: Validation target (opcjonalne)
            verbose: Verbose level (0, 1, 2)

        Returns:
            Dict: Historia treningu
        """
        logger.info(f"Rozpoczęcie treningu TensorFlow NN dla {self.problem_type}")

        # Split validation jeśli nie podano
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        X_val_scaled = self.scaler.transform(X_val)

        # Determine output size
        input_size = X.shape[1]

        if 'classification' in self.problem_type:
            output_size = len(np.unique(y))
        else:
            output_size = 1

        # Build model
        self.model = self._build_model(input_size, output_size)

        # Callbacks
        callback_list = []

        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=verbose
        )
        callback_list.append(early_stop)

        # Reduce LR on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=verbose
        )
        callback_list.append(reduce_lr)

        # TensorBoard
        if self.tensorboard_log_dir:
            tensorboard_cb = callbacks.TensorBoard(
                log_dir=str(self.tensorboard_log_dir),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callback_list.append(tensorboard_cb)

        # Train
        self.history = self.model.fit(
            X_scaled,
            y,
            validation_data=(X_val_scaled, y_val),
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            callbacks=callback_list,
            verbose=verbose
        )

        logger.info("Trening zakończony")

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predykcja."""
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)

        if 'classification' in self.problem_type:
            if predictions.shape[1] == 1:
                # Binary classification
                return (predictions > 0.5).astype(int).flatten()
            else:
                # Multiclass
                return np.argmax(predictions, axis=1)
        else:
            return predictions.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predykcja prawdopodobieństw."""
        if 'classification' not in self.problem_type:
            raise ValueError("predict_proba działa tylko dla klasyfikacji")

        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)

        if predictions.shape[1] == 1:
            # Binary classification - convert to 2-column format
            proba_class_1 = predictions.flatten()
            proba_class_0 = 1 - proba_class_1
            return np.column_stack([proba_class_0, proba_class_1])
        else:
            return predictions

    def save_model(self, path: Path) -> None:
        """Zapisuje model."""
        model_path = path.with_suffix('.h5')
        scaler_path = path.with_suffix('.pkl')

        self.model.save(str(model_path))

        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'hidden_sizes': self.hidden_sizes,
            'problem_type': self.problem_type,
        }, scaler_path)

        logger.info(f"Model zapisany: {model_path}")

    def load_model(self, path: Path) -> None:
        """Ładuje model."""
        model_path = path.with_suffix('.h5')
        scaler_path = path.with_suffix('.pkl')

        self.model = keras.models.load_model(str(model_path))

        import joblib
        data = joblib.load(scaler_path)
        self.scaler = data['scaler']
        self.hidden_sizes = data['hidden_sizes']
        self.problem_type = data['problem_type']

        logger.info(f"Model wczytany: {model_path}")


# Sklearn-compatible wrapper
class TensorFlowNeuralNetwork:
    """Sklearn-compatible wrapper dla TensorFlow NN."""

    def __init__(
        self,
        problem_type: str,
        hidden_sizes: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        dropout_rate: float = 0.3,
        random_state: int = 42
    ):
        self.problem_type = problem_type
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self.trainer = None

    def fit(self, X, y):
        """Trenuje model."""
        # Set seeds
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        self.trainer = TensorFlowTrainer(
            problem_type=self.problem_type,
            hidden_sizes=self.hidden_sizes,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            dropout_rate=self.dropout_rate
        )

        self.trainer.fit(X, y, verbose=0)

        return self

    def predict(self, X):
        """Predykcja."""
        if self.trainer is None:
            raise ValueError("Model nie został wytrenowany")

        return self.trainer.predict(X)

    def predict_proba(self, X):
        """Predykcja prawdopodobieństw."""
        if self.trainer is None:
            raise ValueError("Model nie został wytrenowany")

        return self.trainer.predict_proba(X)