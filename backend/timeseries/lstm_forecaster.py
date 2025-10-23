"""
LSTM Forecaster - Long Short-Term Memory dla time series.

Funkcjonalności:
- LSTM neural network
- Multivariate forecasting
- Sequence-to-sequence
- Attention mechanism (opcjonalne)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow nie jest zainstalowane")

from sklearn.preprocessing import MinMaxScaler

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    LSTM-based time series forecaster.
    
    Architektura:
    Input(sequence) -> LSTM layers -> Dense -> Output(forecast)
    """

    def __init__(
        self,
        lookback: int = 30,
        forecast_horizon: int = 7,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        validation_split: float = 0.2
    ):
        """
        Inicjalizacja LSTM forecaster.

        Args:
            lookback: Liczba timesteps do lookback
            forecast_horizon: Liczba kroków do predykcji
            lstm_units: Lista rozmiarów LSTM layers [50, 50]
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Liczba epok
            validation_split: Proporcja walidacji
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow nie jest zainstalowany")

        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units or [50, 50]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

        logger.info("LSTM Forecaster zainicjalizowany")

    def _create_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tworzy sekwencje dla LSTM.

        Args:
            data: Dane (scaled)

        Returns:
            Tuple: (X, y) sequences
        """
        X, y = [], []

        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon])

        return np.array(X), np.array(y)

    def _build_model(self, n_features: int) -> keras.Model:
        """
        Buduje model LSTM.

        Args:
            n_features: Liczba features

        Returns:
            keras.Model: Model
        """
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(self.lookback, n_features)))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)

            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            ))

        # Output layer
        model.add(layers.Dense(self.forecast_horizon * n_features))
        model.add(layers.Reshape((self.forecast_horizon, n_features)))

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    @handle_errors(show_in_ui=False)
    def fit(
        self,
        series: pd.DataFrame,
        verbose: int = 1
    ) -> 'LSTMForecaster':
        """
        Trenuje model LSTM.

        Args:
            series: DataFrame z szeregami czasowymi
            verbose: Verbose level

        Returns:
            self
        """
        logger.info("Rozpoczęcie treningu LSTM")

        # Scale data
        data_scaled = self.scaler.fit_transform(series.values)

        # Create sequences
        X, y = self._create_sequences(data_scaled)

        logger.info(f"Utworzono {len(X)} sekwencji (lookback={self.lookback}, horizon={self.forecast_horizon})")

        # Build model
        n_features = series.shape[1]
        self.model = self._build_model(n_features)

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Train
        self.history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        logger.info("LSTM wytrenowany")

        return self

    def predict(
        self,
        last_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Generuje predykcje.

        Args:
            last_sequence: Ostatnia sekwencja (lookback x n_features)

        Returns:
            np.ndarray: Predykcje (forecast_horizon x n_features)
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        # Scale
        last_sequence_scaled = self.scaler.transform(last_sequence)

        # Reshape for LSTM
        X = last_sequence_scaled.reshape(1, self.lookback, -1)

        # Predict
        y_pred_scaled = self.model.predict(X, verbose=0)

        # Inverse scale
        y_pred = self.scaler.inverse_transform(
            y_pred_scaled.reshape(self.forecast_horizon, -1)
        )

        return y_pred

    def forecast_iterative(
        self,
        initial_sequence: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Generuje predykcje iteracyjnie (multi-step).

        Args:
            initial_sequence: Początkowa sekwencja
            n_steps: Liczba kroków

        Returns:
            np.ndarray: Predykcje
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        predictions = []
        current_sequence = initial_sequence.copy()

        steps_done = 0

        while steps_done < n_steps:
            # Predict next forecast_horizon steps
            forecast = self.predict(current_sequence[-self.lookback:])

            # How many steps to take from this forecast
            steps_to_take = min(self.forecast_horizon, n_steps - steps_done)

            predictions.append(forecast[:steps_to_take])

            # Update sequence
            current_sequence = np.vstack([current_sequence, forecast[:steps_to_take]])

            steps_done += steps_to_take

        return np.vstack(predictions)

    def plot_training_history(self):
        """Generuje wykres historii treningu."""
        if self.history is None:
            raise ValueError("Model nie został wytrenowany")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def save_model(self, path: str):
        """Zapisuje model."""
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        self.model.save(path)

        import joblib
        joblib.dump(self.scaler, path.replace('.h5', '_scaler.pkl'))

        logger.info(f"Model zapisany: {path}")

    def load_model(self, path: str):
        """Ładuje model."""
        self.model = keras.models.load_model(path)

        import joblib
        self.scaler = joblib.load(path.replace('.h5', '_scaler.pkl'))

        logger.info(f"Model wczytany: {path}")