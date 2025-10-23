"""
PyTorch Neural Network Trainer.

Funkcjonalności:
- Automatyczna architektura sieci
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Tensorboard logging
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class NeuralNetClassifier(nn.Module):
    """
    Feedforward Neural Network dla klasyfikacji.
    
    Architektura:
    Input -> Hidden Layers (ReLU + Dropout) -> Output (Softmax)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout_rate: float = 0.3
    ):
        super(NeuralNetClassifier, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetRegressor(nn.Module):
    """
    Feedforward Neural Network dla regresji.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout_rate: float = 0.3
    ):
        super(NeuralNetRegressor, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class PyTorchTrainer:
    """
    Trainer dla PyTorch neural networks.
    
    Wspiera:
    - Klasyfikację (binary, multiclass)
    - Regresję
    - Automatyczny dobór architektury
    - Early stopping
    - Learning rate scheduling
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
        device: Optional[str] = None,
        tensorboard_log_dir: Optional[Path] = None
    ):
        """
        Inicjalizacja trainera.

        Args:
            problem_type: Typ problemu ('binary_classification', 'multiclass_classification', 'regression')
            hidden_sizes: Lista rozmiarów warstw ukrytych (None = auto)
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Maksymalna liczba epok
            early_stopping_patience: Patience dla early stopping
            dropout_rate: Dropout rate
            device: Device ('cuda' lub 'cpu', None = auto)
            tensorboard_log_dir: Katalog logów TensorBoard
        """
        self.problem_type = problem_type
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"PyTorch Trainer inicjalizowany na device: {self.device}")

        # Model i komponenty
        self.model = None
        self.scaler = StandardScaler()
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # TensorBoard
        self.writer = None
        if tensorboard_log_dir:
            self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        # Historia treningu
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }

    def _auto_architecture(self, input_size: int, output_size: int) -> List[int]:
        """
        Automatyczny dobór architektury sieci.

        Args:
            input_size: Liczba features
            output_size: Liczba klas/outputs

        Returns:
            List[int]: Lista rozmiarów warstw ukrytych
        """
        # Heurystyka: geometryczne malejąca sekwencja
        # Zaczynamy od 2x input_size, kończymy na 2x output_size

        start_size = min(input_size * 2, 512)
        end_size = max(output_size * 2, 32)

        if start_size <= end_size:
            return [start_size]

        # 3 warstwy: start -> middle -> end
        middle_size = int(np.sqrt(start_size * end_size))

        return [start_size, middle_size, end_size]

    @handle_errors(show_in_ui=False)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Trenuje model.

        Args:
            X: Features (numpy array)
            y: Target (numpy array)
            X_val: Validation features (opcjonalne)
            y_val: Validation target (opcjonalne)
            verbose: Czy wyświetlać progress

        Returns:
            Dict: Historia treningu
        """
        logger.info(f"Rozpoczęcie treningu PyTorch NN dla {self.problem_type}")

        # Split validation jeśli nie podano
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        X_val_scaled = self.scaler.transform(X_val)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Determine architecture
        input_size = X.shape[1]

        if 'classification' in self.problem_type:
            num_classes = len(np.unique(y))
            y_train_tensor = y_train_tensor.long()
            y_val_tensor = y_val_tensor.long()

            if self.hidden_sizes is None:
                self.hidden_sizes = self._auto_architecture(input_size, num_classes)

            self.model = NeuralNetClassifier(
                input_size,
                self.hidden_sizes,
                num_classes,
                self.dropout_rate
            ).to(self.device)

            self.criterion = nn.CrossEntropyLoss()

        else:  # regression
            y_train_tensor = y_train_tensor.view(-1, 1)
            y_val_tensor = y_val_tensor.view(-1, 1)

            if self.hidden_sizes is None:
                self.hidden_sizes = self._auto_architecture(input_size, 1)

            self.model = NeuralNetRegressor(
                input_size,
                self.hidden_sizes,
                self.dropout_rate
            ).to(self.device)

            self.criterion = nn.MSELoss()

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=verbose
        )

        # DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        # Training loop
        for epoch in range(self.max_epochs):
            # Train
            train_loss, train_metric = self._train_epoch(train_loader)

            # Validation
            val_loss, val_metric = self._validate_epoch(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Historia
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)

            # TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Metric/train', train_metric, epoch)
                self.writer.add_scalar('Metric/val', val_metric, epoch)

            # Verbose
            if verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Metric: {train_metric:.4f}, Val Metric: {val_metric:.4f}"
                )

            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if self.writer:
            self.writer.close()

        logger.info("Trening zakończony")

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Trenuje jedną epokę."""
        self.model.train()
        total_loss = 0
        total_metric = 0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()

            if 'classification' in self.problem_type:
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_batch).float().mean().item()
                total_metric += accuracy
            else:
                # R² score for regression
                ss_res = ((y_batch - outputs) ** 2).sum().item()
                ss_tot = ((y_batch - y_batch.mean()) ** 2).sum().item()
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                total_metric += r2

            n_batches += 1

        return total_loss / n_batches, total_metric / n_batches

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Waliduje jedną epokę."""
        self.model.eval()
        total_loss = 0
        total_metric = 0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()

                if 'classification' in self.problem_type:
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = (predicted == y_batch).float().mean().item()
                    total_metric += accuracy
                else:
                    ss_res = ((y_batch - outputs) ** 2).sum().item()
                    ss_tot = ((y_batch - y_batch.mean()) ** 2).sum().item()
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    total_metric += r2

                n_batches += 1

        return total_loss / n_batches, total_metric / n_batches

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja.

        Args:
            X: Features

        Returns:
            np.ndarray: Predykcje
        """
        self.model.eval()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

            if 'classification' in self.problem_type:
                _, predicted = torch.max(outputs.data, 1)
                return predicted.cpu().numpy()
            else:
                return outputs.cpu().numpy().flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja prawdopodobieństw (tylko klasyfikacja).

        Args:
            X: Features

        Returns:
            np.ndarray: Prawdopodobieństwa
        """
        if 'classification' not in self.problem_type:
            raise ValueError("predict_proba działa tylko dla klasyfikacji")

        self.model.eval()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)
            return proba.cpu().numpy()

    def save_model(self, path: Path) -> None:
        """Zapisuje model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'hidden_sizes': self.hidden_sizes,
            'problem_type': self.problem_type,
            'device': str(self.device),
        }, path)

        logger.info(f"Model zapisany: {path}")

    def load_model(self, path: Path) -> None:
        """Ładuje model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.hidden_sizes = checkpoint['hidden_sizes']
        self.problem_type = checkpoint['problem_type']
        self.scaler = checkpoint['scaler']

        # Odtwórz architekturę
        # TODO: zapisz i wczytaj pełną architekturę

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model wczytany: {path}")


# Sklearn-compatible wrapper
class PyTorchNeuralNetwork:
    """
    Sklearn-compatible wrapper dla PyTorch NN.
    
    Może być używany w pipeline ML jak standardowy model sklearn.
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

        # Trainer
        self.trainer = None

    def fit(self, X, y):
        """Trenuje model."""
        # Ustaw seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.trainer = PyTorchTrainer(
            problem_type=self.problem_type,
            hidden_sizes=self.hidden_sizes,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            dropout_rate=self.dropout_rate
        )

        self.trainer.fit(X, y, verbose=False)

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