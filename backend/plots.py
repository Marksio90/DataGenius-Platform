"""
Moduł generowania wykresów dla wyników ML.

Funkcjonalności:
- ROC curve
- Precision-Recall curve
- Confusion Matrix
- Calibration curve
- Residual plots (regression)
- QQ plot (regression)
"""

import logging
from io import BytesIO
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)

# Styl wykresów
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


@handle_errors(show_in_ui=False)
def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> Optional[BytesIO]:
    """
    Generuje ROC curve.

    Args:
        y_true: Prawdziwe etykiety
        y_proba: Prawdopodobieństwa klasy pozytywnej
        model_name: Nazwa modelu

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        >>> img = plot_roc_curve(y_true, y_proba)
        >>> img is not None
        True
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania ROC curve: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> Optional[BytesIO]:
    """
    Generuje Precision-Recall curve.

    Args:
        y_true: Prawdziwe etykiety
        y_proba: Prawdopodobieństwa klasy pozytywnej
        model_name: Nazwa modelu

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        >>> img = plot_precision_recall_curve(y_true, y_proba)
        >>> img is not None
        True
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania PR curve: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    model_name: str = "Model"
) -> Optional[BytesIO]:
    """
    Generuje Confusion Matrix.

    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        class_names: Nazwy klas
        model_name: Nazwa modelu

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> img = plot_confusion_matrix(y_true, y_pred)
        >>> img is not None
        True
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names or 'auto',
            yticklabels=class_names or 'auto',
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania confusion matrix: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10
) -> Optional[BytesIO]:
    """
    Generuje Calibration Curve.

    Args:
        y_true: Prawdziwe etykiety
        y_proba: Prawdopodobieństwa
        model_name: Nazwa modelu
        n_bins: Liczba binów

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        >>> img = plot_calibration_curve(y_true, y_proba)
        >>> img is not None
        True
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania calibration curve: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Optional[BytesIO]:
    """
    Generuje wykres residuals (dla regresji).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        model_name: Nazwa modelu

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2])
        >>> img = plot_residuals(y_true, y_pred)
        >>> img is not None
        True
    """
    try:
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predicted - {model_name}')
        axes[0].grid(True, alpha=0.3)

        # Histogram residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution - {model_name}')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania residuals plot: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Optional[BytesIO]:
    """
    Generuje wykres Predicted vs Actual (dla regresji).

    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        model_name: Nazwa modelu

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2])
        >>> img = plot_predictions_vs_actual(y_true, y_pred)
        >>> img is not None
        True
    """
    try:
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.6)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs Actual - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania pred vs actual plot: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance"
) -> Optional[BytesIO]:
    """
    Generuje wykres feature importance.

    Args:
        feature_names: Lista nazw features
        importances: Wartości importance
        top_n: Liczba features do wyświetlenia
        title: Tytuł wykresu

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> features = ['f1', 'f2', 'f3']
        >>> importances = np.array([0.5, 0.3, 0.2])
        >>> img = plot_feature_importance(features, importances)
        >>> img is not None
        True
    """
    try:
        # Sortuj i weź top N
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)))
        ax.barh(range(len(top_features)), top_importances)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania feature importance plot: {e}")
        return None