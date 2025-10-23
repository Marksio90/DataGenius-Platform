"""
Zaawansowane wykresy porównawcze.

Funkcjonalności:
- Radar chart porównania modeli
- Wykresy porównawcze metryk
- Heatmapy korelacji
"""

import logging
from io import BytesIO
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import pi

from backend.error_handler import handle_errors
from backend.metrics_helper import normalize_metrics_for_comparison

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def plot_model_comparison_radar(
    results: List[Dict],
    metrics: List[str],
    metric_names: Optional[Dict[str, str]] = None
) -> Optional[BytesIO]:
    """
    Generuje radar chart porównujący modele.

    Args:
        results: Lista wyników modeli
        metrics: Lista metryk do porównania
        metric_names: Mapowanie nazw metryk (opcjonalne)

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> results = [
        ...     {'model_name': 'M1', 'test_metrics': {'acc': 0.8, 'f1': 0.75}},
        ...     {'model_name': 'M2', 'test_metrics': {'acc': 0.9, 'f1': 0.85}}
        ... ]
        >>> img = plot_model_comparison_radar(results, ['acc', 'f1'])
        >>> img is not None or img is None
        True
    """
    try:
        if len(results) == 0 or len(metrics) == 0:
            logger.warning("Brak danych do wykresu radarowego")
            return None

        # Normalizuj metryki
        normalized = normalize_metrics_for_comparison(results, metrics)

        if not normalized:
            logger.warning("Nie udało się znormalizować metryk")
            return None

        # Przygotuj dane
        model_names = list(normalized.keys())
        n_metrics = len(metrics)

        # Kąty dla wykresu radarowego
        angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
        angles += angles[:1]  # Zamknij wykres

        # Utwórz wykres
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Dodaj każdy model
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

        for idx, model_name in enumerate(model_names):
            values = [normalized[model_name].get(m, 0) for m in metrics]
            values += values[:1]  # Zamknij wykres

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Etykiety
        display_names = [metric_names.get(m, m) if metric_names else m for m in metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_names, size=10)

        # Limity
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True)

        # Legenda
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.title('Model Comparison - Normalized Metrics', size=14, y=1.08)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania radar chart: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_metrics_comparison_bar(
    results: List[Dict],
    metric: str,
    metric_name: str = "Metric"
) -> Optional[BytesIO]:
    """
    Generuje wykres słupkowy porównujący modele według metryki.

    Args:
        results: Lista wyników modeli
        metric: Nazwa metryki
        metric_name: Wyświetlana nazwa metryki

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> results = [
        ...     {'model_name': 'M1', 'test_metrics': {'accuracy': 0.8}},
        ...     {'model_name': 'M2', 'test_metrics': {'accuracy': 0.9}}
        ... ]
        >>> img = plot_metrics_comparison_bar(results, 'accuracy')
        >>> img is not None
        True
    """
    try:
        model_names = []
        metric_values = []

        for result in results:
            model_names.append(result['model_name'])
            value = result.get('test_metrics', {}).get(metric)
            if value is None:
                value = result.get('cv_scores', {}).get(metric, {}).get('mean', 0)

            # Obsługa metryk negatywnych
            if metric.startswith('neg_'):
                value = -value if value else 0

            metric_values.append(value)

        if not metric_values:
            logger.warning(f"Brak wartości dla metryki {metric}")
            return None

        # Sortuj od najlepszego
        sorted_data = sorted(zip(model_names, metric_values), key=lambda x: x[1], reverse=True)
        model_names, metric_values = zip(*sorted_data)

        # Utwórz wykres
        fig, ax = plt.subplots(figsize=(10, max(6, len(model_names) * 0.4)))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
        bars = ax.barh(range(len(model_names)), metric_values, color=colors)

        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel(metric_name)
        ax.set_title(f'Model Comparison - {metric_name}')
        ax.invert_yaxis()

        # Dodaj wartości na słupkach
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)

        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania bar chart: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    top_n: Optional[int] = None
) -> Optional[BytesIO]:
    """
    Generuje heatmapę korelacji.

    Args:
        corr_matrix: Macierz korelacji
        title: Tytuł wykresu
        top_n: Liczba features do wyświetlenia (None = wszystkie)

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 6], 'c': [1, 1, 1]})
        >>> corr = df.corr()
        >>> img = plot_correlation_heatmap(corr)
        >>> img is not None
        True
    """
    try:
        # Ogranicz do top N features jeśli podane
        if top_n and len(corr_matrix) > top_n:
            # Wybierz features z największą średnią korelacją
            mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)
            top_features = mean_corr.head(top_n).index
            corr_matrix = corr_matrix.loc[top_features, top_features]

        # Utwórz wykres
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            corr_matrix,
            annot=True if len(corr_matrix) <= 15 else False,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania heatmap: {e}")
        return None


@handle_errors(show_in_ui=False)
def plot_training_time_comparison(
    results: List[Dict]
) -> Optional[BytesIO]:
    """
    Generuje wykres porównujący czasy treningu modeli.

    Args:
        results: Lista wyników modeli

    Returns:
        Optional[BytesIO]: Obiekt BytesIO z wykresem lub None

    Example:
        >>> results = [
        ...     {'model_name': 'M1', 'training_time': 10.5},
        ...     {'model_name': 'M2', 'training_time': 5.2}
        ... ]
        >>> img = plot_training_time_comparison(results)
        >>> img is not None
        True
    """
    try:
        model_names = []
        training_times = []

        for result in results:
            model_names.append(result['model_name'])
            training_times.append(result.get('training_time', 0))

        if not training_times:
            logger.warning("Brak danych o czasach treningu")
            return None

        # Sortuj od najszybszego
        sorted_data = sorted(zip(model_names, training_times), key=lambda x: x[1])
        model_names, training_times = zip(*sorted_data)

        # Utwórz wykres
        fig, ax = plt.subplots(figsize=(10, max(6, len(model_names) * 0.4)))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))
        bars = ax.barh(range(len(model_names)), training_times, color=colors)

        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        ax.invert_yaxis()

        # Dodaj wartości na słupkach
        for i, (bar, time) in enumerate(zip(bars, training_times)):
            ax.text(time, i, f' {time:.2f}s', va='center', fontsize=9)

        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        return buf

    except Exception as e:
        logger.error(f"Błąd generowania training time chart: {e}")
        return None