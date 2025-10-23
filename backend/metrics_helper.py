"""
Helper funkcje dla metryk ML.

Funkcjonalności:
- Obliczanie dodatkowych metryk
- Normalizacja metryk dla porównania
- Formatowanie metryk
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def normalize_metrics_for_comparison(
    results: List[Dict],
    metrics: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Normalizuje metryki do zakresu [0, 1] dla porównania na radarze.

    Args:
        results: Lista wyników modeli
        metrics: Lista nazw metryk

    Returns:
        Dict: Znormalizowane metryki dla każdego modelu

    Example:
        >>> results = [
        ...     {'model_name': 'Model1', 'test_metrics': {'accuracy': 0.8, 'f1': 0.75}},
        ...     {'model_name': 'Model2', 'test_metrics': {'accuracy': 0.9, 'f1': 0.85}}
        ... ]
        >>> normalized = normalize_metrics_for_comparison(results, ['accuracy', 'f1'])
        >>> 'Model1' in normalized
        True
    """
    # Zbierz wartości dla każdej metryki
    metric_values = {metric: [] for metric in metrics}
    model_names = []

    for result in results:
        model_names.append(result['model_name'])
        test_metrics = result.get('test_metrics', {})

        for metric in metrics:
            value = test_metrics.get(metric)
            if value is not None:
                # Obsługa metryk negatywnych (RMSE, MAE)
                if metric.startswith('neg_'):
                    value = -value
                metric_values[metric].append(value)
            else:
                metric_values[metric].append(None)

    # Normalizuj każdą metrykę
    normalized = {name: {} for name in model_names}

    for metric in metrics:
        values = metric_values[metric]
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            continue

        min_val = min(valid_values)
        max_val = max(valid_values)

        # Uniknij dzielenia przez zero
        if max_val == min_val:
            for i, name in enumerate(model_names):
                if values[i] is not None:
                    normalized[name][metric] = 1.0
            continue

        # Normalizuj
        for i, name in enumerate(model_names):
            if values[i] is not None:
                norm_value = (values[i] - min_val) / (max_val - min_val)
                normalized[name][metric] = norm_value

    return normalized


def format_metric_value(metric_name: str, value: float, decimals: int = 4) -> str:
    """
    Formatuje wartość metryki do wyświetlenia.

    Args:
        metric_name: Nazwa metryki
        value: Wartość
        decimals: Liczba miejsc po przecinku

    Returns:
        str: Sformatowana wartość

    Example:
        >>> format_metric_value('accuracy', 0.8567)
        '0.8567'
        >>> format_metric_value('accuracy', 0.8567, decimals=2)
        '0.86'
    """
    # Obsługa metryk negatywnych
    if metric_name.startswith('neg_'):
        value = -value

    return f"{value:.{decimals}f}"


def get_metric_direction(metric_name: str) -> str:
    """
    Zwraca kierunek optymalizacji metryki.

    Args:
        metric_name: Nazwa metryki

    Returns:
        str: 'maximize' lub 'minimize'

    Example:
        >>> get_metric_direction('accuracy')
        'maximize'
        >>> get_metric_direction('rmse')
        'minimize'
    """
    minimize_metrics = [
        'rmse', 'mae', 'mse', 'error',
        'neg_root_mean_squared_error',
        'neg_mean_absolute_error',
        'neg_mean_squared_error'
    ]

    for min_metric in minimize_metrics:
        if min_metric in metric_name.lower():
            return 'minimize'

    return 'maximize'