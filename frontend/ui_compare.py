"""
UI do porównywania modeli.

Funkcjonalności:
- Tabela porównawcza metryk
- Radar chart
- Ranking modeli
"""

import logging
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from backend.plots_plus import (
    plot_metrics_comparison_bar,
    plot_model_comparison_radar,
    plot_training_time_comparison,
)

logger = logging.getLogger(__name__)


def render_model_comparison_table(
    results: List[Dict],
    metrics: List[str]
) -> pd.DataFrame:
    """
    Renderuje tabelę porównawczą modeli.

    Args:
        results: Lista wyników modeli
        metrics: Lista metryk do wyświetlenia

    Returns:
        pd.DataFrame: DataFrame z wynikami

    Example:
        >>> # W aplikacji Streamlit
        >>> results = [{'model_name': 'M1', 'test_metrics': {'acc': 0.8}}]
        >>> df = render_model_comparison_table(results, ['acc'])
    """
    st.subheader("📊 Porównanie Modeli")

    # Przygotuj dane
    rows = []
    for result in results:
        row = {
            'Model': result['model_name'],
            'Training Time (s)': f"{result.get('training_time', 0):.2f}"
        }

        # Dodaj metryki testowe
        for metric in metrics:
            value = result.get('test_metrics', {}).get(metric)
            if value is not None:
                # Obsługa metryk negatywnych
                if metric.startswith('neg_'):
                    value = -value
                row[metric] = f"{value:.4f}"
            else:
                row[metric] = "N/A"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Wyświetl tabelę
    st.dataframe(df, use_container_width=True)

    return df


def render_radar_comparison(
    results: List[Dict],
    metrics: List[str],
    metric_names: Optional[Dict[str, str]] = None
) -> Optional[BytesIO]:
    """
    Renderuje radar chart porównujący modele.

    Args:
        results: Lista wyników modeli
        metrics: Lista metryk
        metric_names: Mapowanie nazw metryk

    Returns:
        Optional[BytesIO]: Wykres lub None

    Example:
        >>> # W aplikacji Streamlit
        >>> # radar_img = render_radar_comparison(results, ['acc', 'f1'])
    """
    st.subheader("🎯 Radar Chart - Porównanie Metryk")

    with st.spinner("Generowanie radar chart..."):
        radar_img = plot_model_comparison_radar(results, metrics, metric_names)

        if radar_img:
            st.image(radar_img, caption="Model Comparison Radar", use_container_width=True)
            st.caption("Metryki znormalizowane do zakresu [0, 1] dla porównywalności")
            return radar_img
        else:
            st.warning("Nie udało się wygenerować radar chart")
            return None


def render_metric_comparison_bar(
    results: List[Dict],
    metric: str,
    metric_name: str
) -> Optional[BytesIO]:
    """
    Renderuje wykres słupkowy dla pojedynczej metryki.

    Args:
        results: Lista wyników modeli
        metric: Nazwa metryki
        metric_name: Wyświetlana nazwa

    Returns:
        Optional[BytesIO]: Wykres lub None

    Example:
        >>> # W aplikacji Streamlit
        >>> # bar_img = render_metric_comparison_bar(results, 'accuracy', 'Accuracy')
    """
    with st.spinner(f"Generowanie wykresu {metric_name}..."):
        bar_img = plot_metrics_comparison_bar(results, metric, metric_name)

        if bar_img:
            st.image(bar_img, caption=f"{metric_name} Comparison", use_container_width=True)
            return bar_img
        else:
            st.warning(f"Nie udało się wygenerować wykresu {metric_name}")
            return None


def render_training_time_comparison(results: List[Dict]) -> Optional[BytesIO]:
    """
    Renderuje porównanie czasów treningu.

    Args:
        results: Lista wyników modeli

    Returns:
        Optional[BytesIO]: Wykres lub None

    Example:
        >>> # W aplikacji Streamlit
        >>> # time_img = render_training_time_comparison(results)
    """
    st.subheader("⏱️ Porównanie Czasów Treningu")

    with st.spinner("Generowanie wykresu czasów treningu..."):
        time_img = plot_training_time_comparison(results)

        if time_img:
            st.image(time_img, caption="Training Time Comparison", use_container_width=True)
            return time_img
        else:
            st.warning("Nie udało się wygenerować wykresu czasów")
            return None


def render_model_ranking(
    ranking: List[Dict],
    metric_name: str = "Score"
) -> None:
    """
    Renderuje ranking modeli.

    Args:
        ranking: Lista modeli z rankingiem
        metric_name: Nazwa metryki rankingu

    Example:
        >>> # W aplikacji Streamlit
        >>> ranking = [{'model_name': 'M1', 'score': 0.9, 'training_time': 10}]
        >>> render_model_ranking(ranking)
    """
    st.subheader("🏆 Ranking Modeli")

    for idx, model_info in enumerate(ranking, 1):
        model_name = model_info['model_name']
        score = model_info['score']
        time = model_info['training_time']

        # Medal dla top 3
        medal = ""
        if idx == 1:
            medal = "🥇"
        elif idx == 2:
            medal = "🥈"
        elif idx == 3:
            medal = "🥉"

        col1, col2, col3 = st.columns([1, 3, 2])

        with col1:
            st.markdown(f"### {medal} #{idx}")

        with col2:
            st.markdown(f"**{model_name}**")
            st.caption(f"{metric_name}: {score:.4f}")

        with col3:
            st.metric("Training Time", f"{time:.2f}s")

        if idx < len(ranking):
            st.markdown("---")