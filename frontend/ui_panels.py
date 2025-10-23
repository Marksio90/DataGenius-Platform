"""
Panele wyników ML - wizualizacje i metryki.

Funkcjonalności:
- Panel wyników klasyfikacji
- Panel wyników regresji
- Panel feature importance
- Panel porównania modeli
"""

import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from backend.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


def render_classification_results(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: Dict,
    class_names: Optional[List[str]] = None
) -> Dict[str, BytesIO]:
    """
    Renderuje wyniki dla klasyfikacji.

    Args:
        model_name: Nazwa modelu
        model: Wytrenowany model
        X_test: Features testowe
        y_test: Target testowy
        metrics: Słownik z metrykami
        class_names: Nazwy klas

    Returns:
        Dict[str, BytesIO]: Słownik z wygenerowanymi wykresami

    Example:
        >>> # W aplikacji Streamlit
        >>> # plots = render_classification_results(...)
    """
    st.subheader(f"📊 Wyniki: {model_name}")

    plots = {}

    # Metryki w kolumnach
    cols = st.columns(len(metrics))
    for col, (metric_name, metric_value) in zip(cols, metrics.items()):
        with col:
            st.metric(metric_name.upper(), f"{metric_value:.4f}")

    st.markdown("---")

    # Predykcje
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    except Exception as e:
        st.error(f"Błąd predykcji: {e}")
        return plots

    # Dla klasyfikacji binarnej
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        y_proba_pos = y_proba[:, 1]

        # ROC Curve
        with st.spinner("Generowanie ROC Curve..."):
            roc_img = plot_roc_curve(y_test, y_proba_pos, model_name)
            if roc_img:
                st.image(roc_img, caption="ROC Curve", use_container_width=True)
                plots['roc_curve'] = roc_img

        # Precision-Recall Curve
        with st.spinner("Generowanie Precision-Recall Curve..."):
            pr_img = plot_precision_recall_curve(y_test, y_proba_pos, model_name)
            if pr_img:
                st.image(pr_img, caption="Precision-Recall Curve", use_container_width=True)
                plots['pr_curve'] = pr_img

        # Calibration Curve
        with st.spinner("Generowanie Calibration Curve..."):
            cal_img = plot_calibration_curve(y_test, y_proba_pos, model_name)
            if cal_img:
                st.image(cal_img, caption="Calibration Curve", use_container_width=True)
                plots['calibration_curve'] = cal_img

    # Confusion Matrix (dla wszystkich typów klasyfikacji)
    with st.spinner("Generowanie Confusion Matrix..."):
        cm_img = plot_confusion_matrix(y_test, y_pred, class_names, model_name)
        if cm_img:
            st.image(cm_img, caption="Confusion Matrix", use_container_width=True)
            plots['confusion_matrix'] = cm_img

    return plots


def render_regression_results(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: Dict
) -> Dict[str, BytesIO]:
    """
    Renderuje wyniki dla regresji.

    Args:
        model_name: Nazwa modelu
        model: Wytrenowany model
        X_test: Features testowe
        y_test: Target testowy
        metrics: Słownik z metrykami

    Returns:
        Dict[str, BytesIO]: Słownik z wygenerowanymi wykresami

    Example:
        >>> # W aplikacji Streamlit
        >>> # plots = render_regression_results(...)
    """
    st.subheader(f"📊 Wyniki: {model_name}")

    plots = {}

    # Metryki w kolumnach
    cols = st.columns(len(metrics))
    for col, (metric_name, metric_value) in zip(cols, metrics.items()):
        with col:
            # Dla metryk negatywnych pokaż wartość bezwzględną
            display_value = abs(metric_value) if metric_name.startswith('neg_') else metric_value
            st.metric(metric_name.upper(), f"{display_value:.4f}")

    st.markdown("---")

    # Predykcje
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"Błąd predykcji: {e}")
        return plots

    # Predicted vs Actual
    with st.spinner("Generowanie Predicted vs Actual..."):
        pred_vs_actual_img = plot_predictions_vs_actual(y_test, y_pred, model_name)
        if pred_vs_actual_img:
            st.image(pred_vs_actual_img, caption="Predicted vs Actual", use_container_width=True)
            plots['pred_vs_actual'] = pred_vs_actual_img

    # Residuals
    with st.spinner("Generowanie Residuals Plot..."):
        residuals_img = plot_residuals(y_test, y_pred, model_name)
        if residuals_img:
            st.image(residuals_img, caption="Residuals Analysis", use_container_width=True)
            plots['residuals'] = residuals_img

    return plots


def render_feature_importance_panel(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20
) -> Optional[BytesIO]:
    """
    Renderuje panel feature importance.

    Args:
        feature_names: Lista nazw features
        importances: Wartości importance
        top_n: Liczba top features

    Returns:
        Optional[BytesIO]: Wykres lub None

    Example:
        >>> # W aplikacji Streamlit
        >>> # fi_img = render_feature_importance_panel(['f1', 'f2'], [0.6, 0.4])
    """
    st.subheader("🔍 Feature Importance")

    if len(feature_names) == 0 or len(importances) == 0:
        st.warning("Brak danych feature importance")
        return None

    # Utwórz DataFrame
    df_fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Top N w tabeli
    st.markdown(f"**Top {top_n} najważniejszych features:**")
    st.dataframe(df_fi.head(top_n), use_container_width=True)

    # Wykres
    with st.spinner("Generowanie wykresu feature importance..."):
        fi_img = plot_feature_importance(feature_names, importances, top_n=top_n)
        if fi_img:
            st.image(fi_img, caption="Feature Importance", use_container_width=True)
            return fi_img

    return None


def render_model_details_expander(
    model_name: str,
    result: Dict
) -> None:
    """
    Renderuje expander ze szczegółami modelu.

    Args:
        model_name: Nazwa modelu
        result: Słownik z wynikami modelu

    Example:
        >>> # W aplikacji Streamlit
        >>> # render_model_details_expander('RandomForest', result_dict)
    """
    with st.expander(f"🔍 Szczegóły: {model_name}"):
        # CV Scores
        cv_scores = result.get('cv_scores', {})
        if cv_scores:
            st.markdown("**Cross-Validation Scores:**")
            for metric, scores_dict in cv_scores.items():
                mean = scores_dict.get('mean', 0)
                std = scores_dict.get('std', 0)
                st.text(f"{metric}: {mean:.4f} (±{std:.4f})")

        # Test Metrics
        test_metrics = result.get('test_metrics', {})
        if test_metrics:
            st.markdown("**Test Metrics:**")
            for metric, value in test_metrics.items():
                st.text(f"{metric}: {value:.4f}")

        # Training Time
        training_time = result.get('training_time', 0)
        st.text(f"Training Time: {training_time:.2f}s")

        # Parameters
        params = result.get('params', {})
        if params:
            st.markdown("**Parameters:**")
            st.json(params)