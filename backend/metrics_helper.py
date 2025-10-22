from __future__ import annotations
from typing import Tuple, Optional, Dict, Any


def normalize_metric(metric: Optional[str], problem_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    Zwraca tuple (nazwa_wewnętrzna, parametry), gdzie:
      • klasyfikacja:
          - "f1" + {"average": "..."}  (weighted/macro/micro)
          - "accuracy", "balanced_accuracy", "precision", "recall",
          - "roc_auc" (alias: "auc"),
          - "log_loss" (alias: "logloss", "neg_log_loss"),
          - "kappa", "mcc", "average_precision" (PR-AUC)
      • regresja:
          - "rmse", "mae", "mse", "r2", "mape"

    Wewnętrzne nazwy są spójne na potrzeby Twoich metryk/UI.
    (Do scikit-learn możesz użyć własnego helpera mapującego na scoring, jeśli potrzebujesz.)
    """
    pt = (problem_type or "").strip().lower()
    is_class = pt.startswith("class")

    if not metric:
        return ("f1", {"average": "weighted"}) if is_class else ("rmse", {})

    m = str(metric).strip().lower()
    m = m.replace(" ", "_").replace("-", "_")

    # --- aliasy ogólne
    aliases = {
        "acc": "accuracy",
        "bal_acc": "balanced_accuracy",
        "balanced_acc": "balanced_accuracy",
        "auc": "roc_auc",
        "pr_auc": "average_precision",   # PR-AUC
        "logloss": "log_loss",
        "neg_log_loss": "log_loss",
        "f1w": "f1_weighted",
        "f1_w": "f1_weighted",
        "f1weighted": "f1_weighted",
        "f1_macro_avg": "f1_macro",
        "f1_micro_avg": "f1_micro",
    }
    m = aliases.get(m, m)

    if is_class:
        # F1 i warianty
        if m in {"f1_weighted"}:
            return ("f1", {"average": "weighted"})
        if m in {"f1_macro"}:
            return ("f1", {"average": "macro"})
        if m in {"f1_micro"}:
            return ("f1", {"average": "micro"})
        if m in {"f1"}:
            # domyślnie weighted (najczęściej oczekiwane w business-case’ach)
            return ("f1", {"average": "weighted"})

        # Metryki z averaging
        if m in {"precision_weighted"}:
            return ("precision", {"average": "weighted"})
        if m in {"precision_macro"}:
            return ("precision", {"average": "macro"})
        if m in {"precision_micro"}:
            return ("precision", {"average": "micro"})
        if m in {"recall_weighted"}:
            return ("recall", {"average": "weighted"})
        if m in {"recall_macro"}:
            return ("recall", {"average": "macro"})
        if m in {"recall_micro"}:
            return ("recall", {"average": "micro"})

        # Proste jednosłowne
        if m in {
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "roc_auc",
            "log_loss",
            "kappa",
            "mcc",
            "average_precision",  # PR-AUC
        }:
            return (m, {})

        # Fallback: traktuj nieznane jako f1 weighted (bezpieczny default)
        return ("f1", {"average": "weighted"})

    # --- regresja
    if m in {"rmse", "mae", "mse", "r2", "mape"}:
        return (m, {})

    # Fallback regresji
    return ("rmse", {})


def safe_f1_weighted(y_true, y_pred, problem_type: Optional[str] = None):
    """
    Bezpieczne wyliczenie F1 (average='weighted') tylko dla klasyfikacji.
    Zwraca None dla regresji lub gdy nie można policzyć.
    """
    try:
        if problem_type and problem_type.lower().startswith("classif"):
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average="weighted")
        return None
    except Exception:
        return None
