# core/interfaces.py
"""
Interfejsy (Protocols) dla usług TMIV – Advanced ML Platform.

Cel
---
Zdefiniować cienkie, stabilne kontrakty pomiędzy warstwami:
- Data / EDA
- ML (plan + trening)
- Explain / Export (wykresy, PDF/ZIP)
- Insights (LLM/heurystyki)
- Cache
- Telemetry
- DB (opcjonalnie)

Uwaga
-----
- To jest lekki moduł typów – zero ciężkich zależności w runtime.
- Importy pandas/numpy odbywają się wyłącznie dla typowania (`TYPE_CHECKING`).
- Implementacje powinny żyć w `core/services/*` lub `backend/*` i spełniać te protokoły.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, TypedDict, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # tylko do typów, brak zależności w runtime
    import numpy as np
    import pandas as pd


# =========================
# Wspólne typy danych
# =========================

class TrainResult(TypedDict, total=False):
    """Znormalizowany wynik trenowania (kluczowe pola)."""
    problem_type: str
    target: str
    split: Dict[str, Any]
    leaderboard: "pd.DataFrame"
    best_model_name: str
    models: Dict[str, Any]
    preprocessor: Any
    feature_names: Sequence[str]
    feature_importance: "pd.DataFrame"
    y_encoder: Any
    metrics_by_model: Dict[str, Dict[str, float]]
    cv_by_model: Dict[str, Dict[str, float]]
    y_valid: "np.ndarray"
    y_mapping: Dict[int, str]


class PlotsDict(TypedDict, total=False):
    """Mapa nazwa->ścieżka pliku PNG."""
    roc: str
    pr: str
    cm: str
    calibration: str
    residual_hist: str
    pred_vs_true: str
    residuals_vs_pred: str
    feature_importance: str
    radar: str


# =========================
# DATA / EDA
# =========================

@runtime_checkable
class IDataService(Protocol):
    """Ładowanie i normalizacja danych."""

    def load_any(self, src: Any) -> Tuple["pd.DataFrame", str]:
        """Wczytaj dowolny wspierany format (CSV/XLSX/JSON/Parquet). Zwraca (df, nazwa_pliku)."""

    def load_example(self) -> "pd.DataFrame":
        """Wczytaj przykładowy dataset repo (np. data/avocado.csv)."""

    def normalize_columns(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Unikalne, snake_case nazwy kolumn."""

    def fingerprint(self, df: "pd.DataFrame") -> str:
        """Stabilny odcisk danych dla cache/eksportów."""

    def profile_html(self, df: "pd.DataFrame", *, title: str = "TMIV – EDA Profile") -> Optional[str]:
        """Opcjonalny raport EDA (HTML) – zwraca ścieżkę do pliku lub None."""


@runtime_checkable
class IEDAService(Protocol):
    """Eksploracyjna analiza danych (szybkie statystyki i korelacje)."""

    def overview(self, df: "pd.DataFrame") -> Dict[str, Any]:
        """Statystyki, brakujące, typy kolumn, przykłady wartości."""

    def correlations(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Macierz korelacji numerycznych."""

    def distributions(self, df: "pd.DataFrame") -> Dict[str, Any]:
        """Opis rozkładów (np. histogramy/biny – przygotowane dane do wizualizacji)."""

    def profile_html(self, df: "pd.DataFrame") -> Optional[str]:
        """Przepięcie do IDataService.profile_html (wygoda dla UI)."""


# =========================
# ML (plan + trening)
# =========================

@runtime_checkable
class IMLService(Protocol):
    """Plan trenowania, trening wielu modeli, ewaluacja i leaderboard."""

    def build_training_plan(
        self,
        df: "pd.DataFrame",
        target: str,
        *,
        strategy: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Heurystyki planu trenowania (metryki, CV, strategia, tuning/ensembles)."""

    def train_and_evaluate(
        self,
        df: "pd.DataFrame",
        target: str,
        *,
        plan: Optional[Mapping[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: Optional[int] = 3,
        enable_ensembles: bool = True,
        n_jobs: int = -1,
    ) -> TrainResult:
        """Trenowanie wielu modeli, metryki, leaderboard, FI i artefakty."""


# =========================
# Explain / Plots / Export
# =========================

@runtime_checkable
class IExplainService(Protocol):
    """Wizualizacje, PDF explainability i pełny eksport artefaktów."""

    # --- wykresy ---
    def plot_classification_curves(
        self, y_true: "np.ndarray | Sequence[int]", y_proba: "np.ndarray", *, model_name: str = "model"
    ) -> Dict[str, str]:
        """ROC i PR – zwraca {"roc": path, "pr": path}."""

    def plot_confusion_matrix(
        self, y_true: "np.ndarray | Sequence[int]", y_pred: "np.ndarray | Sequence[int]", *, class_names: Optional[Sequence[str]] = None, model_name: str = "model"
    ) -> str:
        """Macierz pomyłek – zwraca ścieżkę PNG."""

    def plot_calibration_curve(
        self, y_true: "np.ndarray | Sequence[int]", y_proba: "np.ndarray", *, model_name: str = "model", n_bins: int = 10
    ) -> str:
        """Wykres kalibracji – zwraca ścieżkę PNG."""

    def plot_regression_diagnostics(
        self, y_true: "np.ndarray | Sequence[float]", y_pred: "np.ndarray | Sequence[float]", *, model_name: str = "model"
    ) -> Dict[str, str]:
        """Diagnostyka regresji – zwraca mapę nazw do PNG."""

    def plot_feature_importance(
        self, fi: "pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[Tuple[str, float]]", *, top_k: int = 20, name: str = "fi"
    ) -> str:
        """Wykres ważności cech – zwraca ścieżkę PNG."""

    def plot_radar_leaderboard(
        self, leaderboard: "pd.DataFrame", *, metrics: Optional[Sequence[str]] = None, name: str = "radar"
    ) -> str:
        """Radar porównania modeli – zwraca ścieżkę PNG."""

    # --- PDF / ZIP ---
    def build_pdf_report(
        self,
        output_path: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        feature_importance: Any | None = None,
        charts: Mapping[str, str] | Sequence[Tuple[str, str]] | None = None,
        title: str = "Explainability Report",
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        problem_type: Optional[str] = None,
        created_by: Optional[str] = None,
        run_id: Optional[str] = None,
        notes: Sequence[str] | str | None = None,
        page_size: str = "A4",
    ) -> str:
        """Wygeneruj PDF explainability (ReportLab) – zwraca ścieżkę pliku."""

    def export_everything(
        self,
        run_id: str,
        *,
        problem_type: str,
        metrics: Mapping[str, Any],
        dataset_name: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        plan: Mapping[str, Any] | None = None,
        cv_metrics: Sequence[Mapping[str, Any]] | None = None,
        leaderboard: Any | None = None,                    # pd.DataFrame | None
        feature_importance: Any | None = None,             # pd.DataFrame | None
        models: Mapping[str, Any] | None = None,
        plots: Mapping[str, str] | None = None,
        configs: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        notes: str | None = None,
        exports_dir: str | None = None,
    ) -> str:
        """Spakuj pełny eksport artefaktów do ZIP – zwraca ścieżkę ZIP."""


# =========================
# Insights (AI/heurystyki)
# =========================

@runtime_checkable
class IInsightsService(Protocol):
    """Opisy kolumn i rekomendacje biznesowo-ML-owe."""

    def describe_columns(self, df: "pd.DataFrame") -> Dict[str, str]:
        """Słownik {kolumna: opis}. Fallback heurystyczny gdy brak kluczy LLM."""

    def generate_recommendations(self, context: Mapping[str, Any]) -> Sequence[str]:
        """Lista rekomendacji/wniosków na podstawie metryk, FI i charakterystyki problemu."""


# =========================
# Cache (fasada)
# =========================

@runtime_checkable
class ICacheService(Protocol):
    """Prosta fasada nad cache/artifacts."""

    def cached_path(self, subdir: str, name: str) -> Path:
        """Ścieżka pliku w cache/artifacts/<subdir>/<name> (tworzy katalogi)."""

    def cache_result(self, namespace: str = "default", ttl: Optional[int] = None):
        """Dekorator cache'ujący wynik funkcji (jeśli dostępny backend cache)."""

    def df_fingerprint(self, df: "pd.DataFrame") -> str:
        """Stabilny odcisk danych."""


# =========================
# Telemetry
# =========================

@runtime_checkable
class ITelemetryService(Protocol):
    """Minimalne API telemetry (OpenTelemetry albo fallback do logów)."""

    def start_span(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> ContextManager[None]:
        """Context manager na span."""

    def add_event(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        ...

    def incr_counter(self, name: str, value: float = 1.0, attrs: Optional[Dict[str, Any]] = None) -> None:
        ...

    def observe_value(self, name: str, value: float, attrs: Optional[Dict[str, Any]] = None) -> None:
        ...

    def start_heartbeat(self, interval_sec: Optional[float] = None) -> None:
        ...

    def stop_heartbeat(self) -> None:
        ...

    def snapshot(self) -> Dict[str, Any]:
        ...


# =========================
# DB (opcjonalnie)
# =========================

@runtime_checkable
class IDBService(Protocol):
    """Abstrakcja nad warstwą bazy (opcjonalne)."""

    def save_dataset_meta(self, *, name: str, fingerprint: str, rows: int, cols: int) -> bool:
        """Zapis metadanych zbioru."""

    def save_metrics(self, *, run_id: str, model_name: str, metrics: Mapping[str, float]) -> bool:
        """Zapis metryk modelu."""

    def save_feature_importance(self, *, run_id: str, df: "pd.DataFrame") -> bool:
        """Zapis FI (DataFrame)."""

    def save_leaderboard(self, *, run_id: str, df: "pd.DataFrame") -> bool:
        """Zapis leaderboardu."""

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Odczyt danych runu (jeśli zaimplementowane)."""


__all__ = [
    "TrainResult",
    "PlotsDict",
    "IDataService",
    "IEDAService",
    "IMLService",
    "IExplainService",
    "IInsightsService",
    "ICacheService",
    "ITelemetryService",
    "IDBService",
]
