"""
Prophet Forecaster - Facebook Prophet dla time series.

Funkcjonalności:
- Automatyczna detekcja sezonowości
- Obsługa holidays
- Trend changes
- Uncertainty intervals
- Cross-validation
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Prophet nie jest zainstalowany - funkcjonalność niedostępna")

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Prophet-based time series forecaster.
    
    Automatycznie wykrywa:
    - Trend (liniowy, logistyczny)
    - Sezonowość (dzienna, tygodniowa, roczna)
    - Punkty zmiany trendu
    - Holidays i events
    """

    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive',
        yearly_seasonality: str = 'auto',
        weekly_seasonality: str = 'auto',
        daily_seasonality: str = 'auto',
        interval_width: float = 0.80
    ):
        """
        Inicjalizacja Prophet forecaster.

        Args:
            growth: Typ trendu ('linear' lub 'logistic')
            changepoint_prior_scale: Elastyczność trendu
            seasonality_prior_scale: Siła sezonowości
            holidays_prior_scale: Siła holidays
            seasonality_mode: 'additive' lub 'multiplicative'
            yearly_seasonality: Sezonowość roczna
            weekly_seasonality: Sezonowość tygodniowa
            daily_seasonality: Sezonowość dzienna
            interval_width: Szerokość przedziałów niepewności
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet nie jest zainstalowany: pip install prophet")

        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width

        self.model = None
        self.df_train = None

        logger.info("Prophet Forecaster zainicjalizowany")

    @handle_errors(show_in_ui=False)
    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = 'ds',
        target_col: str = 'y',
        additional_regressors: Optional[List[str]] = None
    ) -> 'ProphetForecaster':
        """
        Trenuje model Prophet.

        Args:
            df: DataFrame z danymi
            date_col: Nazwa kolumny z datami
            target_col: Nazwa kolumny z wartościami
            additional_regressors: Lista dodatkowych regresorów

        Returns:
            self
        """
        logger.info("Rozpoczęcie treningu Prophet")

        # Przygotuj dane
        if date_col != 'ds' or target_col != 'y':
            df = df.rename(columns={date_col: 'ds', target_col: 'y'})

        # Convert ds to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])

        self.df_train = df[['ds', 'y']].copy()

        # Initialize model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width
        )

        # Add regressors
        if additional_regressors:
            for regressor in additional_regressors:
                if regressor in df.columns:
                    self.model.add_regressor(regressor)
                    self.df_train[regressor] = df[regressor]

        # Fit
        self.model.fit(self.df_train)

        logger.info("Prophet wytrenowany")

        return self

    def predict(
        self,
        periods: int = 30,
        freq: str = 'D',
        include_history: bool = True
    ) -> pd.DataFrame:
        """
        Generuje predykcje.

        Args:
            periods: Liczba okresów do predykcji
            freq: Częstotliwość ('D' = daily, 'W' = weekly, 'M' = monthly)
            include_history: Czy dołączyć dane historyczne

        Returns:
            pd.DataFrame: Predykcje z kolumnami [ds, yhat, yhat_lower, yhat_upper]
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)

        # Predict
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]

    def cross_validate(
        self,
        horizon: str = '30 days',
        initial: str = '365 days',
        period: str = '180 days'
    ) -> pd.DataFrame:
        """
        Przeprowadza cross-validation.

        Args:
            horizon: Horyzont predykcji
            initial: Początkowy okres treningu
            period: Okres między cutoffs

        Returns:
            pd.DataFrame: Wyniki CV
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        logger.info("Rozpoczęcie cross-validation")

        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )

        # Compute metrics
        df_metrics = performance_metrics(df_cv)

        logger.info("Cross-validation zakończona")

        return df_cv, df_metrics

    def get_changepoints(self) -> pd.DataFrame:
        """
        Zwraca punkty zmiany trendu.

        Returns:
            pd.DataFrame: Changepoints
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        changepoints = self.model.changepoints
        deltas = self.model.params['delta'][0]

        df_changepoints = pd.DataFrame({
            'changepoint': changepoints,
            'delta': deltas
        })

        return df_changepoints

    def plot_forecast(self, forecast: pd.DataFrame):
        """
        Generuje wykres predykcji.

        Args:
            forecast: DataFrame z predykcjami

        Returns:
            Figure: Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        fig = self.model.plot(forecast)
        return fig

    def plot_components(self, forecast: pd.DataFrame):
        """
        Generuje wykres komponentów (trend, sezonowość).

        Args:
            forecast: DataFrame z predykcjami

        Returns:
            Figure: Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        fig = self.model.plot_components(forecast)
        return fig

    def get_params(self) -> Dict:
        """Zwraca parametry modelu."""
        return {
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'interval_width': self.interval_width
        }