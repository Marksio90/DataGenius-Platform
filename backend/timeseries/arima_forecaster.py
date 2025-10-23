"""
ARIMA Forecaster - Autoregressive Integrated Moving Average.

Funkcjonalności:
- Auto ARIMA (automatyczny dobór (p,d,q))
- Seasonal ARIMA (SARIMA)
- Diagnostyka reszt
- Forecasting z confidence intervals
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import pmdarima as pm
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Statsmodels/pmdarima nie są zainstalowane")

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    ARIMA-based time series forecaster.
    
    Wspiera:
    - ARIMA(p,d,q)
    - SARIMA(p,d,q)(P,D,Q,s)
    - Auto ARIMA (automatyczny dobór parametrów)
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto: bool = True,
        seasonal: bool = True,
        trace: bool = False
    ):
        """
        Inicjalizacja ARIMA forecaster.

        Args:
            order: (p, d, q) - jeśli None, użyj auto ARIMA
            seasonal_order: (P, D, Q, s) - seasonal parameters
            auto: Czy użyć auto ARIMA
            seasonal: Czy uwzględnić sezonowość
            trace: Czy wyświetlać trace auto ARIMA
        """
        if not ARIMA_AVAILABLE:
            raise ImportError("Wymagane: pip install statsmodels pmdarima")

        self.order = order
        self.seasonal_order = seasonal_order
        self.auto = auto
        self.seasonal = seasonal
        self.trace = trace

        self.model = None
        self.fitted_model = None
        self.series = None

        logger.info("ARIMA Forecaster zainicjalizowany")

    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Sprawdza stacjonarność szeregu (ADF test).

        Args:
            series: Szereg czasowy

        Returns:
            Dict: Wyniki testu ADF
        """
        result = adfuller(series.dropna())

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }

    def difference_series(self, series: pd.Series, order: int = 1) -> pd.Series:
        """
        Różnicuje szereg.

        Args:
            series: Szereg czasowy
            order: Rząd różnicowania

        Returns:
            pd.Series: Różnicowany szereg
        """
        differenced = series.copy()

        for _ in range(order):
            differenced = differenced.diff().dropna()

        return differenced

    @handle_errors(show_in_ui=False)
    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None
    ) -> 'ARIMAForecaster':
        """
        Trenuje model ARIMA.

        Args:
            series: Szereg czasowy (pd.Series z datetime index)
            exog: Zmienne egzogeniczne (opcjonalne)

        Returns:
            self
        """
        logger.info("Rozpoczęcie treningu ARIMA")

        self.series = series

        if self.auto:
            # Auto ARIMA
            logger.info("Użycie Auto ARIMA do doboru parametrów")

            self.model = pm.auto_arima(
                series,
                exogenous=exog,
                seasonal=self.seasonal,
                m=12 if self.seasonal else 1,  # Monthly seasonality default
                trace=self.trace,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            logger.info(f"Auto ARIMA wybrało: {self.model.order}, seasonal: {self.model.seasonal_order}")

        else:
            # Manual ARIMA
            if self.seasonal and self.seasonal_order:
                self.model = SARIMAX(
                    series,
                    exog=exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order
                )
            else:
                self.model = ARIMA(
                    series,
                    exog=exog,
                    order=self.order
                )

            self.fitted_model = self.model.fit()

        logger.info("ARIMA wytrenowany")

        return self

    def predict(
        self,
        steps: int = 30,
        return_conf_int: bool = True,
        alpha: float = 0.05,
        exog: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generuje predykcje.

        Args:
            steps: Liczba kroków do predykcji
            return_conf_int: Czy zwrócić confidence intervals
            alpha: Poziom istotności (1-alpha = confidence level)
            exog: Zmienne egzogeniczne dla przyszłości

        Returns:
            Tuple[pd.Series, Optional[pd.DataFrame]]: (predykcje, confidence intervals)
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany")

        if self.auto:
            # pmdarima model
            forecast, conf_int = self.model.predict(
                n_periods=steps,
                return_conf_int=return_conf_int,
                alpha=alpha,
                exogenous=exog
            )

            # Create index
            last_date = self.series.index[-1]
            freq = pd.infer_freq(self.series.index) or 'D'
            forecast_index = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=freq
            )[1:]

            forecast_series = pd.Series(forecast, index=forecast_index)

            if return_conf_int:
                conf_int_df = pd.DataFrame(
                    conf_int,
                    index=forecast_index,
                    columns=['lower', 'upper']
                )
                return forecast_series, conf_int_df
            else:
                return forecast_series, None

        else:
            # statsmodels model
            forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog)

            forecast_series = forecast_result.predicted_mean

            if return_conf_int:
                conf_int_df = forecast_result.conf_int(alpha=alpha)
                return forecast_series, conf_int_df
            else:
                return forecast_series, None

    def get_residuals(self) -> pd.Series:
        """Zwraca reszty modelu."""
        if self.auto:
            return pd.Series(self.model.resid())
        else:
            return self.fitted_model.resid

    def summary(self) -> str:
        """Zwraca podsumowanie modelu."""
        if self.auto:
            return str(self.model.summary())
        else:
            return str(self.fitted_model.summary())

    def plot_diagnostics(self):
        """Generuje wykresy diagnostyczne."""
        import matplotlib.pyplot as plt

        if self.auto:
            self.model.plot_diagnostics(figsize=(12, 8))
        else:
            self.fitted_model.plot_diagnostics(figsize=(12, 8))

        plt.tight_layout()
        return plt.gcf()

    def plot_forecast(
        self,
        forecast: pd.Series,
        conf_int: Optional[pd.DataFrame] = None,
        n_history: int = 100
    ):
        """
        Generuje wykres predykcji.

        Args:
            forecast: Predykcje
            conf_int: Confidence intervals
            n_history: Liczba punktów historycznych do pokazania

        Returns:
            Figure: Matplotlib figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Historia
        history = self.series[-n_history:]
        ax.plot(history.index, history.values, label='Historical', color='blue')

        # Forecast
        ax.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')

        # Confidence intervals
        if conf_int is not None:
            ax.fill_between(
                forecast.index,
                conf_int['lower'],
                conf_int['upper'],
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )

        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('ARIMA Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def get_params(self) -> Dict:
        """Zwraca parametry modelu."""
        if self.auto:
            return {
                'order': self.model.order,
                'seasonal_order': self.model.seasonal_order,
                'aic': self.model.aic(),
                'bic': self.model.bic()
            }
        else:
            return {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic
            }