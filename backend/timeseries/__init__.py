"""
Time Series module - forecasting i analiza szeregów czasowych.
"""

__version__ = "3.0.0"

from .prophet_forecaster import ProphetForecaster
from .arima_forecaster import ARIMAForecaster
from .lstm_forecaster import LSTMForecaster

__all__ = [
    'ProphetForecaster',
    'ARIMAForecaster',
    'LSTMForecaster',
]