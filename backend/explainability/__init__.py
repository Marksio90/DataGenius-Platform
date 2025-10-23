"""
Explainability module - interpretacja modeli ML.
"""

__version__ = "3.0.0"

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .whatif_analyzer import WhatIfAnalyzer
from .fairness_checker import FairnessChecker

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'WhatIfAnalyzer',
    'FairnessChecker',
]