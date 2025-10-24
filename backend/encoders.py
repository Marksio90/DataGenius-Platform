
from __future__ import annotations
from typing import Optional, List
from sklearn.preprocessing import OneHotEncoder

class EncoderComplianceError(RuntimeError):
    """Błąd enkodera (TMIV-ML-ENC-001): wykryto niedozwolony parametr `sparse`.
    W TMIV dozwolone jest wyłącznie `sparse_output` (scikit-learn ≥ 1.5).
    """

def safe_one_hot_encoder(categories: str | list | None = "auto",
                         drop: Optional[str] = None,
                         handle_unknown: str = "ignore") -> OneHotEncoder:
    """Warstwa zgodności dla OneHotEncoder.
    Zabrania korzystania z przestarzałego parametru `sparse`.
    Zwraca OneHotEncoder z `sparse_output=True`.
    """
    # Jedyna dozwolona ścieżka:
    return OneHotEncoder(categories=categories, drop=drop, handle_unknown=handle_unknown, sparse_output=True)
