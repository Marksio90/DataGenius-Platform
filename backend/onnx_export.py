"""
onnx_export.py
Docstring (PL): Eksport ONNX (gated). Wymaga skl2onnx/onnxruntime; jeśli brak — zwraca komunikat.
"""
from __future__ import annotations
from typing import Any, Dict
import os

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

def export_onnx(pipe, X_sample, out_path: str) -> Dict[str, Any]:
    if not HAS_ONNX:
        return {"status":"TMIV-ONNX-000","message":"Brak pakietów ONNX (skl2onnx, onnxruntime)."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_features = X_sample.shape[1] if hasattr(X_sample, "shape") else len(X_sample.columns)
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return {"status":"OK", "path": out_path}