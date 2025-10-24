
from __future__ import annotations
from typing import Optional
import os, json

def try_export_sklearn_to_onnx(pipeline, sample_X, out_path: str = "artifacts/models/model.onnx") -> dict:
    """Eksporter ONNX (gated). Zwraca status i komunikat.
    Nie instaluje zależności — jeśli skl2onnx/onnxruntime brak, zwraca informację.
    """
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        return {"ok": False, "reason": "missing_deps", "message": "Brak skl2onnx/onnxruntime (gated).", "error": str(e)}

    try:
        n_features = sample_X.shape[1]
        onx = to_onnx(pipeline, sample_X[:1].astype("float32"), target_opset=15)
        with open(out_path, "wb") as f:
            f.write(onx.SerializeToString())
        return {"ok": True, "path": out_path}
    except Exception as e:
        return {"ok": False, "reason": "conversion_error", "error": str(e)}
