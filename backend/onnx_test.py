
from __future__ import annotations
from typing import Dict, Any
import numpy as np

def try_compare_onnx(pipeline, X_sample) -> Dict[str, Any]:
    """Porównuje predykcje sklearn vs onnxruntime (gated). Zwraca różnicę L1/MAE.
    Jeśli brak onnxruntime/skl2onnx – informuje o braku.
    """
    try:
        import onnxruntime as ort
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        return {"ok": False, "reason": "missing_deps", "message": str(e)}
    try:
        onx = to_onnx(pipeline, X_sample[:1].astype("float32"), target_opset=15)
        sess = ort.InferenceSession(onx.SerializeToString())
        inp_name = sess.get_inputs()[0].name
        y_sklearn = pipeline.predict_proba(X_sample)[:,1]
        y_onnx = sess.run(None, {inp_name: X_sample.astype("float32")})[1].ravel() if len(sess.get_outputs())>1 else sess.run(None, {inp_name: X_sample.astype("float32")})[0].ravel()
        diff = float(np.mean(np.abs(y_sklearn - y_onnx)))
        return {"ok": True, "mae": diff, "n": int(X_sample.shape[0])}
    except Exception as e:
        return {"ok": False, "reason": "compare_error", "message": str(e)}
