"""
registry utils (Stage 8)
Docstring (PL): Rejestr modeli — zapis .joblib i manifestu, opcjonalny eksport ONNX.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import os, json, hashlib
from datetime import datetime
import joblib

REG_DIR = "registry"
MAN_DIR = "registry/manifests"
MOD_DIR = "registry/models"

def _ensure():
    os.makedirs(MAN_DIR, exist_ok=True)
    os.makedirs(MOD_DIR, exist_ok=True)

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def register_model(name: str, version: str, metrics: Dict[str, float], paths: Dict[str,str], extra: Dict[str, Any] = None) -> str:
    _ensure()
    man = {
        "name": name,
        "version": version,
        "metrics": metrics,
        "paths": paths,
        "created": datetime.utcnow().isoformat()+"Z",
    }
    if extra: man["extra"] = extra
    data = json.dumps(man, ensure_ascii=False, indent=2).encode("utf-8")
    path = f"{MAN_DIR}/{name}-{version}.json"
    with open(path, "wb") as f:
        f.write(data)
    return path

def promote_pipeline(name: str, version: str, pipe, X_sample, metrics: Dict[str, float], export_onnx: bool = False) -> Dict[str, Any]:
    _ensure()
    job_path = f"{MOD_DIR}/{name}-{version}.joblib"
    joblib.dump(pipe, job_path)
    paths = {"joblib": job_path}
    if export_onnx:
        try:
            from backend.onnx_export import export_onnx
            ox = export_onnx(pipe, X_sample, f"{MOD_DIR}/{name}-{version}.onnx")
            if ox.get("status")=="OK":
                paths["onnx"] = ox["path"]
        except Exception:
            pass
    man_path = register_model(name, version, metrics, paths)
    return {"manifest": man_path, "paths": paths}