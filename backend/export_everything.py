"""
export_everything.py
Docstring (PL): Buduje ZIP artefaktów: plan/metryki/EDA/konfiguracja + manifest z hashami.
Wersja Stage 4 – bez serializacji modeli (dojdzie później); skupiamy się na metadanych i raportach.
"""
from __future__ import annotations
from typing import Dict, Any
import os, json, hashlib, zipfile, io
from datetime import datetime

def _sha256_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def export_zip(out_path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Zbuduj w pamięci pliki JSON
    files = {
        "plan.json": json.dumps(payload.get("plan", {}), ensure_ascii=False, indent=2).encode("utf-8"),
        "results.json": json.dumps(payload.get("results", {}), ensure_ascii=False, indent=2).encode("utf-8"),
        "eda.json": json.dumps(payload.get("eda", {}), ensure_ascii=False, indent=2).encode("utf-8"),
        "config.json": json.dumps(payload.get("config", {}), ensure_ascii=False, indent=2).encode("utf-8"),
        "README.txt": ("TMIV export package\nGenerated: " + datetime.utcnow().isoformat() + "Z\n").encode("utf-8"),
    }
    manifest = {name: _sha256_bytes(content) for name, content in files.items()}

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
        z.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    return out_path