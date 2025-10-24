
from __future__ import annotations
import os, json, hashlib, datetime
from typing import Dict, Any

def _ensure_dir(p: str)->None:
    os.makedirs(p, exist_ok=True)

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def _write_json(path: str, data: Dict[str, Any]) -> str:
    _ensure_dir(os.path.dirname(path))
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    with open(path, "wb") as f:
        f.write(blob)
    return _sha256_bytes(blob)

def save_manifest(name: str, payload: Dict[str, Any], version: str = "1.0.0") -> str:
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    manifest = {"name": name, "version": version, "created_utc": ts, "payload": payload}
    path = os.path.join("registry", "manifests", f"{name}.json")
    sha = _write_json(path, manifest)
    index_path = os.path.join("registry", "manifests", "index.json")
    idx = {"updated_utc": ts, "items": {}}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
        except Exception:
            pass
    idx["updated_utc"] = ts
    idx["items"][name] = {"path": path, "sha256": sha, "version": version}
    _write_json(index_path, idx)
    return path
