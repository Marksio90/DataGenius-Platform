"""
feature_store.py
Docstring (PL): Bardzo prosty feature store plikowy — zapis/odczyt baseline (EDA + schemat) i metadanych.
"""
from __future__ import annotations
from typing import Dict, Any
import os, json, hashlib
from datetime import datetime

BASE_DIR = "artifacts/feature_store"

def _ensure():
    os.makedirs(BASE_DIR, exist_ok=True)

def save_baseline(payload: Dict[str, Any], name: str = "baseline") -> str:
    _ensure()
    path = os.path.join(BASE_DIR, f"{name}.json")
    payload = dict(payload)
    payload["saved_at"] = datetime.utcnow().isoformat()+"Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def load_baseline(name: str = "baseline") -> Dict[str, Any]:
    path = os.path.join(BASE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)