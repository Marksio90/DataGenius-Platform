
from __future__ import annotations
import os, json, hashlib, time, contextlib
from typing import Dict

def sha256_of_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(path: str)->None:
    os.makedirs(path, exist_ok=True)

@contextlib.contextmanager
def time_block(metrics: Dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = round(time.perf_counter() - t0, 6)
