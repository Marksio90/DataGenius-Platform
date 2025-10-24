
from __future__ import annotations
import os, json, hashlib, datetime
import pandas as pd
from typing import List

def _ensure(p:str)->None: os.makedirs(p, exist_ok=True)

def _hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(df.columns)).encode())
    h.update(str(df.shape).encode())
    return h.hexdigest()

def save_features(df: pd.DataFrame, cols: List[str], name: str = "features"):
    fs_dir = "artifacts/feature_store"
    _ensure(fs_dir)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    subset = df[cols].copy()
    path = os.path.join(fs_dir, f"{name}-{ts}.parquet")
    try:
        subset.to_parquet(path)
    except Exception:
        # Fallback to csv if parquet engine not present
        path = os.path.join(fs_dir, f"{name}-{ts}.csv")
        subset.to_csv(path, index=False)
    manifest = {"name": name, "created_utc": ts, "columns": cols, "path": path}
    with open(os.path.join(fs_dir, f"{name}-{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
