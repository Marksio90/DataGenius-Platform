
from __future__ import annotations
import json, hashlib, os, datetime
from typing import Dict, Any
from .types import DataQualityBaseline

def _ensure_dir(path: str)->None:
    os.makedirs(path, exist_ok=True)

def _hash_schema(dtypes: Dict[str,str])->str:
    blob = json.dumps(dtypes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def save_quality_baseline(df, run_id: str, base_dir: str = "artifacts/reports/baselines"):
    dtypes = {c: str(t) for c,t in df.dtypes.items()}
    missingness = {c: float(df[c].isna().mean()) for c in df.columns}
    cardinality = {c: int(getattr(df[c], "nunique")()) for c in df.columns}
    schema_hash = _hash_schema(dtypes)
    baseline = DataQualityBaseline(run_id=run_id, schema_hash=schema_hash,
                                   missingness=missingness, cardinality=cardinality, dtypes=dtypes)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "quality.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baseline.model_dump(), f, ensure_ascii=False, indent=2)
    return out_path
