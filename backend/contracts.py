
from __future__ import annotations
import os, json, hashlib, datetime
import pandas as pd
import pandera as pa
from typing import Dict, Any

def _ensure_dir(p:str)->None:
    os.makedirs(p, exist_ok=True)

def schema_from_df(df: pd.DataFrame) -> pa.DataFrameSchema:
    cols = {}
    for c in df.columns:
        dtype = df[c].dtype
        if pd.api.types.is_integer_dtype(dtype):
            col = pa.Column(int, nullable=df[c].isna().any())
        elif pd.api.types.is_float_dtype(dtype):
            col = pa.Column(float, nullable=df[c].isna().any())
        elif pd.api.types.is_bool_dtype(dtype):
            col = pa.Column(bool, nullable=df[c].isna().any())
        else:
            col = pa.Column(str, nullable=True)
        cols[c] = col
    return pa.DataFrameSchema(cols, coerce=False)

def save_contract(df: pd.DataFrame, run_id: str, base_dir: str = "artifacts/reports/baselines"):
    schema = schema_from_df(df)
    js = schema.to_json()
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "contract.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(js)
    return path

def validate_with_contract(df: pd.DataFrame, contract_path: str) -> Dict[str, Any]:
    schema = pa.DataFrameSchema.from_json(contract_path)
    try:
        schema.validate(df, lazy=True)
        return {"ok": True, "violations": []}
    except pa.errors.SchemaErrors as e:
        violations = [str(err) for err in e.failure_cases.itertuples(index=False)]
        return {"ok": False, "violations": violations}
