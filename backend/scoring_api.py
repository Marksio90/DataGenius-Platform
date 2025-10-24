# -*- coding: utf-8 -*-
import os, json
from typing import List, Any, Dict
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
try:
    from joblib import load as joblib_load
except Exception:
    import pickle
    def joblib_load(p): 
        with open(p, 'rb') as f: 
            return pickle.load(f)

from backend.runtime_preprocessor import apply_plan

ARTIFACTS_DIR = os.path.join('artifacts','last_run')
MODEL_PATHS = ['voting.joblib','stacking.joblib','model.joblib','voting.pkl','stacking.pkl','model.pkl']

def _load_plan():
    with open(os.path.join(ARTIFACTS_DIR,'plan.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def _load_model():
    for name in MODEL_PATHS:
        p = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(p):
            return joblib_load(p)
    raise FileNotFoundError("Brak modelu w artifacts/last_run")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

app = FastAPI(title="TMIV Scoring API")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    model = _load_model()
    plan = _load_plan()
    df_raw = pd.DataFrame(req.records)
    target = plan.get('target') or 'target'
    df_prep = apply_plan(df_raw, target=target, plan=plan)
    X = df_prep.drop(columns=[c for c in df_prep.columns if c==target])
    preds = model.predict(X)
    return {"predictions": [float(x) if hasattr(x,'item') else x for x in preds]}
