from __future__ import annotations
import os, json
from datetime import datetime
from backend.export_everything import export_zip
from backend.export_explain_pdf import build_pdf

def test_export_zip_and_pdf(tmp_path):
    payload = {
        "plan": {"target":"AveragePrice","problem_type":"regression"},
        "results": {"ridge":{"status":"ok","metrics":{"RMSE":1.0,"MAE":0.8,"R2":0.1}}},
        "eda": {"rows":4,"cols":13},
        "config": {"strategy":"balanced","fingerprint":"abc123"}
    }
    out_zip = tmp_path / "tmiv_export.zip"
    path = export_zip(str(out_zip), payload)
    assert os.path.exists(path)

    ctx = {
        "fingerprint": "abc123",
        "strategy": "balanced",
        "problem_type": "regression",
        "target": "AveragePrice",
        "validation": {"cv":{"name":"KFold","n_splits":5}},
        "results": payload["results"],
        "recommendations": ["Test rekomendacji"],
        "seed": 42,
        "generated_at": datetime.utcnow().isoformat()+"Z",
    }
    out_pdf = tmp_path / "tmiv_report.pdf"
    pdf_path = build_pdf(str(out_pdf), ctx)
    assert os.path.exists(pdf_path)