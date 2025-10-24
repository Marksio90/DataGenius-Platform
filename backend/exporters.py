# -*- coding: utf-8 -*-
import os, json
from typing import Optional
import numpy as np

def export_onnx(model, X_sample, path: str) -> Optional[str]:
    """Eksport do ONNX (sklearn/xgboost/lightgbm/catboost przez skl2onnx gdy dostępny)."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        n_features = X_sample.shape[1]
        initial_type = [('input', FloatTensorType([None, n_features]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open(path, 'wb') as f:
            f.write(onx.SerializeToString())
        return path
    except Exception as e:
        try:
            # xgboost direct
            import xgboost as xgb
            if hasattr(model, 'save_model'):
                model.save_model(path.replace('.onnx','.json'))
                return None
        except Exception:
            pass
    return None

def export_pmml(model, X_sample, path: str) -> Optional[str]:
    """Eksport do PMML (najczęściej wymaga nyoka/sklearn2pmml)."""
    try:
        from nyoka import skl_to_pmml
        # Minimalny export; bez cech i pipeline bywa ograniczony
        feature_names = [f'f{i}' for i in range(X_sample.shape[1])]
        skl_to_pmml(model, feature_names, 'target', path)
        return path
    except Exception:
        try:
            from sklearn2pmml import sklearn2pmml
            # W praktyce wymaga Pipeline i PMMLPipeline
            return None
        except Exception:
            pass
    return None

import os, json, csv
from pathlib import Path
def export_metrics_csv(metrics: dict, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True); p = outdir / "metrics.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in metrics.items(): w.writerow([k, v])
    return p
def export_plan_json(plan: dict, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True); p = outdir / "plan.json"
    p.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"); return p
def export_recs_json(recs: dict, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True); p = outdir / "recs.json"
    p.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8"); return p


def save_figure_png(fig, outdir: Path, name: str="plot.png", dpi: int=120) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    try:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    except Exception:
        pass
    return p
