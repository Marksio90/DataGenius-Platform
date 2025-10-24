
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

def unify_feature_importance(pipeline, feature_names_in: List[str]) -> pd.DataFrame:
    """Mapuje importances modelu drzewiastego do kolumn wejściowych (agregacja OHE).
    Wspiera modele z atrybutem `feature_importances_`. Zwraca DataFrame z kolumnami: feature, importance (znormalizowane).
    """
    model = getattr(pipeline, 'named_steps', {}).get('model', pipeline)
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame({'feature': feature_names_in, 'importance': np.full(len(feature_names_in), 1/len(feature_names_in))})
    # spróbuj pozyskać nazwy cech po preprocesingu
    pre = getattr(pipeline, 'named_steps', {}).get('pre')
    try:
        out_names = pre.get_feature_names_out()
    except Exception:
        out_names = feature_names_in
    raw_imp = np.asarray(model.feature_importances_)
    raw_imp = raw_imp[: len(out_names)]
    # agreguj do bazowych kolumn (przed OHE)
    base = [n.split('__',1)[-1].split('_',1)[0] if '__' in n else n.split('_',1)[0] for n in out_names]
    agg: Dict[str, float] = {}
    for name, val in zip(base, raw_imp):
        agg[name] = agg.get(name, 0.0) + float(val)
    # normalizacja
    total = sum(agg.values()) or 1.0
    rows = [{'feature': k, 'importance': v/total} for k,v in agg.items()]
    df = pd.DataFrame(rows).sort_values('importance', ascending=False).reset_index(drop=True)
    return df
