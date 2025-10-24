
from __future__ import annotations
from typing import Dict, Any

def log_run_metrics(task: str, metrics: Dict[str, Any]) -> dict:
    try:
        import mlflow
        with mlflow.start_run(run_name=f"tmiv-{task}"):
            for k,v in metrics.items():
                try: mlflow.log_metric(k, float(v))
                except Exception: pass
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "reason": "missing_mlflow", "message": str(e)}
