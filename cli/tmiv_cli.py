
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
import pandas as pd
from ml.training import train_and_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from backend.thresholding import optimize_threshold_by_cost, optimize_threshold_by_youden, metrics_at_threshold
from backend.registry import save_manifest

def main():
    ap = argparse.ArgumentParser(description="TMIV CLI – train & export")
    ap.add_argument("--data", required=True, help="Ścieżka do CSV/Parquet.")
    ap.add_argument("--export", action="store_true", help="Eksport ZIP/PDF po treningu.")
    ap.add_argument("--threshold", choices=["none","youden","cost"], default="none", help="Strategia progu decyzyjnego (klasyfikacja).")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="Koszt FP (dla --threshold cost).")
    ap.add_argument("--cost-fn", type=float, default=1.0, help="Koszt FN (dla --threshold cost).")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print("Brak pliku danych.", file=sys.stderr); raise SystemExit(2)

    # Gated Polars reader
    import os
    use_polars = os.environ.get("USE_POLARS", "false").lower() in {"1","true","yes"}
    if use_polars:
        try:
            import polars as pl
            df = pl.read_parquet(args.data).to_pandas() if args.data.lower().endswith(".parquet") else pl.read_csv(args.data).to_pandas()
        except Exception:
            df = pd.read_parquet(args.data) if args.data.lower().endswith(".parquet") else pd.read_csv(args.data)
    else:
        df = pd.read_parquet(args.data) if args.data.lower().endswith(".parquet") else pd.read_csv(args.data)

    res = train_and_eval(df)
    os.makedirs("artifacts/models", exist_ok=True)
    with open("artifacts/models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(res["metrics"], f, ensure_ascii=False, indent=2)
    print("OK: trening zakończony")

    # Threshold
    if res["metrics"]["task"] == "classification" and args.threshold != "none":
        X = df.drop(columns=[res["target"]]); y = df[res["target"]]
        _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()==2 else None)
        proba = res["model"].predict_proba(Xte)[:,1]
        import json
        os.makedirs("artifacts/models", exist_ok=True)
        json.dump({"y_true": yte.tolist(), "y_prob": proba.tolist()}, open("artifacts/models/preds.json","w",encoding="utf-8"))
        fpr, tpr, thr = roc_curve(yte, proba)
        if args.threshold == "youden":
            t = optimize_threshold_by_youden(fpr, tpr, thr)
            objective = "youden"
            payload_extra = {}
        else:
            out = optimize_threshold_by_cost(thr, tpr, fpr, {"FP": args.cost_fp, "FN": args.cost_fn})
            t = out["threshold"]; objective = "cost"; payload_extra = {"expected_cost": out["expected_cost"]}
        m = metrics_at_threshold(yte.to_numpy(), proba, t)
        payload = {"threshold": float(t), "objective": objective, "metrics_at_threshold": m} | payload_extra
        path = save_manifest("threshold", payload, version="1.0.0")
        print(f"Zapisano manifest progu: {path}")

    if args.export:
        os.system("python scripts/save_plots.py > /dev/null 2>&1")
        os.system("python scripts/make_pdf.py > /dev/null 2>&1")
        os.system("python scripts/export_zip.py > /dev/null 2>&1")
        print("OK: eksport ZIP/PDF gotowy")

if __name__ == "__main__":
    main()
