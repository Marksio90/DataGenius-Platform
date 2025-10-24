"""
tmiv_cli.py — Headless CLI dla TMIV
Użycie:
  python -m cli.tmiv_cli --data data/avocado.csv --export
  # lub po zainstalowaniu wejścia: tmiv train --data data/avocado.csv --export
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime

from backend import file_upload, dtype_sanitizer, runtime_preprocessor, eda_integration, training_plan, async_ml_trainer
from backend.export_everything import export_zip
from backend.export_explain_pdf import build_pdf
from backend.ai_integration import deterministic_recommendations

def main():
    ap = argparse.ArgumentParser(description="TMIV – CLI")
    ap.add_argument("--data", required=True, help="Ścieżka do pliku danych (CSV/XLSX/Parquet/JSON)")
    ap.add_argument("--strategy", default="balanced", choices=["fast_small","balanced","accurate","advanced"])
    ap.add_argument("--export", action="store_true", help="Zapisz ZIP i PDF artefaktów")
    args = ap.parse_args()

    df, rep = file_upload.load_from_path(args.data)
    if rep.get("code") != "OK":
        raise SystemExit(f"Błąd wczytywania: {rep}")

    df2, _ = dtype_sanitizer.sanitize_dtypes(df)
    df3, meta = runtime_preprocessor.preprocess_runtime(df2)
    eda = eda_integration.quick_eda(df3)

    plan = training_plan.build_training_plan(df3, strategy=args.strategy)
    if plan.get("status") != "OK":
        raise SystemExit(f"Błąd planu: {plan}")

    res = async_ml_trainer.train_async(
        df=df3, target=plan["target"], problem_type=plan["problem_type"],
        strategy=args.strategy, max_parallel=2, max_time_sec=120, random_state=42
    )

    print(json.dumps({"plan": plan, "result": res}, indent=2, ensure_ascii=False))

    if args.export:
        os.makedirs("artifacts/exports", exist_ok=True)
        os.makedirs("artifacts/reports", exist_ok=True)

        payload = {
            "plan": plan,
            "results": res["results"],
            "eda": eda,
            "config": {
                "strategy": plan.get("strategy"),
                "problem_type": plan.get("problem_type"),
                "target": plan.get("target"),
                "fingerprint": meta.get("fingerprint"),
            },
        }
        export_zip("artifacts/exports/tmiv_export.zip", payload)

        best = res.get("best_model")
        best_metrics = res["results"].get(best, {}).get("metrics", {}) if best else {}
        rec = deterministic_recommendations(plan["problem_type"], best or "-", best_metrics)
        ctx = {
            "fingerprint": meta.get("fingerprint"),
            "strategy": plan.get("strategy"),
            "problem_type": plan.get("problem_type"),
            "target": plan.get("target"),
            "validation": plan.get("validation"),
            "results": res["results"],
            "recommendations": rec,
            "seed": 42,
            "generated_at": datetime.utcnow().isoformat()+"Z",
        }
        build_pdf("artifacts/reports/tmiv_report.pdf", ctx)

if __name__ == "__main__":
    main()