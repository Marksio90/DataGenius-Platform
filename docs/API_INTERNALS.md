# TMIV – API INTERNALS (Skeleton)

> Ten dokument opisuje **wewnętrzne** kontrakty i punkty integracji TMIV Advanced ML Platform.
> Służy zespołom Z1–Z7 do bezpiecznego rozwijania modułów bez psucia kompatybilności.

---

## Spis treści
1. [Przegląd architektury](#przegląd-architektury)
2. [DI Container i rejestr usług](#di-container-i-rejestr-usług)
3. [Kontrakty `core/interfaces.py`](#kontrakty-coreinterfacespy)
4. [Usługi domenowe (`core/services/*`)](#usługi-domenowe-coreservices)
5. [Backend (`backend/*`) – API publiczne](#backend-backend--api-publiczne)
6. [Warstwa danych i DB](#warstwa-danych-i-db)
7. [Artefakty, cache i eksporty](#artefakty-cache-i-eksporty)
8. [Telemetria i logowanie](#telemetria-i-logowanie)
9. [Obsługa błędów](#obsługa-błędów)
10. [Cykl życia: od danych do eksportu](#cykl-życia-od-danych-do-eksportu)
11. [Konwencje, wersjonowanie, kompatybilność](#konwencje-wersjonowanie-kompatybilność)

---

## Przegląd architektury

- **Frontend (Streamlit)**: `app.py`, `frontend/*` – UI, panele, porównania.
- **Backend**: `backend/*` – ładowanie/EDA, przygotowanie, modele, wykresy, eksporty.
- **Core (domena)**: `core/*` – interfejsy, DI container, usługi.
- **Config**: `config/*` – Settings, logging.
- **DB (opcjonalna)**: `db/*` – schemat, utils.
- **QA/CI**: `qa/*`, `.github/workflows/*`.

Komunikacja przepływa: **UI → core/services → backend** (zależności miękkie, możliwe fallbacki).

---

## DI Container i rejestr usług

Plik: `core/container.py`

- Singleton (process / sesja Streamlit): `get_container()`.
- Klucze fabryk (domyślne):
  - `settings`, `telemetry`, `security`, `cache`
  - `data`, `eda`, `ml`, `explain`, `export`, `insights`, `db` *(opcjonalne, lazy import)*
- API:
  ```python
  c = get_container()
  settings = c.resolve("settings")
  ml = c.resolve("ml")
  export = c.resolve_or_none("export")
 c = get_container()
  settings = c.resolve("settings")
  ml = c.resolve("ml")
  export = c.resolve_or_none("export")
  explain = c.resolve_or_none("explain")
  eda = c.resolve("eda")
  data = c.resolve("data")
  insights = c.resolve_or_none("insights")
  db = c.resolve_or_none("db")

  # podgląd rejestru (bez sekretów)
  c.snapshot()
Kontrakty core/interfaces.py
Interfejsy (Protocols) – stabilne kontrakty między warstwami.

IDataService

python
Skopiuj kod
load_any(src) -> tuple[pd.DataFrame, str]
load_example() -> pd.DataFrame
normalize_columns(df) -> pd.DataFrame
fingerprint(df) -> str
profile_html(df, *, title="...") -> str | None
IEDAService

python
Skopiuj kod
overview(df) -> dict
correlations(df) -> pd.DataFrame
distributions(df) -> dict
profile_html(df) -> str | None
IMLService

python
Skopiuj kod
build_training_plan(df, target, *, strategy=None, hints=None, random_state=42, n_jobs=-1) -> dict
train_and_evaluate(df, target, *, plan=None, test_size=0.2, random_state=42,
                   cv_folds=3, enable_ensembles=True, n_jobs=-1) -> TrainResult(dict)
IExplainService

python
Skopiuj kod
plot_classification_curves(y_true, y_proba, *, model_name="model") -> dict[name->png]
plot_confusion_matrix(y_true, y_pred, *, class_names=None, model_name="model") -> str
plot_calibration_curve(y_true, y_proba, *, model_name="model", n_bins=10) -> str
plot_regression_diagnostics(y_true, y_pred, *, model_name="model") -> dict
plot_feature_importance(fi, *, top_k=20, name="fi") -> str
plot_radar_leaderboard(leaderboard, *, metrics=None, name="radar") -> str
build_pdf_report(output_path, **kwargs) -> str
export_everything(run_id, **kwargs) -> str
IInsightsService

python
Skopiuj kod
describe_columns(df) -> dict[col -> opis]
generate_recommendations(context: dict) -> list[str]
ICacheService

python
Skopiuj kod
cached_path(subdir, name) -> Path
cache_result(namespace="default", ttl=None) -> decorator
df_fingerprint(df) -> str
ITelemetryService

python
Skopiuj kod
start_span(name, attrs=None) -> contextmanager
add_event(name, attrs=None) -> None
incr_counter(name, value=1.0, attrs=None) -> None
observe_value(name, value, attrs=None) -> None
start_heartbeat(interval_sec=None) -> None
stop_heartbeat() -> None
snapshot() -> dict
IDBService (opcjonalnie)

python
Skopiuj kod
save_dataset_meta(name, fingerprint, rows, cols) -> bool
save_metrics(run_id, model_name, metrics: dict) -> bool
save_feature_importance(run_id, df: pd.DataFrame|seq) -> bool
save_leaderboard(run_id, df: pd.DataFrame) -> bool
get_run(run_id) -> dict
Usługi domenowe (core/services/*)
DataService – adapter na backend.file_upload (+ normalizacja kolumn, fingerprint, profil EDA).

EDAService – overview/correlations/distributions; profil EDA przez backend.profiling_eda.

MLService – plan + baseline trening (sklearn + opcjonalnie XGB/LGBM/Cat); zwraca leaderboard, modele, FI itd.

ExplainService – wykresy (backend/plots.py), PDF (backend/export_explain_pdf.py), ZIP (backend/export_everything.py).

ExportService – zapisy JSON/CSV/TXT/model + PDF/ZIP (fallbacki).

InsightsService – opisy i rekomendacje via backend.ai_integration (jeśli jest) z heurystyką.

CacheService – fasada nad backend.cache_manager z in-memory fallbackiem.

TelemetryService – OpenTelemetry jeśli dostępny, inaczej log-only.

Backend (backend/*) – API publiczne
file_upload.py: load_any_file, load_example_dataset, normalize_columns

profiling_eda.py: build_profile_html(df, **opts) -> str | None

training_plan.py: build_training_plan(df, target, **opts) -> dict

plots.py: plot_* (ROC/PR/CM/kalibracja/regresja/FI/radar) → ścieżki PNG

export_explain_pdf.py: build_pdf_report(output_path, **kwargs) -> str

export_everything.py: export_everything(run_id, **kwargs) -> str

ai_integration.py (opc.): describe_columns(df), generate_recommendations(ctx)

Warstwa danych i DB
DDL: db/schema.sql (SQLite/DuckDB).

Utils: db/db_utils.py (init_db, get_connection, db_transaction, CRUD).

Serwis: core/services/db_service.py – zapis runów/metryk/FI/leaderboard.

URL-e DB (Settings.database_url)
None → sqlite:///tmiv.db • sqlite:///:memory: • sqlite:///path.db • duckdb:///path.duckdb

Artefakty, cache i eksporty
Struktura:

bash
Skopiuj kod
cache/artifacts/
  profiles/   plots/   reports/   tmp_export/   models/
exports/
  <run_id>.zip
API: backend.cache_manager.cached_path(subdir, name) -> Path, df_fingerprint(df) -> str.

Telemetria i logowanie
Konfiguracja logów: config/logging.yaml.

Telemetria: TelemetryService (OTel lub fallback).
Przykład:

python
Skopiuj kod
tele = c.resolve("telemetry")
with tele.start_span("training", {"target": target, "rows": len(df)}):
    tele.add_event("train.begin")
    ...
    tele.add_event("train.end")
Obsługa błędów
W backend/* rzucaj krótkie, czytelne wyjątki (ValueError, IOError, RuntimeError).

W core/services/* stosuj łagodne fallbacki (np. brak pakietu → None / pusta kolekcja).

Nie loguj/eksportuj sekretów; komunikaty user-friendly mapowane przez backend/error_handler.py (jeśli używany).

Cykl życia: od danych do eksportu
python
Skopiuj kod
c = get_container()
data = c.resolve("data")
eda = c.resolve("eda")
ml = c.resolve("ml")
explain = c.resolve("explain")
export = c.resolve("export")

df, name = data.load_any("data/avocado.csv")
plan = ml.build_training_plan(df, target="average_price")
result = ml.train_and_evaluate(df, target="average_price", plan=plan)

# wykresy
charts = {}
if result["problem_type"] == "classification":
    # przykładowo...
    pass
else:
    preds = result["models"][result["best_model_name"]].predict(df.drop(columns=[result["target"]]))
    charts.update(explain.plot_regression_diagnostics(df[result["target"]].to_numpy(), preds))

# PDF + ZIP
pdf = explain.build_pdf_report("tmiv_report.pdf",
                               metrics=result["metrics_by_model"].get(result["best_model_name"], {}),
                               feature_importance=result["feature_importance"],
                               charts=charts,
                               title="TMIV – Explainability")
bundle = explain.export_everything(
    "run-001",
    problem_type=result["problem_type"],
    metrics=result["metrics_by_model"].get(result["best_model_name"], {}),
    leaderboard=result["leaderboard"],
    feature_importance=result["feature_importance"],
    plots=charts,
)
Konwencje, wersjonowanie, kompatybilność
SemVer:

MAJOR – zmiany kontraktów w core/interfaces.py.

MINOR – nowe funkcje/fabryki DI, zgodne wstecz.

PATCH – poprawki i optymalizacje.

Publiczne API warstw: interfejsy + core/services/* + sygnatury funkcji backend/*.

Importy ciężkich pakietów → soft-import (łagodne fallbacki).







