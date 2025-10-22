# TMIV – ARCHITEKTURA (Skeleton)

**TMIV Advanced ML Platform** to warstwowa, modułowa aplikacja Streamlit pełniąca rolę „Auto-Data Scientist”. Projekt dzieli się na **Frontend (UI)**, **Core (domena i usługi)**, **Backend (narzędzia ML/EDA/wykresy/eksporty)** oraz **Infra (konfiguracja, logowanie, DB, CI/CD)**.

---

## 1) Widok wysoki (High-Level)

┌──────────┐ UI events ┌─────────────┐ kontrakty ┌───────────┐
│ Frontend │ ─────────────────▶ │ Core │ ─────────────────▶ │ Backend │
│ Streamlit│ ◀──────────────────│ Services │ ◀──────────────────│ Utils │
└──────────┘ artefakty └─────────────┘ adaptery └───────────┘
│ │
└─────────────── Logs/Telemetry/DB/Cache ────────────┘

markdown
Skopiuj kod

- **Frontend**: strony, panele, porównania; brak logiki biznesowej.
- **Core**: interfejsy (`Protocols`), DI container, usługi domenowe (data/eda/ml/explain/insights/export/cache/telemetry/db).
- **Backend**: implementacje techniczne (I/O, wykresy, PDF, pliki, plan treningu).
- **Infra**: `config/settings.py`, `config/logging.yaml`, `db/*`, `.github/workflows/*`.

---

## 2) Stos technologiczny

- **Język**: Python 3.11
- **UI**: Streamlit
- **ML**: scikit-learn (+ opcjonalnie XGBoost/LightGBM/CatBoost)
- **Wizualizacje**: Matplotlib (PNG; bez stylów globalnych)
- **Profiling EDA**: ydata-profiling / pandas-profiling *(opcjonalne)*
- **Konfiguracja**: Pydantic Settings v2
- **Cache/Artefakty**: lokalne ścieżki w `cache/artifacts/*`
- **PDF/ZIP**: ReportLab / zipfile
- **DB**: SQLite/DuckDB (opcjonalnie)
- **Telemetria**: OpenTelemetry (opcjonalnie) lub log-only
- **CI/CD**: GitHub Actions (CI + Release Zip)

---

## 3) Struktura repo (skrót)

tmiv-advanced-ml-platform/
├─ app.py
├─ core/ # DI + interfejsy + usługi domenowe
├─ backend/ # I/O, przygotowanie, wykresy, eksporty, plan treningu
├─ frontend/ # UI (komponenty, panele, dokumentacja w aplikacji)
├─ config/ # Settings + logging
├─ db/ # schema.sql + utils + migrations
├─ data/ # przykładowe dane (avocado.csv)
├─ qa/ # testy
└─ docs/ # README/ARCHITECTURE/UX_GUIDE/API_INTERNALS

yaml
Skopiuj kod

---

## 4) Kontrakty i warstwa domenowa

**Źródło prawdy**: `core/interfaces.py` (stabilne Protocols)

- `IDataService`, `IEDAService`, `IMLService`, `IExplainService`,
  `IInsightsService`, `ICacheService`, `ITelemetryService`, `IDBService`.

**DI**: `core/container.py` – rejestruje fabryki i singletony; w Streamlit trzymany w `st.session_state`.

---

## 5) Przepływy (Sequence)

### 5.1 Wczytanie i EDA
```mermaid
sequenceDiagram
  participant UI as Streamlit UI
  participant C as Core.DataService
  participant B as Backend.file_upload
  participant E as Core.EDAService
  UI->>C: load_any(file)
  C->>B: load_any_file(file)
  B-->>C: (df,name)
  C-->>UI: (df_norm,name)
  UI->>E: overview(df_norm)
  E-->>UI: stats + distributions + correlations
  UI->>E: profile_html(df_norm)
  E-->>UI: path/to/profile.html (opcjonalnie)
5.2 Planowanie i trening
mermaid
Skopiuj kod
sequenceDiagram
  participant UI
  participant M as Core.MLService
  participant TP as Backend.training_plan
  participant P as Backend.plots
  UI->>M: build_training_plan(df,target,opts)
  M->>TP: build_training_plan(...)
  TP-->>M: plan (problem_type, metrics, test_size,...)
  UI->>M: train_and_evaluate(df,target,plan)
  M-->>UI: result (leaderboard, models, FI, y_valid, metrics_by_model)
  UI->>P: plot_* (y_true,y_pred/proba,FI,leaderboard)
  P-->>UI: PNG paths
5.3 Eksport / PDF / ZIP
mermaid
Skopiuj kod
sequenceDiagram
  participant UI
  participant X as Core.ExplainService
  UI->>X: build_pdf_report(..., charts, metrics, FI)
  X-->>UI: path/to/report.pdf
  UI->>X: export_everything(run_id, artifacts...)
  X-->>UI: path/to/<run_id>.zip
6) Dane, stan i cache
Stateless UI – stan użytkownika w st.session_state (per sesja).

Cache artefaktów – cache/artifacts/<subdir>/<name> przez backend.cache_manager.cached_path.

Fingerprint danych – stabilny skrót struktury/preview (df_fingerprint) do kluczy cache.

Eksporty – exports/<run_id>.zip.

7) Przygotowanie danych i typ problemu
Sanityzacja: backend/dtype_sanitizer.py, runtime_preprocessor.py, auto_prep.py.

Heurystyki celu: backend/training_plan.py (detekcja problemu: classification/regression/timeseries).

Split:

klasyfikacja: train_test_split(..., stratify=y)

regresja: train_test_split(...)

series: TimeSeriesSplit / tail-split wg kolumny czasu

8) Metryki i ewaluacja (baseline)
Klasyfikacja: accuracy, f1, f1_weighted, roc_auc (OVR dla multi), logloss, APS.

Regresja: r2, rmse, mae.

Leaderboard: sort wg metryki głównej (kierunek zależny od metryki).

Feature Importance: drzewa (feature_importances_) lub modele liniowe (|coef| → normalizacja).

9) Wizualizacje
Matplotlib only (jedna figura = jeden wykres; bez stylów globalnych).

Klasyfikacja: ROC, PR, CM, kalibracja.

Regresja: rozkład błędów, pred_vs_true, residuals_vs_pred.

FI: bar chart (top-K).

Radar: normalizacja metryk (większe lepsze / mniejsze lepsze).

10) Telemetria i logowanie
Logowanie: config/logging.yaml (console/JSON/rotacja).

Telemetry: core/services/telemetry_service.py

OTel (jeśli dostępny) lub fallback log-only.

Heartbeat co 30s (tmiv.heartbeat).

Zdarzenia: artifact_saved, training_started/finished, itp.

11) Bezpieczeństwo i sekrety
Settings: config/settings.py (Pydantic v2, .env, ENV).

Sekrety: .streamlit/secrets.toml lokalnie; w CI – GitHub Secrets.

Szyfrowanie w sesji: opcjonalny Fernet key (TMIV_FERNET_KEY).

Zasady: brak sekretów w logach/eksportach; redakcja snapshotów konfigu.

12) Baza danych (opcjonalna)
DDL: db/schema.sql (SQLite/DuckDB).

Utils: db/db_utils.py (init_db, db_transaction, CRUD).

Serwis: core/services/db_service.py – zapis metryk/FI/leaderboardu per run_id.

13) Wydajność i skalowanie
Próbkowanie do profili EDA (wiersze + kolumny).

Pipeline: imputacja, OHE, skalowanie → wektorowe operacje.

Modele gradientowe (XGB/LGBM/CatBoost) opcjonalnie dla dużych danych.

Cache wyników i artefaktów per fingerprint.

Bezpieczne ograniczenia (max top-K, limity binów histogramów).

14) Rozszerzalność
Nowy provider LLM: podłącz w backend/ai_integration.py, kontrakt IInsightsService.

Nowe wykresy: dopisz w backend/plots.py i owiń w ExplainService.

Nowe modele: rozbuduj ml/pipelines/* lub _build_*_models() w MLService.

DB: zamień URL na Postgres (przez SQLAlchemy) w przyszłości; interfejs IDBService pozostaje stabilny.

Eksporty: dodawaj wpisy do ZIP w export_everything.py.

15) CI/CD i wersjonowanie
CI: .github/workflows/ci.yml – lint (ruff/black/isort), mypy (opcjonalnie), pytest (smoke).

Release: .github/workflows/release.yml – tag v* → paczka ZIP artefaktów repo.

SemVer:

MAJOR: zmiana kontraktów z core/interfaces.py.

MINOR: nowe funkcje/fabryki DI; kompatybilne zmiany.

PATCH: poprawki, wydajność, bugfix.

16) Tryby uruchomienia
Dev: ENV=development (szersze logi, profilowanie opcjonalne).

Prod: ENV=production (logi INFO/ERROR, telemetria włączona, bez profilu EDA na pełnych danych).

17) Runbook (skrót)
python -m pip install -r requirements.txt

Ustaw .env z APP_NAME i ewentualnymi kluczami (OpenAI/…).

(Opcj.) python -c "from db.db_utils import init_db; init_db()"

streamlit run app.py

W UI: Analiza → Trening → Wyniki → Rekomendacje → Eksporty

18) Granice i decyzje architektoniczne
Brak twardej zależności na PyCaret – baseline własny (sklearn).

Wykresy: Matplotlib → prostota i kontrola (konsekwentny output PNG).

Soft-importy na ciężkie pakiety (XGB/LGBM/Cat, profiling, OTel).

DI umożliwia zamianę implementacji bez zmian w UI.

Brak globalnych singletonów modeli; artefakty i cache są jawne.

Ten dokument jest „żywy” — aktualizujemy go przy istotnych zmianach kontraktów lub przepływów.