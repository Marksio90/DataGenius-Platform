
# TMIV – The Most Important Variables v2.0 Pro

**Opis (PL):** Kompleksowa, automatyczna aplikacja Streamlit do analizy najważniejszych cech w zbiorach danych.
Interfejs prowadzi: **Upload → EDA → Plan → Trening (async) → Wyniki → Eksport (ZIP/PDF) → Rekomendacje**.

- UI/README po polsku; kod/identyfikatory po angielsku.
- Kompatybilność: Python 3.11; Streamlit ≥1.37; pandas ≥2.2; numpy ≥1.26; scikit-learn ≥1.5.
- Zakaz użycia `OneHotEncoder(sparse=...)` — tylko `sparse_output` (test TMIV-ML-ENC-001).

## Uruchomienie
```bash
mamba env create -f environment.yml
mamba activate tmiv
make setup
streamlit run app.py
```


## A11y & Skróty
- Kontrast zgodny z WCAG (motyw dark), treści informacyjne w UI mają charakter opisowy (aria-live w komunikatach Streamlit).
- Skróty (umowne): **u** – uruchom automat, **g** – wyniki, **s** – eksport, **/** – szukaj, **?** – pomoc, **Esc** – zamknij.


## Runbook (PL)
### Szybki start
```bash
mamba env create -f environment.yml
mamba activate tmiv
make setup
streamlit run app.py
```
### Tryb demo (CLI, pełny eksport)
```bash
make demo-export
# ZIP: artifacts/exports/tmiv_export.zip
# PDF: artifacts/reports/report.pdf
```
### Skróty klawiszowe
`u` – automat, `g` – wyniki, `s` – eksport, `/` – szukaj, `?` – pomoc, `Esc` – zamknij.

## Troubleshooting (TMIV-XXX)
- **TMIV-ML-ENC-001** – Wykryto `sparse=` w OneHotEncoder. Używaj wyłącznie `sparse_output`. Wersja scikit-learn ≥1.5.
- **TMIV-ML-TIMEOUT** – Przekroczono `max_train_time_sec`. Zmniejsz złożoność modelu lub liczbę danych.
- **TMIV-EDA-PROFILE-TO** – Profil trwa zbyt długo. Włącz large-mode lub ogranicz kolumny.
- **TMIV-SYS-OOM-WARN** – Niski budżet pamięci. Użyj Parquet, włącz Polars (`USE_POLARS=true`), ogranicz top-N.
- **Gated deps** – SBOM/ONNX/MLflow/GE wymagają dodatkowych pakietów.

## Security (DLP/Keys)
- Klucze tylko w ENV/.env/.streamlit/secrets.toml. Zero logowania kluczy.
- PII Guard: podgląd maski i brak zapisu oryginalnych wartości w logach.

## Release checklist
- Testy ≥ 80% (krytyczne) — minimalny zestaw pokazowy przechodzi.
- Brak `sparse=` w encoderze.
- ZIP+PDF+manifest SHA.
- CI matrix OK.
