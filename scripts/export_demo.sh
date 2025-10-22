#!/usr/bin/env bash
# TMIV – Export Demo Bundle
# Pakuje przykładowe artefakty i pliki repo do jednego ZIP-a.
# Użycie:
#   scripts/export_demo.sh                       # domyślna nazwa pliku w exports/
#   scripts/export_demo.sh -o exports/foo.zip    # własna nazwa
#   scripts/export_demo.sh -n run-demo-001       # niestandardowy run_id w MANIFEST.json
# Wymagania: bash, zip

set -euo pipefail

# ---------- opcje ----------
OUT_ZIP=""
RUN_ID="run-demo"
QUIET=0

while getopts ":o:n:q" opt; do
  case "${opt}" in
    o) OUT_ZIP="${OPTARG}" ;;
    n) RUN_ID="${OPTARG}" ;;
    q) QUIET=1 ;;
    \?) echo "Nieznana opcja: -$OPTARG" >&2; exit 2 ;;
    :)  echo "Opcja -$OPTARG wymaga argumentu." >&2; exit 2 ;;
  esac
done

# ---------- narzędzia ----------
cmd_exists() { command -v "$1" >/dev/null 2>&1; }
log() { [ "$QUIET" -eq 1 ] || echo -e "$*"; }

if ! cmd_exists zip; then
  echo "❌ Brak polecenia 'zip'. Zainstaluj (apt-get install zip / brew install zip) i spróbuj ponownie." >&2
  exit 1
fi

# ---------- lokalizacje ----------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
EXPORTS_DIR="$ROOT_DIR/exports"
CACHE_DIR="$ROOT_DIR/cache/artifacts/tmp_export"
STAMP="$(date +%Y%m%d-%H%M%S)"
GIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")"

mkdir -p "$EXPORTS_DIR" "$CACHE_DIR"

if [ -z "$OUT_ZIP" ]; then
  OUT_ZIP="$EXPORTS_DIR/tmiv-demo-${STAMP}.zip"
fi

STAGE_DIR="$(mktemp -d "$CACHE_DIR/stage_XXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT

log "📦 Tworzę paczkę w: $OUT_ZIP"
log "   Stage: $STAGE_DIR"

# ---------- zbieranie plików ----------
# dane demo
mkdir -p "$STAGE_DIR/data"
cp -f "$ROOT_DIR/data/avocado.csv" "$STAGE_DIR/data/" 2>/dev/null || true
[ -f "$ROOT_DIR/data/README.md" ] && cp -f "$ROOT_DIR/data/README.md" "$STAGE_DIR/data/"

# dokumentacja
mkdir -p "$STAGE_DIR/docs"
for f in README.md ARCHITECTURE.md UX_GUIDE.md API_INTERNALS.md; do
  [ -f "$ROOT_DIR/docs/$f" ] && cp -f "$ROOT_DIR/docs/$f" "$STAGE_DIR/docs/"
done

# konfiguracja
mkdir -p "$STAGE_DIR/config" "$STAGE_DIR/.streamlit"
[ -f "$ROOT_DIR/config/env.template" ] && cp -f "$ROOT_DIR/config/env.template" "$STAGE_DIR/config/"
[ -f "$ROOT_DIR/config/logging.yaml" ] && cp -f "$ROOT_DIR/config/logging.yaml" "$STAGE_DIR/config/"
[ -f "$ROOT_DIR/.streamlit/config.toml" ] && cp -f "$ROOT_DIR/.streamlit/config.toml" "$STAGE_DIR/.streamlit/"

# zależności / entrypoint
for f in app.py requirements.txt pyproject.toml environment.yml; do
  [ -f "$ROOT_DIR/$f" ] && cp -f "$ROOT_DIR/$f" "$STAGE_DIR/"
done

# frontend (bez ciężkich assetów, jeśli brak)
mkdir -p "$STAGE_DIR/frontend"
[ -f "$ROOT_DIR/frontend/styles.css" ] && cp -f "$ROOT_DIR/frontend/styles.css" "$STAGE_DIR/frontend/"
for f in ui_components.py ui_docs.py ui_panels.py ui_compare.py; do
  [ -f "$ROOT_DIR/frontend/$f" ] && cp -f "$ROOT_DIR/frontend/$f" "$STAGE_DIR/frontend/"
done

# backend – kluczowe wrappery (opcjonalnie, jeśli są w repo)
mkdir -p "$STAGE_DIR/backend"
for f in file_upload.py profiling_eda.py export_everything.py export_explain_pdf.py plots.py training_plan.py cache_manager.py utils.py; do
  [ -f "$ROOT_DIR/backend/$f" ] && cp -f "$ROOT_DIR/backend/$f" "$STAGE_DIR/backend/"
done

# manifest + README eksportu
cat > "$STAGE_DIR/MANIFEST.json" <<EOF
{
  "name": "tmiv-demo-bundle",
  "created_at": "${STAMP}",
  "run_id": "${RUN_ID}",
  "git_sha": "${GIT_SHA}",
  "contents": [
    "app.py",
    "requirements.txt / pyproject.toml / environment.yml",
    "docs/*",
    "config/*",
    ".streamlit/config.toml",
    "frontend/*",
    "backend/* (wybrane)",
    "data/avocado.csv"
  ]
}
EOF

cat > "$STAGE_DIR/README_EXPORT.txt" <<'EOF'
TMIV – Demo Export
==================

Zawartość:
- Minimalny zestaw plików do odpalenia aplikacji i prześledzenia przepływu.
- Dane demo: data/avocado.csv
- Dokumentacja: docs/*.md
- Konfiguracje: config/*, .streamlit/config.toml

Szybki start:
1) python -m pip install -r requirements.txt
2) streamlit run app.py
3) W UI: użyj "Użyj przykładowego" w sekcji "📊 Analiza Danych".

Uwagi:
- Bundle nie zawiera ciężkich modeli/artefaktów. Generuj je z poziomu UI.
- W przypadku braków pakietów opcjonalnych funkcje przełączą się na fallbacki.
EOF

# ---------- budowa ZIP ----------
( cd "$STAGE_DIR" && zip -r -q "$OUT_ZIP" . )
log "✅ Zapisano: $OUT_ZIP"
