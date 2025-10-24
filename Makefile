
.PHONY: setup run test lint fmt type export pdf clean
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
run:
	streamlit run app.py
test:
	pytest -q
lint:
	python -m pip install ruff black
	ruff check .
fmt:
	black .
type:
	python -m pip install mypy
	mypy backend ml
export:
	python scripts/export_zip.py
pdf:
	python scripts/make_pdf.py
clean:
	rm -rf artifacts registry __pycache__ .pytest_cache


a11y:
	@echo "A11y hints ready (WCAG contrast, aria-live hints in UI captions)."

onnx:
	python -c "from backend.onnx_export import try_export_sklearn_to_onnx; print('Use from app/CLI with sample X')"

hpo:
	@echo "HPO gated (Optuna) — do implementacji po włączeniu flag."

precommit:
	pre-commit install
	pre-commit run -a


demo-export:
	python cli/tmiv_cli.py --data data/avocado.csv --threshold cost --cost-fp 1.0 --cost-fn 2.0 --export

release:
	@echo "Tworzę pakiet release TMIV_2.0_Pro_Full.zip ..."
	python cli/tmiv_cli.py --data data/avocado.csv --threshold cost --cost-fp 1.0 --cost-fn 2.0 --export > /dev/null 2>&1 || true
	python scripts/save_plots.py > /dev/null 2>&1 || true
	python scripts/make_pdf.py > /dev/null 2>&1 || true
	python scripts/export_zip.py > /dev/null 2>&1 || true
	python - <<'PY'
import zipfile, os
z=zipfile.ZipFile('TMIV_2.0_Pro_Full.zip','w',zipfile.ZIP_DEFLATED)
for root,_,fs in os.walk('.'):
    for f in fs:
        p=os.path.join(root,f)
        if p.startswith('./.git') or p.startswith('./__pycache__'):
            continue
        z.write(p,p)
z.close()
print('TMIV_2.0_Pro_Full.zip')
PY
