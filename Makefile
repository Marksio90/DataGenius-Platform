.PHONY: setup run test lint format mypy export pdf clean help

help:
	@echo "Dostępne komendy:"
	@echo "  make setup    - Instalacja środowiska"
	@echo "  make run      - Uruchomienie aplikacji"
	@echo "  make test     - Uruchomienie testów"
	@echo "  make lint     - Sprawdzenie jakości kodu"
	@echo "  make format   - Formatowanie kodu"
	@echo "  make mypy     - Sprawdzenie typów"
	@echo "  make export   - Generowanie przykładowych artefaktów"
	@echo "  make pdf      - Generowanie przykładowego PDF"
	@echo "  make clean    - Czyszczenie plików tymczasowych"

setup:
	mamba env create -f environment.yml
	@echo "Aktywuj środowisko: mamba activate tmiv"

run:
	streamlit run app.py

test:
	pytest qa/tests/ -v --cov=backend --cov-report=html

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	black .
	isort .
	ruff check --fix .

mypy:
	mypy backend/ --ignore-missing-imports --warn-return-any

export:
	python -c "from backend.export_everything import create_sample_export; create_sample_export()"

pdf:
	python -c "from backend.export_explain_pdf import create_sample_pdf; create_sample_pdf()"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -rf artifacts/
	rm -rf outputs/