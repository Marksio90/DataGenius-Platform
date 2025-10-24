.PHONY: setup run lint format mypy test

setup:
	python -m pip install -r requirements.txt

run:
	streamlit run app.py

lint:
	ruff check . || true

format:
	black .

mypy:
	mypy --ignore-missing-imports . || true

test:
	pytest -q || true

cli-train:
\tpython -m cli.tmiv_cli --data data/avocado.csv --export
