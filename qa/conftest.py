# qa/conftest.py
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

# -------------------------------
# Ustawienia globalne dla testów
# -------------------------------

def _find_project_root(start: Path | None = None) -> Path:
    """Znajdź root repo (szuka pyproject.toml / .git / app.py w górę)."""
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists() or (parent / "app.py").exists():
            return parent
    return p.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    # Skromna konfiguracja logów na czas testów
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@pytest.fixture(autouse=True, scope="session")
def _env_for_tests():
    """Bezpieczne ENV-y na czas testów."""
    os.environ.setdefault("ENV", "test")
    os.environ.setdefault("APP_NAME", "TMIV Advanced ML Platform (tests)")
    yield


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Deterministyczny seed w każdym teście."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    yield


@pytest.fixture(autouse=True, scope="session")
def _matplotlib_headless():
    """Matplotlib bez okna (Agg), gdyby testy generowały wykresy."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    yield


# -------------------------------
# Fikstury pomocnicze
# -------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Ścieżka do katalogu głównego repozytorium."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Katalog z danymi demonstracyjnymi."""
    return project_root / "data"


@pytest.fixture(scope="session")
def demo_csv_path(data_dir: Path) -> Path:
    """Ścieżka do przykładowego datasetu."""
    return data_dir / "avocado.csv"


@pytest.fixture(scope="session")
def demo_df(demo_csv_path: Path):
    """Załaduj przykładowy DataFrame (jeśli potrzebny w testach)."""
    import pandas as pd
    return pd.read_csv(demo_csv_path)


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    """
    Tymczasowy katalog artefaktów; przydatny gdy testy chcą coś zapisać.
    Zwraca Path, kasowany po teście.
    """
    p = tmp_path / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p
