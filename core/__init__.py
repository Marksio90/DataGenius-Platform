from __future__ import annotations
# -*- coding: utf-8 -*-
"""Core package marker.

Pozwala na importy w stylu `from core ...`.
Eksporuje bezpiecznie wersję pakietu i wycisza ostrzeżenia logowania.
"""

from typing import Final, Optional, Iterable
import os
import re

# -- ciche logowanie: brak "No handler found" gdy aplikacja nie skonfigurowała logów
import logging as _logging

_logger = _logging.getLogger(__name__)
# Dodaj NullHandler tylko jeśli brak innych handlerów (unikniemy duplikacji)
if not _logger.handlers:
    _logger.addHandler(_logging.NullHandler())

# -- wersja pakietu: importlib.metadata (py>=3.8) z fallbackami
try:  # stdlib (Py 3.8+)
    from importlib.metadata import (
        version as _im_version,
        PackageNotFoundError as _IMPkgNotFound,  # type: ignore[attr-defined]
    )
except Exception:  # pragma: no cover - fallback na backport
    try:
        from importlib_metadata import (  # type: ignore
            version as _im_version,
            PackageNotFoundError as _IMPkgNotFound,
        )
    except Exception:  # ostateczny fallback – brak importlib metadata
        _im_version = None  # type: ignore
        class _IMPkgNotFound(Exception):  # type: ignore
            pass

# Opcjonalny dodatkowy fallback: pkg_resources (cięższe, więc używane na końcu)
try:
    import pkg_resources as _pkg_resources  # type: ignore
except Exception:  # pragma: no cover
    _pkg_resources = None  # type: ignore


def _dist_name_candidates() -> list[str]:
    """
    Spróbuj odgadnąć nazwę dystrybucji na podstawie nazwy pakietu oraz
    opcjonalnych zmiennych środowiskowych:
      - <TOP>_DISTNAME  (np. CORE_DISTNAME)
      - PACKAGE_DISTNAME
    """
    top = (__name__.split(".", 1)[0] or __package__ or "core").strip()
    env_candidates = [
        os.getenv(f"{top.upper()}_DISTNAME"),
        os.getenv("PACKAGE_DISTNAME"),
    ]
    names = {top}
    for n in env_candidates:
        if n and n.strip():
            names.add(n.strip())
    # Kilka typowych wariantów nazwy dystrybucji
    names.add(top.replace("_", "-"))
    names.add(top.replace("-", "_"))
    return [n for n in names if n]


def _read_version_file() -> Optional[str]:
    """
    Spróbuj odczytać wersję z pliku VERSION znajdującego się obok __init__.py.
    """
    try:
        from pathlib import Path
        p = Path(__file__).with_name("VERSION")
        if p.exists():
            s = p.read_text(encoding="utf-8").strip()
            if s:
                return s
    except Exception:
        pass
    return None


def _normalize_version(v: str) -> str:
    """
    Uprość wersję do formatu semver-owego (nie ingerując w suffixy typu +meta).
    Jeżeli wygląda „dziwnie”, zwróci oryginał (bez spacji).
    """
    v = (v or "").strip()
    if not v:
        return v
    # akceptujemy klasyczne X.Y[.Z] oraz z sufiksami np. 1.2.3rc1
    if re.match(r"^\d+(\.\d+){0,2}([a-zA-Z0-9\-\.\+]+)?$", v):
        return v
    return v


def _detect_version() -> str:
    """
    Strategia detekcji wersji (najpierw lekkie, potem cięższe):
      1) ENV: <TOP>_VERSION lub PACKAGE_VERSION
      2) plik VERSION obok __init__.py
      3) importlib.metadata.version dla kandydatów dystrybucji
      4) pkg_resources.get_distribution (jeśli dostępne)
      5) „0.0.0”
    """
    # 1) ENV
    top = (__name__.split(".", 1)[0] or __package__ or "core").strip()
    env_version = os.getenv(f"{top.upper()}_VERSION") or os.getenv("PACKAGE_VERSION")
    if env_version and env_version.strip():
        return _normalize_version(env_version)

    # 2) Plik VERSION
    vfile = _read_version_file()
    if vfile:
        return _normalize_version(vfile)

    # 3) importlib.metadata
    if _im_version is not None:
        for dist in _dist_name_candidates():
            try:
                return _normalize_version(_im_version(dist))
            except _IMPkgNotFound:
                continue

    # 4) pkg_resources (cięższe, ale dość niezawodne)
    if _pkg_resources is not None:  # pragma: no cover
        for dist in _dist_name_candidates():
            try:
                return _normalize_version(_pkg_resources.get_distribution(dist).version)  # type: ignore[attr-defined]
            except Exception:
                continue

    # 5) Fallback
    return "0.0.0"


def _parse_version_tuple(v: str) -> tuple[int, int, int]:
    """
    Zwróć (major, minor, patch) wyciągnięte z wersji, brakujące części ustaw na 0.
    Sufiksy są ignorowane (np. '1.2.3rc1' -> (1,2,3)).
    """
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", v or "")
    if not m:
        return (0, 0, 0)
    parts = [int(p) if p is not None else 0 for p in m.groups()]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]


def get_version() -> str:
    """Publiczna funkcja zwracająca wersję pakietu."""
    return __version__


def get_version_info() -> tuple[int, int, int]:
    """Zwróć wersję jako krotkę (major, minor, patch)."""
    return _parse_version_tuple(__version__)


__version__: Final[str] = _detect_version()

# Co eksportujemy z `from core import *`
__all__ = ["__version__", "get_version", "get_version_info"]
