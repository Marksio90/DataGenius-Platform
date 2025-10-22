from __future__ import annotations

import os
import zipfile
import pathlib
from typing import Iterable, Optional, Sequence
from fnmatch import fnmatch


# =========================
# Pomocnicze
# =========================

_DEFAULT_EXCLUDES: tuple[str, ...] = (
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
    "*.tmp",
    "*.temp",
    "*.log",
)

def _should_exclude(name: str, excludes: Sequence[str]) -> bool:
    base = os.path.basename(name)
    if base.startswith(".") and base not in (".gitkeep",):  # ukryte pliki
        return True
    for pat in excludes:
        if fnmatch(base, pat) or fnmatch(name.replace("\\", "/"), pat):
            return True
    return False

def _iter_files_sorted(base: pathlib.Path, excludes: Sequence[str]) -> Iterable[pathlib.Path]:
    # deterministyczny porządek
    for root, dirs, files in os.walk(base):
        # odfiltruj katalogi po wzorcach (np. __pycache__)
        dirs[:] = sorted([d for d in dirs if not _should_exclude(d, excludes)])
        for f in sorted(files):
            fp = pathlib.Path(root) / f
            if _should_exclude(str(fp), excludes):
                continue
            yield fp

def _open_zip(zip_path: pathlib.Path, *, compresslevel: int = 9):
    # ZIP_DEFLATED jeśli dostępny, inaczej ZIP_STORED
    compression = getattr(zipfile, "ZIP_DEFLATED", zipfile.ZIP_STORED)
    # compresslevel obsługiwany w Py3.7+
    try:
        return zipfile.ZipFile(zip_path, "w", compression=compression, compresslevel=compresslevel)
    except TypeError:  # starsze Pythony bez compresslevel
        return zipfile.ZipFile(zip_path, "w", compression=compression)

def _write_file(
    zf: zipfile.ZipFile,
    src_path: pathlib.Path,
    arcname: str,
    *,
    reproducible: bool
) -> None:
    if reproducible:
        # ustaw stały timestamp i prawa pliku dla powtarzalnych buildów
        zi = zipfile.ZipInfo(arcname)
        zi.date_time = (2020, 1, 1, 0, 0, 0)  # stała data
        zi.external_attr = 0o644 << 16        # rw-r--r--
        with open(src_path, "rb") as f:
            zf.writestr(zi, f.read())
    else:
        zf.write(src_path, arcname=arcname)


# =========================
# Publiczne API
# =========================

def make_artifacts_zip(
    base_dir: str = "artifacts",
    zip_name: str = "artifacts_with_report.zip",
    *,
    excludes: Sequence[str] = _DEFAULT_EXCLUDES,
    compresslevel: int = 9,
    reproducible: bool = False,
) -> str:
    """
    Spakuj cały katalog `artifacts` do ZIP-a w katalogu nadrzędnym projektu,
    z zachowaniem prefiksu ścieżek w archiwum jako `artifacts/...`.

    Parametry:
      - excludes: wzorce wykluczeń (fnmatch), np. ["__pycache__", "*.tmp"]
      - compresslevel: 1..9 (jeśli wspierane przez daną wersję Pythona)
      - reproducible: jeśli True, ustawia stałe timestampy i uprawnienia w ZIP-ie

    Zwraca pełną ścieżkę do ZIP-a (str).
    """
    base = pathlib.Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # ZIP w katalogu projektu (nadrzędny względem artifacts)
    zip_path = (base.parent / zip_name) if str(base.parent) != "" else pathlib.Path(zip_name)
    if zip_path.exists():
        zip_path.unlink()

    with _open_zip(zip_path, compresslevel=compresslevel) as zf:
        for fp in _iter_files_sorted(base, excludes):
            # arcname z prefiksem "artifacts/..." (jak w Twojej wersji)
            rel = fp.relative_to(base.parent if str(base.parent) != "" else pathlib.Path("."))
            # na wszelki wypadek wymuś slashe POSIX (spójność na Windows)
            arc = str(rel).replace("\\", "/")
            _write_file(zf, fp, arc, reproducible=reproducible)

    return str(zip_path)


def make_plots_zip(
    plots_dir: str = "artifacts/plots",
    zip_name: str = "plots_bundle.zip",
    *,
    excludes: Sequence[str] = _DEFAULT_EXCLUDES,
    compresslevel: int = 9,
    reproducible: bool = False,
) -> str:
    """
    Spakuj `artifacts/plots` do ZIP-a (plik w `artifacts/`) z zachowaniem
    prefiksu `artifacts/plots/...` w archiwum (jak w Twojej wersji).

    Parametry analogiczne jak w `make_artifacts_zip`.
    """
    base = pathlib.Path(plots_dir)
    base.mkdir(parents=True, exist_ok=True)

    # ZIP w katalogu artifacts (rodzicu plots)
    zip_path = (base.parent / zip_name) if str(base.parent) != "" else pathlib.Path(zip_name)
    if zip_path.exists():
        zip_path.unlink()

    with _open_zip(zip_path, compresslevel=compresslevel) as zf:
        for fp in _iter_files_sorted(base, excludes):
            rel = fp.relative_to(base.parent if str(base.parent) != "" else pathlib.Path("."))
            arc = str(rel).replace("\\", "/")
            _write_file(zf, fp, arc, reproducible=reproducible)

    return str(zip_path)
