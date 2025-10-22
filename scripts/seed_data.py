# scripts/seed_data.py
"""
TMIV – Seed Data Script (ready-to-use)

Generuje syntetyczne zbiory danych do szybkich testów UI/EDA/ML:
- Klasyfikacja (binary/multiclass)
- Regresja
- Szeregi czasowe (univariate + exog)

Bez twardych zależności: korzysta z numpy/pandas.
Jeśli dostępny scikit-learn, użyje go do lepszej jakości danych (soft-import).

Użycie (przykłady):
    # 1) Klasyfikacja binarna, 5k wierszy, 12 cech
    python scripts/seed_data.py --kind classification --rows 5000 --features 12 \
        --classes 2 --imbalance 0.2 --out data/cls_demo.csv

    # 2) Regresja, 10k wierszy, domyślne parametry
    python scripts/seed_data.py --kind regression --rows 10000 --features 10 --out data/reg_demo.csv

    # 3) Szereg czasowy (2 lata dziennie)
    python scripts/seed_data.py --kind timeseries --rows 730 --freq D --out data/ts_demo.csv

Parametry przydatne:
    --missing 0.02        # 2% losowych braków (poza kolumnami kluczowymi)
    --outliers 0.005      # 0.5% outlierów w cechach num.
    --seed 42             # deterministyczny seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# opcjonalnie scikit-learn (lepsze generatory), ale skrypt działa także bez niego
try:  # pragma: no cover
    from sklearn.datasets import make_classification as _make_classification
    from sklearn.datasets import make_regression as _make_regression

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


# =========================
# Utils
# =========================

def _fingerprint_df(df: pd.DataFrame, *, rows: int = 64) -> str:
    """Lekki fingerprint DF na podstawie kilku pierwszych wierszy i schematu."""
    head = df.head(rows).to_csv(index=False).encode("utf-8", errors="ignore")
    schema = json.dumps(
        {"cols": [(c, str(t)) for c, t in zip(df.columns, df.dtypes)]}, sort_keys=True
    ).encode("utf-8", errors="ignore")
    h = hashlib.sha1()
    h.update(head)
    h.update(schema)
    return h.hexdigest()[:16]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _random_state(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(int(seed) if seed is not None else None)


def _snake_names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i+1}" for i in range(n)]


def add_missing_values(df: pd.DataFrame, frac: float, *, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Losowo wprowadź NaN w wybranym ułamku komórek (poza exclude)."""
    if frac <= 0:
        return df
    rs = _random_state(12345)
    mask_cols = [c for c in df.columns if c not in set(exclude)]
    total = df[mask_cols].size
    k = int(total * float(frac))
    if k <= 0:
        return df
    # losowe indeksy (wiersz, kolumna)
    rows = rs.integers(0, len(df), size=k)
    cols = rs.integers(0, len(mask_cols), size=k)
    out = df.copy()
    for r, c in zip(rows, cols):
        col = mask_cols[int(c)]
        # nie psuj typów nie-numeric w sposób destrukcyjny (np. int→float); pandas poradzi sobie z NaN
        out.at[int(r), col] = np.nan
    return out


def add_outliers(df: pd.DataFrame, frac: float, *, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Wstrzyknij outliery do kolumn numerycznych (losowy znak i skala)."""
    if frac <= 0:
        return df
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in set(exclude)]
    if not num_cols:
        return df
    rs = _random_state(54321)
    out = df.copy()
    m = int(len(out) * float(frac))
    for col in num_cols:
        idx = rs.choice(len(out), size=max(1, m // max(1, len(num_cols))), replace=False)
        scale = np.nanstd(out[col].to_numpy(dtype=float)) or 1.0
        noise = rs.normal(loc=0.0, scale=6.0 * scale, size=len(idx))
        # losowo dodatnie/ujemne
        sign = rs.choice([-1.0, 1.0], size=len(idx))
        out.loc[out.index[idx], col] = out.loc[out.index[idx], col].to_numpy(dtype=float) + sign * noise
    return out


def save_df(df: pd.DataFrame, out_path: Path) -> Path:
    """Zapisz DF do formatu wynikającego z rozszerzenia (csv/json/parquet)."""
    _ensure_dir(out_path)
    ext = out_path.suffix.lower()
    if ext == ".csv" or ext == "":
        df.to_csv(out_path, index=False)
    elif ext == ".json":
        df.to_json(out_path, orient="records", lines=False, force_ascii=False)
    elif ext == ".parquet":
        try:
            df.to_parquet(out_path, index=False)  # wymaga pyarrow lub fastparquet
        except Exception as e:
            # fallback na CSV obok
            csv_fallback = out_path.with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            print(f"[warn] parquet niedostępny ({e}); zapisano CSV: {csv_fallback}")
            return csv_fallback
    else:
        # nieznane rozszerzenie → CSV
        csv = out_path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv
    return out_path


# =========================
# Generatory datasetów
# =========================

def gen_classification(
    rows: int,
    *,
    features: int = 8,
    classes: int = 2,
    imbalance: float = 0.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Klasyfikacja: wygeneruj X (features_*) + label.
    - imbalance: 0..0.9 (udział klasy 0.0..0.9 jako dominującej)
    """
    rs = _random_state(seed)
    if _HAVE_SKLEARN:
        weights = None
        if classes == 2 and imbalance > 0:
            p0 = min(max(float(imbalance), 0.0), 0.95)
            weights = [p0, 1.0 - p0]
        X, y = _make_classification(
            n_samples=int(rows),
            n_features=int(features),
            n_informative=max(2, int(features * 0.6)),
            n_redundant=max(0, int(features * 0.2)),
            n_repeated=0,
            n_classes=int(classes),
            weights=weights,
            class_sep=1.2,
            flip_y=0.01,
            random_state=int(seed) if seed is not None else None,
        )
    else:
        # fallback: mieszanki gaussowskie
        centers = rs.normal(0.0, 3.0, size=(int(classes), int(features)))
        X = []
        y = []
        # rozdziel próbki z uwzględnieniem (opcjonalnego) niezbalansowania
        if classes == 2 and imbalance > 0:
            p0 = min(max(float(imbalance), 0.0), 0.95)
            counts = [int(rows * p0), rows - int(rows * p0)]
        else:
            per = rows // int(classes)
            counts = [per] * int(classes)
            counts[-1] += rows - sum(counts)
        for cls, n in enumerate(counts):
            X.append(rs.normal(centers[cls], 1.2, size=(int(n), int(features))))
            y.append(np.full(int(n), cls, dtype=int))
        X = np.vstack(X)
        y = np.concatenate(y)
        idx = rs.permutation(rows)
        X, y = X[idx], y[idx]

    cols = _snake_names("feature", int(features))
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y.astype(int)
    return df


def gen_regression(
    rows: int,
    *,
    features: int = 10,
    noise: float = 0.1,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Regresja: X (features_*) + target."""
    rs = _random_state(seed)
    if _HAVE_SKLEARN:
        X, y = _make_regression(
            n_samples=int(rows),
            n_features=int(features),
            n_informative=max(2, int(features * 0.6)),
            noise=float(noise) * 10.0,
            random_state=int(seed) if seed is not None else None,
        )
    else:
        X = rs.normal(0.0, 1.0, size=(int(rows), int(features)))
        w = rs.normal(0.0, 2.0, size=(int(features), 1))
        y = (X @ w).ravel() + rs.normal(0.0, float(noise) * 5.0, size=int(rows))
    cols = _snake_names("feature", int(features))
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.astype(float)
    return df


def gen_timeseries(
    rows: int,
    *,
    freq: str = "D",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Szereg czasowy (univariate) z trendem + sezonowościami + szumem.
    Kolumny: date, target, exog_* (dow, month), opcjonalne szumy.
    """
    rs = _random_state(seed)
    n = int(rows)
    if n < 10:
        raise ValueError("Za mało wierszy dla szeregu czasowego (min 10).")

    # Osie czasu
    start = pd.Timestamp("2018-01-01")
    idx = pd.date_range(start=start, periods=n, freq=freq)

    t = np.arange(n)
    trend = 0.02 * t  # lekki trend
    season_w = np.sin(2 * np.pi * t / max(7, int(7)))  # tygodniowa
    season_y = np.sin(2 * np.pi * t / max(365, int(365 / (1 if freq.upper() == "D" else 7))))  # roczna (przybliżenie)
    noise = rs.normal(0.0, 0.5, size=n)
    y = 10 + trend + 2.0 * season_w + 1.5 * season_y + noise

    df = pd.DataFrame({"date": idx, "target": y.astype(float)})
    # egzogeniczne / kalendarzowe
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    # weekend flag
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


# =========================
# CLI
# =========================

@dataclass
class Args:
    kind: str
    rows: int
    features: int
    classes: int
    imbalance: float
    noise: float
    freq: str
    missing: float
    outliers: float
    seed: int | None
    out: Path


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="TMIV – Seed synthetic datasets.")
    p.add_argument("--kind", choices=["classification", "regression", "timeseries"], required=True)
    p.add_argument("--rows", type=int, required=True, help="Liczba wierszy/okresów")
    p.add_argument("--features", type=int, default=10, help="Liczba cech (cls/reg)")
    p.add_argument("--classes", type=int, default=2, help="Liczba klas (classification)")
    p.add_argument("--imbalance", type=float, default=0.0, help="Niezbalansowanie (0..0.95) dla binary")
    p.add_argument("--noise", type=float, default=0.1, help="Poziom szumu (regresja)")
    p.add_argument("--freq", type=str, default="D", help="Częstotliwość TS (np. D, W, M)")
    p.add_argument("--missing", type=float, default=0.0, help="Ułamek braków (0..1)")
    p.add_argument("--outliers", type=float, default=0.0, help="Ułamek outlierów (0..1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None, help="Ścieżka wyjściowa (csv/json/parquet)")
    a = p.parse_args()

    # domyślna nazwa pliku gdy brak --out
    if a.out is None:
        fname = f"{a.kind}_rows{a.rows}.csv"
        a.out = Path("data") / fname

    return Args(
        kind=a.kind,
        rows=int(a.rows),
        features=int(a.features),
        classes=int(a.classes),
        imbalance=float(a.imbalance),
        noise=float(a.noise),
        freq=str(a.freq),
        missing=float(a.missing),
        outliers=float(a.outliers),
        seed=int(a.seed) if a.seed is not None else None,
        out=a.out,
    )


def main() -> None:
    args = parse_args()

    if args.kind == "classification":
        df = gen_classification(
            args.rows,
            features=args.features,
            classes=args.classes,
            imbalance=args.imbalance,
            seed=args.seed,
        )
        exclude = ["label"]
    elif args.kind == "regression":
        df = gen_regression(
            args.rows,
            features=args.features,
            noise=args.noise,
            seed=args.seed,
        )
        exclude = ["target"]
    else:  # timeseries
        df = gen_timeseries(args.rows, freq=args.freq, seed=args.seed)
        exclude = ["date", "target"]

    # brakujące i outliery (opcjonalnie)
    if args.missing > 0:
        df = add_missing_values(df, args.missing, exclude=exclude)
    if args.outliers > 0:
        df = add_outliers(df, args.outliers, exclude=exclude)

    # zapis
    path = save_df(df, args.out)
    fp = _fingerprint_df(df)
    print(json.dumps(
        {
            "ok": True,
            "path": str(path),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(map(str, df.columns)),
            "fingerprint": fp,
            "args": asdict(args),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
