import re
from difflib import SequenceMatcher
from math import sqrt
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from backend.safe_utils import truthy_df_safe

# --- Słowa kluczowe, które ZWIĘKSZAJĄ szanse kolumny na bycie targetem ---
_TARGET_SYNONYMS = [
    # Twoje
    "target", "y", "label", "class", "classes", "outcome", "response",
    "churn", "fraud", "default", "clicked", "click", "convert", "converted",
    "type", "category", "segment",
    # Dodatkowe typowe
    "is_", "has_", "flag", "status", "success", "failure", "won", "lost",
    "positive", "negative", "pass", "fail", "accepted", "rejected", "approved",
    "cancelled", "returned", "active", "inactive"
]

# --- Słowa kluczowe, które ZMNIEJSZAJĄ szanse (predykcje, indeksy, czasy) ---
_EXCLUDE_NAME_TOKENS = [
    "id", "uuid", "guid", "index", "row", "timestamp", "time", "date",
    "created", "updated", "prob", "proba", "probability", "score",
    "prediction", "pred", "fold", "split"
]


def _similar(a: str, b: str) -> float:
    """Fuzzy similarity in [0,1] (case-insensitive)."""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _is_datetime_series(s: pd.Series) -> bool:
    """
    Heurystyka: typ datetime lub ≥80% próbek parsowalnych na datę
    (z pominięciem kolumn czysto numerycznych).
    """
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if pd.api.types.is_numeric_dtype(s):
            return False
        sample = s.dropna()
        if sample.empty:
            return False
        if len(sample) > 80:
            sample = sample.sample(80, random_state=0)
        parsed = pd.to_datetime(sample.astype(str), errors="coerce")
        return parsed.notna().mean() >= 0.80
    except Exception:
        return False


def _looks_like_id(name: str) -> bool:
    nm = name.lower()
    return any(tok in nm for tok in ["id", "_id", "uuid", "guid", "index"])


def _looks_like_pred(name: str) -> bool:
    nm = name.lower()
    return any(tok in nm for tok in ["prob", "proba", "probability", "score", "prediction", "pred"])


def _penalize_by_name(name: str) -> float:
    nm = name.lower()
    pen = 0.0
    for tok in _EXCLUDE_NAME_TOKENS:
        if tok in nm:
            pen -= 0.8
    # dodatkowa kara za bardzo „techniczne” nazwy
    if re.search(r"(id|uuid|guid)$", nm):
        pen -= 0.6
    return pen


def _rank_candidate_targets(df: pd.DataFrame) -> List[Tuple[str, float, str]]:
    """
    Zwraca listę (kolumna, score, powód) posortowaną malejąco po score.
    Heurystyki uwzględniają nazwę, typ, liczność wartości, braki etc.
    """
    out: List[Tuple[str, float, str]] = []
    n = len(df)
    last_idx = len(df.columns) - 1

    for i, c in enumerate(df.columns):
        name = str(c)
        name_l = name.lower()
        col = df[c]
        s = 0.0
        reason_bits = []

        # 1) Nazwa kolumny vs słowniki
        for syn in _TARGET_SYNONYMS:
            if syn == name_l:
                s += 2.5
                reason_bits.append("exact-name")
            elif syn in name_l:
                s += 1.2
                reason_bits.append(f"name-contains:{syn}")

        s += max(_similar(name_l, "target"), _similar(name_l, "label")) * 1.0

        # 2) Kara za „złe” nazwy (ID, predykcje, timestamp)
        pen = _penalize_by_name(name)
        if pen != 0.0:
            s += pen
            if pen < 0:
                reason_bits.append("name-penalty")

        # 3) Typ/rozpiętość/unikalność/braki
        nunq = int(col.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(col)
        is_bool_like = pd.api.types.is_bool_dtype(col)
        is_datetime = _is_datetime_series(col)
        miss = float(col.isna().mean())

        if is_bool_like or nunq == 2:
            s += 1.2
            reason_bits.append("binary")

        if (not is_num) and (not is_datetime) and nunq <= 50:
            s += 1.0
            reason_bits.append("categorical<=50")

        if is_num and nunq <= 20:
            s += 0.7
            reason_bits.append("numeric<=20")

        # mały bonus jeśli kolumna jest „na końcu” (często target)
        if i == last_idx:
            s += 0.2
            reason_bits.append("last-column")

        # kara za prawie unikalne
        if n > 0 and nunq / n > 0.90:
            s -= 0.8
            reason_bits.append("almost-unique")

        # kara za datetime
        if is_datetime:
            s -= 1.0
            reason_bits.append("datetime")

        # jeśli bardzo ciągła zmienna liczbowa (dużo unikatów), lekka kara
        if is_num and n > 0 and nunq > max(50, int(0.2 * n)):
            s -= 0.3
            reason_bits.append("very-continuous")

        # kara za typowe ID-like nazwy
        if _looks_like_id(name):
            s -= 0.7
            reason_bits.append("id-like-name")

        # kara za predykcyjne nazwy
        if _looks_like_pred(name):
            s -= 0.8
            reason_bits.append("pred/proba-like-name")

        # kara za dużo braków
        if miss >= 0.5:
            s -= 0.5
            reason_bits.append("many-missing")

        out.append((c, s, "+".join(reason_bits)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _is_reasonable_target(df: pd.DataFrame, col_name: str) -> bool:
    """
    Czy kolumna nadaje się na target: nie czas, nie ID/proba, nie prawie unikalna,
    nie stała, nie zbyt pusta.
    """
    s = df[col_name]
    if _is_datetime_series(s):
        return False
    nm = str(col_name).lower()
    if _looks_like_id(nm) or _looks_like_pred(nm):
        return False
    n = len(s)
    nunq = int(s.nunique(dropna=True))
    if nunq <= 1:
        return False
    if n > 0 and nunq / n > 0.98:
        return False
    if float(s.isna().mean()) > 0.9:
        return False
    return True


def auto_select_target(df: pd.DataFrame, preferred: Optional[str]) -> Tuple[str, str]:
    """
    Automatyczny wybór kolumny celu.
    Zwraca (target_column_name, reason).
    """
    if df is None or df.empty:
        raise ValueError("auto_select_target: podano pusty DataFrame")

    cols = list(map(str, df.columns))

    # 1) dokładna preferencja
    if truthy_df_safe(preferred) and preferred in cols and _is_reasonable_target(df, preferred):
        return preferred, "preferred"

    # 2) warianty nazwy preferowanej
    if truthy_df_safe(preferred):
        base = str(preferred)
        variants = {
            base, base.strip(),
            base.replace(" ", "_"), base.replace("_", " "),
            base.lower(), base.upper(), base.title()
        }
        for v in variants:
            if v in cols and _is_reasonable_target(df, v):
                return v, f"variant:{v}"

        # 2a) prefix match: np. 'type' -> 'type_organic'
        regex = re.compile(rf"^{re.escape(base)}[_\-]", re.IGNORECASE)
        for c in cols:
            if regex.search(c) and _is_reasonable_target(df, c):
                return c, f"prefix-match:{c}"

        # 2b) fuzzy dopasowanie
        best = None
        best_score = 0.0
        for c in cols:
            sc = _similar(base, c)
            if sc > best_score and _is_reasonable_target(df, c):
                best, best_score = c, sc
        # próg obniżony do 0.70, żeby łapać np. 'type' vs 'type_organic'
        if truthy_df_safe(best) and best_score >= 0.70:
            return best, f"fuzzy:{best}(score={best_score:.2f})"

    # 3) ranking kandydatów i wybór pierwszego sensownego
    ranked = _rank_candidate_targets(df)
    for c, sc, why in ranked:
        if _is_reasonable_target(df, c):
            return c, f"auto:{why} (score={sc:.2f})"

    # 4) twardy fallback: ostatnia kolumna
    return cols[-1], "fallback:last_column"


def _can_stratify_series(y: pd.Series) -> bool:
    """Stratyfikacja możliwa tylko, gdy każda klasa ma ≥2 próbki i jest ich ≥2."""
    try:
        vc = y.value_counts(dropna=False)
        return (len(vc) > 1) and (vc.min() >= 2)
    except Exception:
        return False


def stratified_sample_df(
    df: pd.DataFrame,
    target: str,
    n_max: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, str]:
    """
    Zwraca próbkę o rozmiarze n_max:
    - Klasyfikacja: stratyfikacja z zachowaniem udziałów (min. 1 na klasę jeśli to możliwe).
    - Regresja (ciągła): quasi-stratyfikacja przez qcut na koszyki (<= 10 binów).
    Gwarantuje dokładnie n_max wierszy (o ile n_max <= len(df)).
    """
    n = len(df)
    if n_max <= 0:
        return df.iloc[0:0].copy(), "Poproszono o 0 wierszy — zwracam pustą próbkę."
    if n == 0:
        return df.copy(), "Zbiór jest pusty — zwracam pustą próbkę."
    if n_max >= n:
        return df.copy(), f"Zbiór ma {n} wierszy ≤ {n_max} — używam pełnego zbioru."

    if target not in df.columns:
        raise KeyError(f"Brak kolumny '{target}' w DataFrame.")

    s = df[target]
    # dla stratyfikacji pracujemy wyłącznie na wierszach z niepustym targetem
    idx_valid = s.dropna().index
    work = df.loc[idx_valid]
    if work.empty:
        # nie ma wierszy z targetem — zwróć losową próbkę z oryginału
        out = df.sample(n=n_max, random_state=random_state)
        out = out.sample(frac=1.0, random_state=random_state).head(n_max)
        return out, f"Target pusty — losowe {n_max} z {n}."

    y = work[target]

    # Klasyfikacja: niska liczba unikatów lub typ nie-numeryczny/bool
    is_classification = (not pd.api.types.is_numeric_dtype(y)) or (int(y.nunique()) <= 20)

    if is_classification:
        # Jeśli stratyfikacja „bezpieczna”, licz proporcjonalnie; w przeciwnym razie prosty losowy
        if not _can_stratify_series(y):
            out = work.sample(n=n_max, random_state=random_state)
            out = out.sample(frac=1.0, random_state=random_state).head(n_max)
            return out, f"Klasyfikacja: niewystarczająca liczebność do stratyfikacji — losowe {n_max} z {n}."

        # Stratyfikacja po klasach — przydział proporcjonalny (metoda „largest remainder”)
        counts = y.value_counts()
        classes = counts.index.tolist()
        total_valid = int(counts.sum())
        frac = n_max / total_valid

        alloc = {}
        remainders = []
        for cls in classes:
            ideal = counts[cls] * frac
            k = int(np.floor(ideal))
            rem = float(ideal - k)
            alloc[cls] = k
            remainders.append((cls, rem))

        # min. 1 dla klas, jeśli w ogóle to możliwe (gdy n_max ≥ liczba klas)
        if n_max >= len(classes):
            for cls in classes:
                if alloc[cls] == 0 and counts[cls] > 0:
                    alloc[cls] = 1

        # skoryguj, aby suma == n_max
        current = sum(alloc.values())
        if current < n_max:
            remainders.sort(key=lambda x: x[1], reverse=True)
            for cls, _ in remainders:
                if current >= n_max:
                    break
                if alloc[cls] < counts[cls]:
                    alloc[cls] += 1
                    current += 1
        elif current > n_max:
            for cls in sorted(alloc.keys(), key=lambda k: alloc[k], reverse=True):
                while current > n_max and alloc[cls] > 0:
                    alloc[cls] -= 1
                    current -= 1
                if current <= n_max:
                    break

        parts = []
        rng = np.random.RandomState(random_state)
        for cls in classes:
            need = max(0, min(alloc[cls], counts[cls]))
            if need > 0:
                g = work[work[target] == cls]
                parts.append(g.sample(n=need, random_state=int(rng.randint(0, 2**32 - 1))))
        out = pd.concat(parts, axis=0) if parts else work.sample(n=n_max, random_state=random_state)

        # uzupełnij, jeśli przez małe klasy zabrakło rekordów
        if len(out) < n_max:
            missing = n_max - len(out)
            extra = work.drop(index=out.index, errors="ignore").sample(
                n=missing, random_state=random_state
            )
            out = pd.concat([out, extra], axis=0)

        out = out.sample(frac=1.0, random_state=random_state).head(n_max)
        return out, f"Klasyfikacja: stratyfikowana próbka {len(out)} z {n}."

    else:
        # Regresja: quasi-stratyfikacja przez quantyle
        y_num = pd.to_numeric(y, errors="coerce").dropna()
        if y_num.empty:
            out = work.sample(n=n_max, random_state=random_state)
            out = out.sample(frac=1.0, random_state=random_state).head(n_max)
            return out, f"Regresja: target pusty po konwersji — losowe {n_max} z {n}."

        # liczba koszyków: min(10, sqrt(n)), ale >= 3
        n_bins = max(3, min(10, int(round(sqrt(len(y_num))))))
        try:
            bins = pd.qcut(y_num, q=n_bins, duplicates="drop")
        except Exception:
            # gdy qcut się nie uda – fallback na prosty losowy sampling
            out = work.sample(n=n_max, random_state=random_state)
            out = out.sample(frac=1.0, random_state=random_state).head(n_max)
            return out, f"Regresja: próbka {n_max} z {n} (fallback)."

        work2 = work.loc[y_num.index].copy()
        work2["_bin"] = bins

        counts = work2["_bin"].value_counts()
        total_valid = int(counts.sum())
        frac = n_max / total_valid

        alloc = {}
        remainders = []
        for b in counts.index:
            ideal = counts[b] * frac
            k = int(np.floor(ideal))
            rem = float(ideal - k)
            alloc[b] = k
            remainders.append((b, rem))

        # koryguj do n_max
        current = sum(alloc.values())
        if current < n_max:
            remainders.sort(key=lambda x: x[1], reverse=True)
            for b, _ in remainders:
                if current >= n_max:
                    break
                if alloc[b] < counts[b]:
                    alloc[b] += 1
                    current += 1
        elif current > n_max:
            for b in sorted(alloc.keys(), key=lambda k: alloc[k], reverse=True):
                while current > n_max and alloc[b] > 0:
                    alloc[b] -= 1
                    current -= 1
                if current <= n_max:
                    break

        parts = []
        rng = np.random.RandomState(random_state)
        for b in counts.index:
            need = max(0, min(alloc[b], counts[b]))
            if need > 0:
                g = work2[work2["_bin"] == b].drop(columns=["_bin"])
                parts.append(g.sample(n=need, random_state=int(rng.randint(0, 2**32 - 1))))
        out = pd.concat(parts, axis=0) if parts else work.sample(n=n_max, random_state=random_state)

        # uzupełnij, jeśli przez małe biny zabrakło rekordów
        if len(out) < n_max:
            missing = n_max - len(out)
            extra = work.drop(index=out.index, errors="ignore").sample(
                n=missing, random_state=random_state
            )
            out = pd.concat([out, extra], axis=0)

        out = out.sample(frac=1.0, random_state=random_state).head(n_max)
        return out, f"Regresja: quasi-stratyfikowana próbka {len(out)} z {n}."
