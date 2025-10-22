from __future__ import annotations

from backend.safe_utils import truthy_df_safe

from typing import Optional, Tuple, List, Dict, Iterable
import unicodedata
import difflib
import pandas as pd


def _strip_accents(s: str) -> str:
    try:
        return "".join(
            ch for ch in unicodedata.normalize("NFKD", s)
            if not unicodedata.combining(ch)
        )
    except Exception:
        return s


def _normalize_key(x: object) -> str:
    """
    Aggressive, but stable normalizer:
    - cast to str
    - lowercase
    - strip accents (ą->a, ł->l, etc.)
    - replace spaces/dashes with underscores
    - remove non-alnum+underscore
    - collapse multiple underscores
    - trim underscores
    """
    s = str(x)
    s = _strip_accents(s).lower()
    s = s.replace(" ", "_").replace("-", "_")
    # keep only [a-z0-9_]
    s = "".join(ch if (("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_") else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _build_norm_map(cols: Iterable[object]) -> Dict[str, object]:
    """
    Map normalized -> original column name.
    If collisions happen, keep the first occurrence to stay deterministic.
    """
    out: Dict[str, object] = {}
    for c in cols:
        key = _normalize_key(c)
        if key not in out:
            out[key] = c
    return out


def resolve_target_column(
    requested: Optional[str],
    df: pd.DataFrame,
    candidates: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Returns (target, note). Tries hard to resolve the target column if the requested one
    is missing or slightly different.

    Strategy (in order):
      1) exact match
      2) case-insensitive exact match
      3) normalized match (spaces/dashes/accents ignored, punctuation collapsed)
      4) fallback from candidates (exact, case-insensitive, normalized)
      5) fuzzy match (difflib) over normalized names (threshold 0.82)
      6) raise KeyError with a short suggestion list

    Notes:
      - Works with non-string column names (casts to str for comparison, but returns the
        original column name).
      - `note` is empty string when `requested` is already a valid column.
    """
    if df is None or df.shape[1] == 0:
        raise ValueError("resolve_target_column: provided dataframe is empty")

    cols = list(df.columns)
    cols_lower_map = {str(c).lower(): c for c in cols}
    norm_map = _build_norm_map(cols)

    # 1) exact match
    if truthy_df_safe(requested) and requested in cols:
        return requested, ""

    # 2) case-insensitive exact
    if truthy_df_safe(requested):
        rq_low = str(requested).lower()
        if rq_low in cols_lower_map:
            col = cols_lower_map[rq_low]
            return col, f"⚠️ Kolumna celu '{requested}' nie istnieje w dokładnej postaci – użyto '{col}'."

    # 3) normalized match
    if truthy_df_safe(requested):
        rq_norm = _normalize_key(requested)
        if rq_norm in norm_map:
            col = norm_map[rq_norm]
            return col, f"⚠️ Kolumna celu '{requested}' nie istnieje – użyto najbliższego odpowiednika '{col}'."

    # Prepare candidate list (augment with common names)
    if not truthy_df_safe(candidates):
        candidates = ["type_organic", "type", "target", "label", "y", "class", "response", "outcome"]

    # 4) try candidates (exact → case-insensitive → normalized)
    for c in candidates:
        # exact
        if c in cols:
            return c, f"⚠️ Kolumna celu '{requested}' nie istnieje – użyto '{c}'."
        # case-insensitive
        cl = c.lower()
        if cl in cols_lower_map:
            col = cols_lower_map[cl]
            return col, f"⚠️ Kolumna celu '{requested}' nie istnieje – użyto '{col}'."
        # normalized
        cn = _normalize_key(c)
        if cn in norm_map:
            col = norm_map[cn]
            return col, f"⚠️ Kolumna celu '{requested}' nie istnieje – użyto '{col}'."

    # 5) fuzzy over normalized keys
    if truthy_df_safe(requested):
        rq_norm = _normalize_key(requested)
        norm_keys = list(norm_map.keys())
        # use a relatively strict cutoff to avoid surprising matches
        matches = difflib.get_close_matches(rq_norm, norm_keys, n=1, cutoff=0.82)
        if matches:
            best = matches[0]
            col = norm_map[best]
            return col, f"⚠️ Kolumna celu '{requested}' nie istnieje – użyto najbardziej podobnej '{col}'."

    # 6) give helpful error with suggestions
    # Build top-5 suggestions by similarity
    norm_keys = list(norm_map.keys())
    rq_key = _normalize_key(requested) if truthy_df_safe(requested) else ""
    scored = sorted(
        ((difflib.SequenceMatcher(a=rq_key, b=k).ratio(), k) for k in norm_keys),
        key=lambda t: t[0],
        reverse=True
    )
    suggestions = [str(norm_map[k]) for _, k in scored[:5] if _]
    preview_cols = [str(c) for c in cols[:25]]
    hint = f" Podobne: {', '.join(suggestions)}." if suggestions else ""
    raise KeyError(
        f"Nie znaleziono kolumny celu '{requested}'. Dostępne kolumny (pierwsze 25): {preview_cols}.{hint}"
    )
