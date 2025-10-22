from __future__ import annotations

import pandas as pd

def sanitize_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolica typy tak, by nie sprawiały kłopotów w sklearn/plotting:
      • string[python]/string[pyarrow]  -> object
      • BooleanDtype (z NA)             -> object (żeby nie wybuchało przy astype(bool))
      • BooleanDtype (bez NA)           -> bool
      • category                        -> category (bez zmian)
      • nullable Int* (Int64 itp.)      -> int64 (bez NA) lub float64 (jeśli są NA)
      • ArrowDtype (numeryczne)         -> int64/float64
      • datetime z TZ                   -> UTC bez TZ (naive)
    Reszta pozostaje bez zmian.
    """
    df = df.copy()

    from pandas.api.types import (
        is_string_dtype, is_bool_dtype, is_categorical_dtype, is_integer_dtype,
        is_float_dtype, is_extension_array_dtype, is_datetime64_any_dtype, is_object_dtype
    )

    def _is_arrow_dtype(dtype) -> bool:
        # Bez zależności od pyarrow — rozpoznaj po nazwie klasy dtype
        try:
            return dtype.__class__.__name__.endswith("ArrowDtype")
        except Exception:
            return False

    for c in df.columns:
        s = df[c]
        dt = s.dtype

        try:
            # --- STRING (w tym string[pyarrow]) -> object
            if is_string_dtype(dt):
                df[c] = s.astype(object)
                continue

            # --- ARROW-backed typy (obsługa numerycznych/tekstowych/bool)
            if _is_arrow_dtype(dt):
                # stringowe Arrow -> object
                if is_string_dtype(dt):
                    df[c] = s.astype(object)
                    continue
                # bool Arrow
                if is_bool_dtype(dt):
                    if s.isna().any():
                        df[c] = s.astype(object)      # bezpiecznie dla NA
                    else:
                        # U niektórych wersji pandas/pyarrow potrzebny jest "astype(bool)" na seriach bez NA
                        df[c] = s.astype(bool)
                    continue
                # liczby Arrow
                if is_integer_dtype(dt):
                    # jeżeli są NA — float64 (sklearn toleruje NaN w float)
                    # jeżeli brak NA — int64
                    if s.isna().any():
                        df[c] = pd.to_numeric(s, errors="coerce").astype("float64")
                    else:
                        df[c] = pd.to_numeric(s, errors="coerce").astype("int64")
                    continue
                if is_float_dtype(dt):
                    df[c] = pd.to_numeric(s, errors="coerce").astype("float64")
                    continue
                # fallback dla nieobsłużonych Arrow typów
                df[c] = s.astype(object)
                continue

            # --- BOOLEAN (pandas BooleanDtype / numpy bool_)
            if is_bool_dtype(dt):
                # pandas BooleanDtype ma NA; astype(bool) na takich kolumnach wybucha
                if str(dt).lower() in ("boolean", "boolean[pyarrow]") or "BooleanDtype" in dt.__class__.__name__:
                    if s.isna().any():
                        df[c] = s.astype(object)      # zostaw jako object gdy są NA
                    else:
                        df[c] = s.astype(bool)
                else:
                    df[c] = s.astype(bool)
                continue

            # --- CATEGORY (bez zmian, ale ustalony dtype)
            if is_categorical_dtype(dt):
                try:
                    df[c] = s.astype("category")
                except Exception:
                    # jeżeli kategorie są popsute – fallback do object
                    df[c] = s.astype(object)
                continue

            # --- DATETIME (uzbrój — bez TZ)
            if is_datetime64_any_dtype(dt):
                # pewna konwersja + zbij strefę i zostaw naive UTC
                ser = pd.to_datetime(s, errors="coerce")
                try:
                    # jeśli tz-aware, sprowadź do UTC i usuń TZ
                    if getattr(ser.dt, "tz", None) is not None:
                        ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
                except Exception:
                    # jeżeli tz-aware ale nie da się tz_convert, spróbuj tz_localize(None)
                    try:
                        ser = ser.dt.tz_localize(None)
                    except Exception:
                        pass
                df[c] = ser
                continue

            # --- Nullable Int* (ExtensionArray) -> int64 lub float64
            if is_extension_array_dtype(dt) and is_integer_dtype(dt):
                if s.isna().any():
                    df[c] = s.astype("float64")       # float pozwala na NaN
                else:
                    df[c] = s.astype("int64")
                continue

            # --- Obiekty (często stringi lub mieszanki) — nic na siłę
            # Jeżeli chcesz agresywnie konwertować "liczbowe stringi" do liczb,
            # zrób to na etapie preprocesingu (tu pozostawiamy object).
            if is_object_dtype(dt):
                # nic nie robimy; OneHotEncoder poradzi sobie z object
                continue

            # --- klasyczne liczby: bez zmian
            # (gdybyś chciał wymusić float64 dla kolumn z NaN, można dodać tu logikę)
        except Exception:
            # ostatnia linia obrony: nic nie rób — zachowaj oryginał
            pass

    return df
