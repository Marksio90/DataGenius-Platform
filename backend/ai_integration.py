# -*- coding: utf-8 -*-
import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from backend.safe_utils import truthy_df_safe

import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

# ======================
# DODANE: bezpieczne klucze i smart error handling
# ======================
from backend.security_manager import credential_manager
from backend.error_handler import SmartErrorHandler


# =======================================================================
#                            AIDescriptionGenerator
# =======================================================================
class AIDescriptionGenerator:
    """Generator opisów kolumn i rekomendacji ML z użyciem OpenAI/Anthropic + fallback symulacyjny."""

    _COLUMN_DESC_CACHE_DIR = Path("cache/column_descriptions")

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._ensure_cache_dir()
        self.setup_clients()

    # ======================
    # Inicjalizacja klientów
    # ======================
    def setup_clients(self):
        """Inicjalizuje klientów API AI (priorytet: credential_manager -> ENV -> st.session_state)."""
        try:
            # ---- OpenAI (priorytet: credential_manager -> ENV -> session_state)
            openai_key = None
            try:
                if truthy_df_safe(credential_manager):
                    openai_key = credential_manager.get_api_key("openai")
            except Exception:
                pass

            if not truthy_df_safe(openai_key):
                openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")

            if truthy_df_safe(openai_key):
                try:
                    # kompatybilnie z nowszym SDK
                    try:
                        from openai import OpenAI
                        self.openai_client = OpenAI(api_key=openai_key)
                    except Exception:
                        import openai  # starsze SDK
                        self.openai_client = openai.OpenAI(api_key=openai_key)
                except ImportError:
                    # backend – nie hałasujemy UI; ewentualne logowanie zrób wyżej
                    pass
                except Exception:
                    pass

            # ---- Anthropic (priorytet: credential_manager -> ENV -> session_state)
            anthropic_key = None
            try:
                if truthy_df_safe(credential_manager):
                    anthropic_key = credential_manager.get_api_key("anthropic")
            except Exception:
                pass

            if not truthy_df_safe(anthropic_key):
                anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_api_key")

            if truthy_df_safe(anthropic_key):
                try:
                    from anthropic import Anthropic
                    self.anthropic_client = Anthropic(api_key=anthropic_key)
                except ImportError:
                    pass
                except Exception:
                    pass

        except Exception:
            # brak rzucania wyjątków do UI
            pass

    # ======================
    # Narzędzia do cache'u (statyczne/klasowe)
    # ======================
    @classmethod
    def _ensure_cache_dir(cls):
        try:
            cls._COLUMN_DESC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @staticmethod
    def _normalize_key(x: str) -> str:
        return re.sub(r"[\W_]+", "", str(x or "").strip().lower())

    @classmethod
    def _df_signature_for_desc(cls, df: pd.DataFrame) -> str:
        """Stabilny podpis danych: kolumny + dtypes + próbka 50 wierszy."""
        cols = list(map(str, df.columns))
        dtypes = df.dtypes.astype(str).to_dict()
        sample = df.head(50).to_json(orient="split", index=False)
        payload = json.dumps({"cols": cols, "dtypes": dtypes, "sample": sample}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _desc_cache_path(cls, sig: str) -> Path:
        return cls._COLUMN_DESC_CACHE_DIR / f"{sig}.json"

    @classmethod
    def _desc_cache_load(cls, sig: str) -> Optional[dict]:
        p = cls._desc_cache_path(sig)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    @classmethod
    def _desc_cache_save(cls, sig: str, data: dict) -> None:
        try:
            cls._desc_cache_path(sig).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ======================
    # Opisy kolumn (public)
    # ======================
    def generate_column_descriptions(self, df: pd.DataFrame, *, force: bool = False) -> Dict[str, str]:
        """
        Generuje/odczytuje opisy kolumn z potrójnym cache:
        1) st.session_state
        2) plik cache/column_descriptions/<hash>.json
        3) dopiero wtedy AI (OpenAI/Anthropic) lub fallback.

        ZAWSZE zwraca słownik z kluczami == dokładnym nazwom kolumn.
        Parametr `force=True` pozwala ręcznie odświeżyć opisy.
        """
        sig = self._df_signature_for_desc(df)
        ss_key = f"col_desc::{sig}"

        # 1) cache w sesji
        if not force and ss_key in st.session_state and isinstance(st.session_state[ss_key], dict):
            return self._align_descriptions(st.session_state[ss_key], df)

        # 2) cache na dysku
        if not truthy_df_safe(force):
            disk = self._desc_cache_load(sig)
            if isinstance(disk, dict) and disk:
                aligned = self._align_descriptions(disk, df)
                st.session_state[ss_key] = aligned
                return aligned

        # 3) generacja tylko jeśli brak cache lub wymuszenie
        if self.openai_client:
            data = self._generate_with_openai(df)
        elif self.anthropic_client:
            data = self._generate_with_anthropic(df)
        else:
            data = self._generate_with_simulation(df)

        aligned = self._align_descriptions(data, df)
        # zapis do cache'ów
        self._desc_cache_save(sig, aligned)
        st.session_state[ss_key] = aligned

        return aligned

    # ======================
    # Wyrównanie opisów do kolumn (exact → normalized → fuzzy → fallback)
    # ======================
    def _align_descriptions(self, raw_desc: Dict[str, str], df: pd.DataFrame) -> Dict[str, str]:
        """
        Zwraca słownik {dokładna_nazwa_kolumny: opis} dla KAŻDEJ kolumny:
        1) dopasowanie exact (ten sam klucz),
        2) dopasowanie po normalizacji (usunięcie znaków, case-insensitive),
        3) fuzzy (SequenceMatcher) z progiem,
        4) braki uzupełnione opisem symulacyjnym.
        """
        if not isinstance(raw_desc, dict):
            raw_desc = {}

        # Obsługa ewentualnego gniazdowania {"descriptions": {...}}
        if "descriptions" in raw_desc and isinstance(raw_desc["descriptions"], dict):
            raw_desc = raw_desc["descriptions"]

        # Indeksy pomocnicze (zawartość -> po kluczu znormalizowanym)
        norm_to_text = {}
        for k, v in raw_desc.items():
            norm_to_text[self._normalize_key(k)] = str(v)

        # Jednorazowo licz opis symulacyjny (fallback)
        sim_all = self._generate_with_simulation(df)

        out: Dict[str, str] = {}
        for col in df.columns:
            key_exact = str(col)
            key_norm = self._normalize_key(col)

            # (1) exact
            if key_exact in raw_desc:
                out[key_exact] = str(raw_desc[key_exact])
                continue
            # (2) normalized
            if key_norm in norm_to_text:
                out[key_exact] = norm_to_text[key_norm]
                continue
            # (3) fuzzy
            best_text = None
            best_score = 0.0
            for k_raw, v in raw_desc.items():
                score = SequenceMatcher(None, key_norm, self._normalize_key(k_raw)).ratio()
                if score > best_score:
                    best_text, best_score = v, score
            if best_text is not None and best_score >= 0.86:
                out[key_exact] = str(best_text)
                continue

            # (4) fallback
            out[key_exact] = sim_all.get(key_exact, "Kolumna – opis niedostępny.")

        return out

    # ======================
    # OpenAI / Anthropic
    # ======================
    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Ekstrahuje JSON z odpowiedzi modelu:
        - usuwa ```json ... ```/``` ... ```
        - wybiera największy poprawny blok { ... }
        - w razie niepowodzenia – podnosi wyjątek
        """
        if not isinstance(text, str):
            raise ValueError("Odpowiedź modelu nie jest tekstem.")
        # usuń code fences
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.IGNORECASE)
        # jeśli cały tekst jest JSON-em
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        # spróbuj wybrać największy blok {...}
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start:end + 1]
            return json.loads(snippet)
        # ostatecznie – nie udało się
        raise ValueError("Nie udało się wyodrębnić poprawnego JSON.")

    @SmartErrorHandler.api_error_handler
    def _generate_with_openai(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generuje opisy używając OpenAI GPT (z dekoratorem SmartErrorHandler)."""
        sample_data = df.head(3).to_string()
        column_info: List[str] = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = int(df[col].nunique())
            null_count = int(df[col].isnull().sum())

            if pd.api.types.is_numeric_dtype(df[col]):
                stats = f"min: {df[col].min():.2f}, max: {df[col].max():.2f}, mean: {df[col].mean():.2f}"
            else:
                top_values = df[col].value_counts(dropna=True).head(3).index.tolist()
                stats = f"top values: {top_values}"

            column_info.append(f"{col} ({dtype}): {nunique} unique, {null_count} nulls, {stats}")

        columns_list = [str(c) for c in df.columns]

        prompt = f"""
Przeanalizuj poniższy zbiór danych i wygeneruj KRÓTKIE, biznesowe opisy dla KAŻDEJ kolumny.
ZWRÓĆ DOKŁADNIE TE SAME KLUCZE jak w 'kolumny' (BEZ tłumaczenia ani zmian znaków).

kolumny = {columns_list}

Informacje o kolumnach:
{chr(10).join(column_info)}

Przykładowe dane:
{sample_data}

Zasady:
- Klucze JSON MUSZĄ być IDENTYCZNE jak w 'kolumny'.
- Dla każdej kolumny 1–2 zdania (PL), praktycznie i zwięźle.
- Jeśli brak pewności, użyj neutralnego opisu (np. „Kolumna numeryczna, ciągła; możliwe outliery.”).

Odpowiedz w formacie JSON (obiekt mapujący): {{"nazwa_kolumny": "opis", ...}}
        """.strip()

        try:
            client = self.openai_client
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Jesteś ekspertem data science. Generujesz zwięzłe, praktyczne opisy kolumn danych."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.2,
                )
                content = response.choices[0].message.content
            else:
                raise RuntimeError("OpenAI client nie obsługuje chat.completions")

            descriptions = self._extract_json(content)
            if not isinstance(descriptions, dict):
                raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            return {str(k): str(v) for k, v in descriptions.items()}
        except Exception:
            return self._generate_with_simulation(df)

    @SmartErrorHandler.api_error_handler
    def _generate_with_anthropic(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generuje opisy używając Anthropic Claude (z dekoratorem SmartErrorHandler)."""
        sample_data = df.head(3).to_string()
        column_info: List[str] = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = int(df[col].nunique())
            null_count = int(df[col].isnull().sum())

            if pd.api.types.is_numeric_dtype(df[col]):
                stats = f"min: {df[col].min():.2f}, max: {df[col].max():.2f}, mean: {df[col].mean():.2f}"
            else:
                top_values = df[col].value_counts(dropna=True).head(3).index.tolist()
                stats = f"top values: {top_values}"

            column_info.append(f"{col} ({dtype}): {nunique} unique, {null_count} nulls, {stats}")

        columns_list = [str(c) for c in df.columns]

        prompt = f"""
Przeanalizuj poniższy zbiór danych i wygeneruj KRÓTKIE, biznesowe opisy dla KAŻDEJ kolumny.
ZWRÓĆ DOKŁADNIE TE SAME KLUCZE jak w 'kolumny' (BEZ tłumaczenia ani zmian znaków).

kolumny = {columns_list}

Informacje o kolumnach:
{chr(10).join(column_info)}

Przykładowe dane:
{sample_data}

Zasady:
- Klucze JSON MUSZĄ być IDENTYCZNE jak w 'kolumny'.
- Dla każdej kolumny 1–2 zdania (PL), praktycznie i zwięźle.

Odpowiedz w formacie JSON (obiekt mapujący): {{"nazwa_kolumny": "opis", ...}}
        """.strip()

        try:
            client = self.anthropic_client
            msg = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            content = msg.content[0].text if hasattr(msg, "content") else str(msg)

            descriptions = self._extract_json(content)
            if not isinstance(descriptions, dict):
                raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            return {str(k): str(v) for k, v in descriptions.items()}
        except Exception:
            return self._generate_with_simulation(df)

    # ======================
    # Fallback – opisy z prostej analizy statystycznej
    # ======================
    def _generate_with_simulation(self, df: pd.DataFrame) -> Dict[str, str]:
        descriptions: Dict[str, str] = {}

        for column in df.columns:
            col_data = df[column]
            null_count = int(col_data.isnull().sum())
            null_pct = float((null_count / max(len(col_data), 1)) * 100.0)
            unique_count = int(col_data.nunique(dropna=True))
            unique_pct = float((unique_count / max(len(col_data), 1)) * 100.0)

            parts: List[str] = []
            if pd.api.types.is_numeric_dtype(col_data):
                # numeryczna
                col_num = pd.to_numeric(col_data, errors="coerce")
                min_val = col_num.min()
                max_val = col_num.max()
                mean_val = col_num.mean()

                if unique_count <= 10:
                    parts.append(f"Zmienna kategoryczna numeryczna z {unique_count} unikalnymi wartościami")
                elif self._is_id_column(column, col_data):
                    parts.append("Prawdopodobnie kolumna identyfikacyjna (ID)")
                elif self._is_year_column(column, col_data):
                    parts.append("Kolumna reprezentująca rok lub datę")
                elif self._is_binary_numeric(col_data):
                    parts.append("Zmienna binarna (0/1) – flaga/indykator")
                else:
                    parts.append("Zmienna numeryczna ciągła")

                if pd.notna(min_val) and pd.notna(max_val) and pd.notna(mean_val):
                    parts.append(f"Zakres: {min_val:.2f}–{max_val:.2f}, średnia: {mean_val:.2f}")
            else:
                # tekst/kategoria
                most_common = col_data.value_counts(dropna=True).head(3)
                if unique_pct > 90:
                    parts.append("Kolumna z niemal unikalnymi wartościami (prawdopodobnie ID tekstowe)")
                elif unique_count <= 10:
                    parts.append(f"Zmienna kategoryczna z {unique_count} kategoriami")
                else:
                    parts.append(f"Zmienna tekstowa z {unique_count} unikalnymi wartościami")

                if len(most_common) > 0:
                    parts.append(f"Najczęstsze: {', '.join(most_common.index.astype(str)[:2])}")

            if null_count > 0:
                parts.append(f"⚠️ {null_pct:.1f}% brakujących danych")

            descriptions[column] = ". ".join(parts)

        return descriptions

    # ======================
    # Metody pomocnicze dla klasyfikacji kolumn
    # ======================
    def _is_id_column(self, column_name: str, col_data: pd.Series) -> bool:
        """Sprawdza czy kolumna to prawdopodobnie ID."""
        name_indicators = ["id", "key", "index", "_id", "uuid", "guid"]
        name_lower = str(column_name).lower()

        if any(indicator in name_lower for indicator in name_indicators):
            return True

        unique_ratio = col_data.nunique(dropna=True) / max(1, len(col_data))
        return unique_ratio > 0.95

    def _is_year_column(self, column_name: str, col_data: pd.Series) -> bool:
        """Sprawdza czy kolumna reprezentuje rok."""
        name_indicators = ["year", "rok", "date", "data"]
        name_lower = str(column_name).lower()
        if any(indicator in name_lower for indicator in name_indicators):
            return True

        try:
            numeric_data = pd.to_numeric(col_data, errors="coerce").dropna()
            if len(numeric_data) == 0:
                return False
            min_val = float(numeric_data.min())
            max_val = float(numeric_data.max())
            return (1800 <= min_val <= 2200) and (1800 <= max_val <= 2200)
        except Exception:
            return False

    def _is_binary_numeric(self, col_data: pd.Series) -> bool:
        """Sprawdza czy kolumna to binarna (0/1)."""
        unique_values = set(pd.to_numeric(col_data, errors="coerce").dropna().unique())
        return unique_values.issubset({0, 1}) or unique_values.issubset({0.0, 1.0})

    # ======================
    # (Opcjonalne) Rekomendacje biznesowe — bez zmian merytorycznych
    # ======================
    def generate_recommendations(self, results: Dict[str, Any], target_column: str, problem_type: str) -> Dict[str, Any]:
        """Generuje rekomendacje biznesowe na podstawie wyników modelu."""
        if self.openai_client:
            return self._generate_recommendations_openai(results, target_column, problem_type)
        elif self.anthropic_client:
            return self._generate_recommendations_anthropic(results, target_column, problem_type)
        else:
            return self._generate_recommendations_simulation(results, target_column, problem_type)

    def _generate_recommendations_openai(self, results: Dict[str, Any], target_column: str, problem_type: str) -> Dict[str, Any]:
        feature_importance = results.get("feature_importance", pd.DataFrame())
        top_features = (
            feature_importance.head(5)["feature"].tolist()
            if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty and "feature" in feature_importance.columns
            else []
        )
        metrics = {k: v for k, v in results.items() if k in ["r2", "mae", "accuracy", "f1"]}

        prompt = f"""
Wygeneruj biznesowe rekomendacje dla modelu ML:

- Typ problemu: {problem_type}
- Zmienna docelowa: {target_column}
- Najważniejsze cechy: {top_features[:3]}
- Metryki: {metrics}

Wygeneruj:
1. Kluczowe wnioski (2-3 zdania)
2. 5 konkretnych rekomendacji działań
3. Ograniczenia modelu (1-2 zdania)
4. 5 następnych kroków

Odpowiedz w JSON:
{{
  "key_insights": "tekst",
  "action_items": ["akcja1", "akcja2", "..."],
  "limitations": "tekst",
  "next_steps": ["krok1", "krok2", "..."]
}}
        """.strip()

        try:
            client = self.openai_client
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem ML i business intelligence. Generujesz praktyczne rekomendacje biznesowe."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            recs = self._extract_json(content)
            if not isinstance(recs, dict):
                raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            return recs
        except Exception:
            return self._generate_recommendations_simulation(results, target_column, problem_type)

    def _generate_recommendations_anthropic(self, results: Dict[str, Any], target_column: str, problem_type: str) -> Dict[str, Any]:
        feature_importance = results.get("feature_importance", pd.DataFrame())
        top_features = (
            feature_importance.head(5)["feature"].tolist()
            if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty and "feature" in feature_importance.columns
            else []
        )
        metrics = {k: v for k, v in results.items() if k in ["r2", "mae", "accuracy", "f1"]}

        prompt = f"""
Wygeneruj biznesowe rekomendacje dla modelu ML:

- Typ problemu: {problem_type}
- Zmienna docelowa: {target_column}
- Najważniejsze cechy: {top_features[:3]}
- Metryki: {metrics}

Odpowiedz w JSON z kluczami: key_insights, action_items, limitations, next_steps
        """.strip()

        try:
            client = self.anthropic_client
            msg = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            content = msg.content[0].text if hasattr(msg, "content") else str(msg)
            recs = self._extract_json(content)
            if not isinstance(recs, dict):
                raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            return recs
        except Exception:
            return self._generate_recommendations_simulation(results, target_column, problem_type)

    def _generate_recommendations_simulation(self, results: Dict[str, Any], target_column: str, problem_type: str) -> Dict[str, Any]:
        feature_importance = results.get("feature_importance", pd.DataFrame())
        top_features = (
            feature_importance.head(5)["feature"].tolist()
            if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty and "feature" in feature_importance.columns
            else []
        )
        top_display = ", ".join(top_features[:3]) if top_features else "główne czynniki"

        return {
            "key_insights": f"Model {str(problem_type).lower()} dla '{target_column}' identyfikuje {top_display} jako najważniejsze predyktory.",
            "action_items": [
                f"Optymalizuj {', '.join(top_features[:2]) if top_features else 'kluczowe zmienne wejściowe'}",
                "Monitoruj kluczowe metryki (np. R²/MAE albo Accuracy/F1)",
                "Zbierz dodatkowe, aktualne dane treningowe",
                "Wdróż alerty jakości (drift, spadek metryk)",
                "Planuj testy A/B dla decyzji opartych na modelu",
            ],
            "limitations": f"Model bazuje na danych historycznych i może gorzej generalizować do nietypowych przypadków {target_column}.",
            "next_steps": [
                "Walidacja na świeżych danych produkcyjnych",
                "Optymalizacja hiperparametrów",
                "Monitoring dryfu danych i cech",
                "Integracja z pipeline'em produkcyjnym",
                "Regularny retraining według harmonogramu",
            ],
        }

    # ======================
    # Rekomendacje treningu (AI/fallback)
    # ======================
    def generate_training_recommendations(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generuje rekomendacje treningu za pomocą AI lub fallback."""
        if self.openai_client:
            return self._generate_training_recommendations_openai(df, target_column)
        elif self.anthropic_client:
            return self._generate_training_recommendations_anthropic(df, target_column)
        else:
            return self._generate_training_recommendations_simulation(df, target_column)

    def _generate_training_recommendations_openai(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generuje rekomendacje treningu używając OpenAI."""
        n_samples = len(df)
        n_features = max(len(df.columns) - 1, 0)
        missing_pct = float((df.isnull().sum().sum() / max((df.shape[0] * max(df.shape[1], 1)), 1)) * 100.0)

        # Analiza targetu
        if pd.api.types.is_numeric_dtype(df[target_column]):
            target_info = f"numeryczna, zakres: {df[target_column].min():.2f} - {df[target_column].max():.2f}"
            problem_type = "regresja"
        else:
            target_counts = df[target_column].value_counts()
            target_info = f"kategoryczna, {len(target_counts)} klas, rozkład: {dict(target_counts.head(3))}"
            problem_type = "klasyfikacja"

        numeric_cols = max(len(df.select_dtypes(include=[np.number]).columns) - 1, 0)
        categorical_cols = len(df.select_dtypes(include=["object", "category"]).columns)

        prompt = f"""
Jako ekspert ML, przeanalizuj poniższy zbiór danych i podaj konkretne rekomendacje treningu:

DATASET INFO:
- Próbek: {n_samples}
- Cech: {n_features} (numerycznych: {numeric_cols}, kategorycznych: {categorical_cols})
- Target: {target_column} ({target_info})
- Problem: {problem_type}
- Brakujące dane: {missing_pct:.1f}%

Podaj rekomendacje w JSON:
{{
  "recommended_train_size": 0.8,
  "train_size_reason": "powód",
  "recommended_cv": 5,
  "cv_reason": "powód",
  "use_full_data": true,
  "performance_reason": "powód",
  "recommended_strategy": "balanced",
  "strategy_reason": "powód",
  "enable_hyperparameter_tuning": true,
  "tuning_reason": "powód",
  "recommended_metric": "roc_auc",
  "metric_reason": "powód",
  "enable_ensemble": false,
  "ensemble_reason": "powód",
  "special_considerations": ["uwaga1", "uwaga2"]
}}
        """.strip()

        try:
            client = self.openai_client
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem machine learning. Analizujesz zbiory danych i dajesz konkretne rekomendacje treningu."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            recs = self._extract_json(content)
            if not isinstance(recs, dict):
                raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            return recs
        except Exception:
            return self._generate_training_recommendations_simulation(df, target_column)

    def _generate_training_recommendations_anthropic(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generuje rekomendacje treningu używając Anthropic - fallback do symulacji."""
        return self._generate_training_recommendations_simulation(df, target_column)

    def _generate_training_recommendations_simulation(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """INTELIGENTNE rekomendacje treningu z zaawansowaną analizą — fallback."""
        n_samples = len(df)
        n_features = max(len(df.columns) - 1, 0)
        missing_pct = float((df.isnull().sum().sum() / max((df.shape[0] * max(df.shape[1], 1)), 1)) * 100.0)

        target_data = df[target_column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(target_data)

        if truthy_df_safe(is_numeric):
            problem_type = "regression"
            skewness = abs(target_data.skew()) if len(target_data) > 3 else 0
            target_range = target_data.max() - target_data.min() if len(target_data) > 0 else 0
        else:
            problem_type = "classification"
            value_counts = target_data.value_counts()
            imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 1 and value_counts.min() > 0 else 1
            n_classes = len(value_counts)

        special_considerations = []

        # Train size
        if n_samples < 500:
            train_size = 0.85
            train_reason = "Mały zbiór - maksymalizujemy dane treningowe (85/15)"
        elif n_samples < 2000:
            train_size = 0.8
            train_reason = "Średni zbiór - standardowy podział 80/20"
        elif n_samples < 10000:
            train_size = 0.75
            train_reason = "Duży zbiór - więcej danych na walidację (75/25)"
        else:
            train_size = 0.7
            train_reason = "Bardzo duży zbiór - ekstensywna walidacja (70/30)"

        if problem_type == "classification" and 'imbalance_ratio' in locals() and imbalance_ratio > 5:
            train_size = min(0.85, train_size + 0.05)
            train_reason += " + bonus za niezbalansowane klasy"
            special_considerations.append("Niezbalansowane klasy - rozważ stratified sampling")

        # CV
        if problem_type == "classification":
            if 'n_classes' in locals() and n_classes > 10:
                cv_folds = 3
                cv_reason = "Wiele klas - 3-fold CV wystarczające"
            elif n_samples < 1000:
                cv_folds = 3
                cv_reason = "Mały zbiór - 3-fold CV"
            else:
                cv_folds = 5
                cv_reason = "5-fold CV - standard dla klasyfikacji"
        else:
            if n_samples > 20000:
                cv_folds = 3
                cv_reason = "Duży zbiór - 3-fold CV wystarczające"
            elif n_samples < 1000:
                cv_folds = 3
                cv_reason = "Mały zbiór - 3-fold CV"
            else:
                cv_folds = 5
                cv_reason = "5-fold CV - standard dla regresji"

        # Performance
        if n_samples > 50000:
            use_full_data = False
            performance_reason = "Bardzo duży zbiór - użyj sampling"
            special_considerations.append("Rozważ distributed training dla zbiorów >50k")
        elif n_samples > 15000:
            use_full_data = False
            performance_reason = "Duży zbiór - sampling do ~8000 próbek dla szybkości"
            special_considerations.append("Automatyczny sampling do ~8000 próbek")
        else:
            use_full_data = True
            performance_reason = "Średni/mały zbiór - używaj wszystkich danych"

        # Strategia
        if n_samples < 1000:
            recommended_strategy = "balanced"
            strategy_reason = "Mały zbiór — zrównoważony zestaw modeli"
        elif n_samples < 10000:
            if problem_type == "classification" and 'imbalance_ratio' in locals() and imbalance_ratio > 3:
                recommended_strategy = "advanced"
                strategy_reason = "Niezbalansowane klasy — preferuj boosting/ensemble"
            else:
                recommended_strategy = "accurate"
                strategy_reason = "Średni zbiór — stawiamy na dokładność"
        else:
            recommended_strategy = "fast_small"
            strategy_reason = "Duży zbiór — priorytet: szybkość"

        # Tuning
        if n_samples < 5000 and n_features < 50:
            enable_hyperparameter_tuning = True
            tuning_reason = "Mały/średni zbiór — tuning może wiele dać"
        elif n_samples > 20000:
            enable_hyperparameter_tuning = False
            tuning_reason = "Bardzo duży zbiór — HPO zbyt wolne"
        else:
            enable_hyperparameter_tuning = False
            tuning_reason = "Domyślne parametry wystarczające"

        # Metryka
        if problem_type == "classification":
            if 'imbalance_ratio' in locals() and imbalance_ratio > 3:
                recommended_metric = "f1_weighted"
                metric_reason = "Niezbalansowane klasy — F1 weighted lepsze niż accuracy"
            else:
                recommended_metric = "accuracy"
                metric_reason = "Zbalansowane — accuracy"
        else:
            if truthy_df_safe(is_numeric) and 'skewness' in locals() and skewness > 2:
                recommended_metric = "neg_mean_absolute_error"
                metric_reason = "Silna skośność — MAE odporniejsze"
            else:
                recommended_metric = "r2"
                metric_reason = "R² — standard w regresji"

        # Ensemble
        if n_samples > 5000 and n_features > 10:
            enable_ensemble = True
            ensemble_reason = "Więcej danych i cech — ensemble pomaga"
        else:
            enable_ensemble = False
            ensemble_reason = "Mały zbiór — pojedyncze modele wystarczą"

        if missing_pct > 20:
            special_considerations.append(f"Dużo braków ({missing_pct:.1f}%) — rozważ zaawansowaną imputację")
        if n_features > n_samples:
            special_considerations.append("Więcej cech niż próbek — ryzyko overfittingu; rozważ selekcję/regularizację")
        if n_features > 100:
            special_considerations.append("Dużo cech — rozważ selekcję cech lub PCA")
        if problem_type == "regression" and 'target_range' in locals() and target_range > 1000:
            special_considerations.append("Szeroki zakres targetu — rozważ transformację (np. log)")

        if n_samples > 20000:
            special_considerations.append("Duży zbiór — trening może potrwać dłużej")

        return {
            "recommended_train_size": train_size,
            "train_size_reason": train_reason,
            "recommended_cv": cv_folds,
            "cv_reason": cv_reason,
            "use_full_data": use_full_data,
            "performance_reason": performance_reason,
            "recommended_strategy": recommended_strategy,
            "strategy_reason": strategy_reason,
            "enable_hyperparameter_tuning": enable_hyperparameter_tuning,
            "tuning_reason": tuning_reason,
            "recommended_metric": recommended_metric,
            "metric_reason": metric_reason,
            "enable_ensemble": enable_ensemble,
            "ensemble_reason": ensemble_reason,
            "special_considerations": special_considerations or ["Dataset wygląda dobrze do standardowego treningu"]
        }


# =======================================================================
#                                AIRecommender
# =======================================================================
class AIRecommender:
    """Centralny punkt rekomendacji AI: data prep plan + hyperparam rekomendacje.
    Działa z OpenAI/Anthropic jeśli klucz jest dostępny, inaczej fallback regułowy.
    """
    def __init__(self):
        self.provider = None
        self._setup_clients()

    def _setup_clients(self):
        self.openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_api_key")
        self.provider = "openai" if self.openai_key else ("anthropic" if self.anthropic_key else "fallback")

    def detect_problem(self, df: pd.DataFrame, target: Optional[str]) -> str:
        if truthy_df_safe(target) and target in df.columns:
            y = df[target].dropna()
            # klasyfikacja jeśli ma <= 20 unikalnych wartości nieciągłych i/lub typ object/bool/categorical
            if y.dtype.name in ("object", "bool", "category") or (y.nunique() <= 20 and not np.issubdtype(y.dtype, np.floating)):
                return "classification"
            return "regression"
        # brak targetu -> heurystyka po nazwach
        for col in df.columns:
            if str(col).lower() in ("target", "y", "label", "class", "price", "value", "score"):
                return "classification" if df[col].nunique() <= 20 else "regression"
        return "regression"

    def build_dataprep_plan(self, df: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
        """Zwraca plan kroków: imputacja, detekcja outlierów, skalowanie, kodowanie, split, CV."""
        problem = self.detect_problem(df, target)
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dropna().dtype, np.number) and c != target]
        cat_cols = [c for c in df.columns if c != target and c not in numeric_cols]

        plan = {
            "problem": problem,
            "target": target,
            "steps": []
        }
        plan["steps"].append({"name": "Detect target", "detail": f"Detected problem: {problem}, target: {target}"})
        plan["steps"].append({"name": "Missing values", "detail": "Numeric -> median; Categorical -> most_frequent"})
        if len(numeric_cols) > 0:
            plan["steps"].append({"name": "Outliers", "detail": "IQR capping (Q1-1.5*IQR, Q3+1.5*IQR) for numeric features"})
            plan["steps"].append({"name": "Scaling", "detail": "StandardScaler for numeric features"})
        if len(cat_cols) > 0:
            plan["steps"].append({"name": "Encoding", "detail": "One-Hot (low-card ≤15), Target/Ordinal (high-card)"})
        plan["steps"].append({"name": "Train/Validation split", "detail": "Stratified (classification) 80/20; standard (regression)"})
        plan["steps"].append({"name": "Cross-Validation", "detail": "KFold=5 (Regression) / StratifiedKFold=5 (Classification)"})
        return plan

    def build_training_recommendations(self, df: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
        """Zwraca zestaw rekomendowanych modeli i metryk + bazowe hyperparametry."""
        problem = self.detect_problem(df, target)
        if problem == "classification":
            models = [
                {"name": "RandomForestClassifier", "params": {"n_estimators": 300, "max_depth": None, "class_weight": "balanced"}},
                {"name": "XGBClassifier", "params": {"n_estimators": 400, "max_depth": 6, "subsample": 0.9}},
                {"name": "LGBMClassifier", "params": {"n_estimators": 500, "num_leaves": 64}},
                {"name": "CatBoostClassifier", "params": {"iterations": 500, "depth": 6, "verbose": 0}},
                {"name": "HistGradientBoostingClassifier", "params": {}},
                {"name": "LogisticRegression", "params": {"max_iter": 1000, "class_weight": "balanced"}}
            ]
            metrics = ["f1_weighted", "roc_auc_ovr", "accuracy", "precision_weighted", "recall_weighted"]
        else:
            models = [
                {"name": "RandomForestRegressor", "params": {"n_estimators": 400, "max_depth": None}},
                {"name": "XGBRegressor", "params": {"n_estimators": 600, "max_depth": 6, "subsample": 0.9}},
                {"name": "LGBMRegressor", "params": {"n_estimators": 800, "num_leaves": 128}},
                {"name": "CatBoostRegressor", "params": {"iterations": 800, "depth": 8, "verbose": 0}},
                {"name": "HistGradientBoostingRegressor", "params": {}},
                {"name": "ElasticNet", "params": {"l1_ratio": 0.5, "alpha": 0.001}}
            ]
            metrics = ["rmse", "mae", "r2", "mape", "smape"]
        return {
            "problem": problem,
            "target": target,
            "models": models,
            "metrics": metrics,
            "cv_folds": 5
        }


    # =======================================================================
    #                                 HELPERY
    # =======================================================================
    def _llm_json(prompt: str, openai_key: str = None, anthropic_key: str = None) -> dict:
        """Próbuje zawołać LLM i zwrócić JSON; przy błędzie zwraca pusty dict."""
        try:
            if truthy_df_safe(openai_key):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    content = resp.choices[0].message.content.strip()
                    return json.loads(content)
                except Exception:
                    pass
            if truthy_df_safe(anthropic_key):
                try:
                    from anthropic import Anthropic
                    client = Anthropic(api_key=anthropic_key)
                    msg = client.messages.create(
                        model="claude-3-5-haiku-latest",
                        max_tokens=1500,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = msg.content[0].text.strip()
                    return json.loads(content)
                except Exception:
                    pass
        except Exception:
            pass
        return {}


    def _dataset_brief(df: pd.DataFrame, target: Optional[str]) -> str:
        cols = []
        for c in df.columns[:50]:
            t = str(df[c].dtype)
            nuniq = int(df[c].nunique())
            cols.append(f"{c}({t}, uniq={nuniq})")
        return "Columns: " + ", ".join(cols) + (f" | target={target}" if target else "")


    def _speed_profile(df: pd.DataFrame) -> str:
        # Heurystyka prędkości względem rozmiaru
        rows, cols = df.shape
        if rows > 150_000 or cols > 200:
            return "ultra_fast"
        if rows > 50_000 or cols > 80:
            return "fast"
        return "balanced"