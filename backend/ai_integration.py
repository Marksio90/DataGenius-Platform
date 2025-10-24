import os
import json
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# ======================
# DODANE: bezpieczne klucze i smart error handling
# ======================
from backend.security_manager import credential_manager
from backend.error_handler import SmartErrorHandler


class AIDescriptionGenerator:
    """Generator opisów kolumn i rekomendacji ML z użyciem OpenAI/Anthropic + fallback symulacyjny."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
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
                if credential_manager:
                    openai_key = credential_manager.get_api_key("openai")
            except Exception:
                pass

            if not openai_key:
                openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")

            if openai_key:
                try:
                    import openai
                    self.openai_client = openai.OpenAI(api_key=openai_key)
                except ImportError:
                    st.warning("Biblioteka openai nie jest zainstalowana. Użyj: pip install openai")
                except Exception as e:
                    st.warning(f"Problem z inicjalizacją OpenAI: {e}")

            # ---- Anthropic (priorytet: credential_manager -> ENV -> session_state)
            anthropic_key = None
            try:
                if credential_manager:
                    anthropic_key = credential_manager.get_api_key("anthropic")
            except Exception:
                pass

            if not anthropic_key:
                anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_api_key")

            if anthropic_key:
                try:
                    import anthropic
                    self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                except ImportError:
                    st.warning("Biblioteka Anthropic nie jest zainstalowana. Użyj: pip install anthropic")
                except Exception as e:
                    st.warning(f"Problem z inicjalizacją Anthropic: {e}")

        except Exception as e:
            st.warning(f"Błąd podczas inicjalizacji klientów AI: {e}")

    # ======================
    # Opisy kolumn
    # ======================
    def generate_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generuje opisy kolumn używając AI lub fallback do symulacji."""
        if self.openai_client:
            return self._generate_with_openai(df)
        elif self.anthropic_client:
            return self._generate_with_anthropic(df)
        else:
            return self._generate_with_simulation(df)

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

        prompt = f"""
Przeanalizuj poniższy zbiór danych i wygeneruj krótkie, biznesowe opisy dla każdej kolumny.

Informacje o kolumnach:
{chr(10).join(column_info)}

Przykładowe dane:
{sample_data}

Dla każdej kolumny napisz zwięzły opis (1-2 zdania) wyjaśniający:
- Co reprezentuje ta kolumna
- Jaki ma typ danych
- Czy są jakieś szczególne cechy (brakujące dane, outliers, etc.)

Odpowiedz w formacie JSON: {{"nazwa_kolumny": "opis", ...}}
        """.strip()

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem data science. Generujesz zwięzłe, praktyczne opisy kolumn danych."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            # Bezpieczne parsowanie JSON
            try:
                descriptions = json.loads(content)
                if not isinstance(descriptions, dict):
                    raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            except Exception:
                # Spróbuj wyłuskać blok JSON jeśli model zwrócił tekst z dodatkami
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    descriptions = json.loads(content[start: end + 1])
                else:
                    raise
            st.success("✅ Opisy wygenerowane przez OpenAI GPT")
            return {str(k): str(v) for k, v in descriptions.items()}
        except Exception as e:
            st.warning(f"Błąd OpenAI API lub parsowania JSON: {e}. Używam symulacji.")
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

        prompt = f"""
Przeanalizuj poniższy zbiór danych i wygeneruj krótkie, biznesowe opisy dla każdej kolumny.

Informacje o kolumnach:
{chr(10).join(column_info)}

Przykładowe dane:
{sample_data}

Dla każdej kolumny napisz zwięzły opis (1-2 zdania) wyjaśniający co reprezentuje ta kolumna.
Odpowiedz w formacie JSON: {{"nazwa_kolumny": "opis", ...}}
        """.strip()

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text if hasattr(response, "content") else str(response)
            try:
                descriptions = json.loads(content)
                if not isinstance(descriptions, dict):
                    raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    descriptions = json.loads(content[start: end + 1])
                else:
                    raise
            st.success("✅ Opisy wygenerowane przez Anthropic Claude")
            return {str(k): str(v) for k, v in descriptions.items()}
        except Exception as e:
            st.warning(f"Błąd Anthropic API lub parsowania JSON: {e}. Używam symulacji.")
            return self._generate_with_simulation(df)

    def _generate_with_simulation(self, df: pd.DataFrame) -> Dict[str, str]:
        """Fallback – opisy z prostej analizy statystycznej."""
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
                min_val = pd.to_numeric(col_data, errors="coerce").min()
                max_val = pd.to_numeric(col_data, errors="coerce").max()
                mean_val = pd.to_numeric(col_data, errors="coerce").mean()

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
        name_lower = column_name.lower()
        
        # Sprawdź nazwę
        if any(indicator in name_lower for indicator in name_indicators):
            return True
        
        # Sprawdź charakterystykę danych (wysokie unique ratio)
        unique_ratio = col_data.nunique() / len(col_data)
        return unique_ratio > 0.95

    def _is_year_column(self, column_name: str, col_data: pd.Series) -> bool:
        """Sprawdza czy kolumna reprezentuje rok."""
        name_indicators = ["year", "rok", "date", "data"]
        name_lower = column_name.lower()
        
        if any(indicator in name_lower for indicator in name_indicators):
            return True
        
        # Sprawdź zakres wartości (typowe lata)
        try:
            numeric_data = pd.to_numeric(col_data, errors="coerce")
            if numeric_data.notna().sum() > 0:
                min_val = numeric_data.min()
                max_val = numeric_data.max()
                return 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100
        except:
            pass
        
        return False

    def _is_binary_numeric(self, col_data: pd.Series) -> bool:
        """Sprawdza czy kolumna to binarna (0/1)."""
        unique_values = set(col_data.dropna().unique())
        return unique_values <= {0, 1} or unique_values <= {0.0, 1.0}

    # ======================
    # Rekomendacje biznesowe
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
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem ML i business intelligence. Generujesz praktyczne rekomendacje biznesowe."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.4,
            )
            content = response.choices[0].message.content
            try:
                recs = json.loads(content)
                if not isinstance(recs, dict):
                    raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    recs = json.loads(content[start: end + 1])
                else:
                    raise
            st.success("✅ Rekomendacje wygenerowane przez OpenAI")
            return recs
        except Exception as e:
            st.warning(f"Błąd OpenAI API lub parsowania JSON: {e}. Używam symulacji.")
            return self._generate_recommendations_simulation(results, target_column, problem_type)

    def _generate_recommendations_anthropic(self, results: Dict[str, Any], target_column: str, problem_type: str) -> Dict[str, Any]:
        # Implementacja analogiczna do OpenAI
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
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text if hasattr(response, "content") else str(response)
            try:
                recs = json.loads(content)
                if not isinstance(recs, dict):
                    raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    recs = json.loads(content[start: end + 1])
                else:
                    raise
            st.success("✅ Rekomendacje wygenerowane przez Anthropic Claude")
            return recs
        except Exception as e:
            st.warning(f"Błąd Anthropic API lub parsowania JSON: {e}. Używam symulacji.")
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
            "key_insights": f"Model {problem_type.lower()} dla '{target_column}' identyfikuje {top_display} jako najważniejsze predyktory.",
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
    # NAPRAWIONE Rekomendacje treningu
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

        # Analiza cech
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
  "special_considerations": ["uwaga1", "uwaga2"]
}}

Skup się na praktycznych aspektach: rozmiarze datasetu, balansie klas, outlierach, czasie treningu.
        """.strip()

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem machine learning. Analizujesz zbiory danych i dajesz konkretne rekomendacje treningu."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            try:
                recs = json.loads(content)
                if not isinstance(recs, dict):
                    raise ValueError("Odpowiedź nie jest słownikiem JSON.")
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    recs = json.loads(content[start: end + 1])
                else:
                    raise
            st.success("✅ Rekomendacje treningu wygenerowane przez OpenAI")
            return recs
        except Exception as e:
            st.warning(f"Błąd OpenAI API lub parsowania JSON: {e}. Używam symulacji.")
            return self._generate_training_recommendations_simulation(df, target_column)

    def _generate_training_recommendations_anthropic(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generuje rekomendacje treningu używając Anthropic - fallback do symulacji."""
        return self._generate_training_recommendations_simulation(df, target_column)

    # backend/ai_integration.py - ZAMIEŃ funkcję generate_training_recommendations

    def _generate_training_recommendations_simulation(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """INTELIGENTNE rekomendacje treningu z zaawansowaną analizą"""
        n_samples = len(df)
        n_features = max(len(df.columns) - 1, 0)
        missing_pct = float((df.isnull().sum().sum() / max((df.shape[0] * max(df.shape[1], 1)), 1)) * 100.0)
        
        # Analiza targetu
        target_data = df[target_column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(target_data)
        
        if is_numeric:
            unique_ratio = target_data.nunique() / len(target_data) if len(target_data) > 0 else 0
            problem_type = "regression"
            # Analiza rozkładu dla regresji
            skewness = abs(target_data.skew()) if len(target_data) > 3 else 0
            target_range = target_data.max() - target_data.min() if len(target_data) > 0 else 0
        else:
            value_counts = target_data.value_counts()
            problem_type = "classification"
            imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 1 and value_counts.min() > 0 else 1
            n_classes = len(value_counts)
        
        # INTELIGENTNA logika rekomendacji
        recommendations = {}
        special_considerations = []
        
        # === TRAIN SIZE ===
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
        
        # Dostosowanie dla niezbalansowanych danych klasyfikacyjnych
        if problem_type == "classification" and 'imbalance_ratio' in locals() and imbalance_ratio > 5:
            train_size = min(0.85, train_size + 0.05)
            train_reason += " + bonus za niezbalansowane klasy"
            special_considerations.append("Niezbalansowane klasy - warto użyć stratified sampling")
        
        # === CV FOLDS ===
        if problem_type == "classification":
            if 'n_classes' in locals() and n_classes > 10:
                cv_folds = 3
                cv_reason = "Wiele klas - 3-fold CV wystarczające"
            elif n_samples < 1000:
                cv_folds = 3
                cv_reason = "Mały zbiór - 3-fold CV ze względu na rozmiar"
            else:
                cv_folds = 5
                cv_reason = "5-fold CV - standard dla klasyfikacji"
        else:  # regression
            if n_samples > 20000:
                cv_folds = 3
                cv_reason = "Duży zbiór - 3-fold CV wystarczające"
            elif n_samples < 1000:
                cv_folds = 3
                cv_reason = "Mały zbiór - 3-fold CV"
            else:
                cv_folds = 5
                cv_reason = "5-fold CV - standard dla regresji"
        
        # === PERFORMANCE SETTINGS ===
        if n_samples > 15000:
            use_full_data = False
            performance_reason = "Duży zbiór - użyj sampling dla szybkości (8k próbek)"
            special_considerations.append("Dla dużych zbiorów automatyczne sampling do 8000 próbek")
        elif n_samples > 50000:
            use_full_data = False
            performance_reason = "Bardzo duży zbiór - zdecydowanie użyj sampling"
            special_considerations.append("Rozważ distributed training dla zbiorów >50k")
        else:
            use_full_data = True
            performance_reason = "Średni/mały zbiór - używaj wszystkich danych"
        
        # === STRATEGY RECOMMENDATIONS ===
        if n_samples < 1000:
            recommended_strategy = "all"
            strategy_reason = "Mały zbiór - przetestuj wszystkie algorytmy"
        elif n_samples < 10000:
            if problem_type == "classification" and 'imbalance_ratio' in locals() and imbalance_ratio > 3:
                recommended_strategy = "ensemble"
                strategy_reason = "Niezbalansowane klasy - ensemble methods radzą sobie lepiej"
            else:
                recommended_strategy = "accurate"
                strategy_reason = "Średni zbiór - skup się na dokładności"
        else:
            recommended_strategy = "fast"
            strategy_reason = "Duży zbiór - priorytet: szybkość treningu"
        
        # === HYPERPARAMETER TUNING ===
        if n_samples < 5000 and n_features < 50:
            enable_hyperparameter_tuning = True
            tuning_reason = "Mały/średni zbiór - hyperparameter tuning może znacznie poprawić wyniki"
        elif n_samples > 20000:
            enable_hyperparameter_tuning = False
            tuning_reason = "Duży zbiór - hyperparameter tuning będzie zbyt wolny"
        else:
            enable_hyperparameter_tuning = False
            tuning_reason = "Domyślne parametry wystarczające dla tego rozmiaru"
        
        # === METRIC RECOMMENDATION ===
        if problem_type == "classification":
            if 'imbalance_ratio' in locals() and imbalance_ratio > 3:
                recommended_metric = "f1_weighted"
                metric_reason = "Niezbalansowane klasy - F1 weighted lepiej niż accuracy"
            else:
                recommended_metric = "accuracy"
                metric_reason = "Balanced klasy - accuracy jest najlepszą metryką"
        else:  # regression
            if is_numeric and 'skewness' in locals() and skewness > 2:
                recommended_metric = "neg_mean_absolute_error"
                metric_reason = "Skewed target - MAE odporniejsze na outliers niż R²"
            else:
                recommended_metric = "r2"
                metric_reason = "R² - standard dla regresji"
        
        # === ENSEMBLE RECOMMENDATION ===
        if n_samples > 5000 and n_features > 10:
            enable_ensemble = True
            ensemble_reason = "Duży zbiór z wieloma cechami - ensemble może zwiększyć accuracy"
        else:
            enable_ensemble = False
            ensemble_reason = "Mały zbiór - pojedyncze modele wystarczające"
        
        # === DODATKOWE UWAGI ===
        if missing_pct > 20:
            special_considerations.append(f"Dużo brakujących danych ({missing_pct:.1f}%) - rozważ advanced imputation")
        
        if n_features > n_samples:
            special_considerations.append("Więcej cech niż próbek - wysokie ryzyko overfitting, użyj regularization")
        
        if n_features > 100:
            special_considerations.append("Dużo cech - rozważ feature selection lub PCA")
        
        if problem_type == "regression" and 'target_range' in locals() and target_range > 1000:
            special_considerations.append("Duży zakres target values - rozważ log transformation")
        
        # === CZAS TRENINGU ===
        if n_samples > 20000:
            special_considerations.append("Duży zbiór - trenowanie może zająć 10-30 minut")
        elif n_samples < 1000:
            special_considerations.append("Mały zbiór - trenowanie będzie bardzo szybkie (<2 min)")
        
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

import os
import json
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st

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
        if target and target in df.columns:
            y = df[target].dropna()
            # klasyfikacja jeśli ma <= 20 unikalnych wartości nieciągłych i/lub typ object/bool/categorical
            if y.dtype.name in ("object","bool","category") or (y.nunique()<=20 and not np.issubdtype(y.dtype, np.floating)):
                return "classification"
            return "regression"
        # brak targetu -> heurystyka po nazwach
        for col in df.columns:
            if col.lower() in ("target","y","label","class","price","value","score"):
                return "classification" if df[col].nunique()<=20 else "regression"
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
        plan["steps"].append({"name":"Detect target","detail": f"Detected problem: {problem}, target: {target}"})
        plan["steps"].append({"name":"Missing values","detail": f"Numeric -> median; Categorical -> most_frequent"})
        if len(numeric_cols)>0:
            plan["steps"].append({"name":"Outliers","detail": "IQR capping (Q1-1.5*IQR, Q3+1.5*IQR) for numeric features"})
            plan["steps"].append({"name":"Scaling","detail": "StandardScaler for numeric features"})
        if len(cat_cols)>0:
            plan["steps"].append({"name":"Encoding","detail": "Ordinal/One-Hot mixed: OneHot for low-cardinality (<=15), Target/Ordinal for high-cardinality"})
        plan["steps"].append({"name":"Train/Validation split","detail": "Stratified split for classification (80/20), standard split for regression"})
        plan["steps"].append({"name":"Cross-Validation","detail": "KFold=5 (Regression) / StratifiedKFold=5 (Classification)"})
        return plan

    def build_training_recommendations(self, df: pd.DataFrame, target: Optional[str]) -> Dict[str, Any]:
        """Zwraca zestaw rekomendowanych modeli i metryk + bazowe hyperparametry."""
        problem = self.detect_problem(df, target)
        if problem == "classification":
            models = [
                {"name":"RandomForestClassifier", "params":{"n_estimators":300, "max_depth":None, "class_weight":"balanced"}},
                {"name":"XGBClassifier", "params":{"n_estimators":400, "max_depth":6, "subsample":0.9}},
                {"name":"LGBMClassifier", "params":{"n_estimators":500, "num_leaves":64}},
                {"name":"CatBoostClassifier", "params":{"iterations":500, "depth":6, "verbose":0}},
                {"name":"HistGradientBoostingClassifier", "params":{}},
                {"name":"LogisticRegression", "params":{"max_iter":1000, "class_weight":"balanced"}}
            ]
            metrics = ["f1_weighted","roc_auc_ovr","accuracy","precision_weighted","recall_weighted"]
        else:
            models = [
                {"name":"RandomForestRegressor", "params":{"n_estimators":400, "max_depth":None}},
                {"name":"XGBRegressor", "params":{"n_estimators":600, "max_depth":6, "subsample":0.9}},
                {"name":"LGBMRegressor", "params":{"n_estimators":800, "num_leaves":128}},
                {"name":"CatBoostRegressor", "params":{"iterations":800, "depth":8, "verbose":0}},
                {"name":"HistGradientBoostingRegressor", "params":{}},
                {"name":"ElasticNet", "params":{"l1_ratio":0.5, "alpha":0.001}}
            ]
            metrics = ["rmse","mae","r2","mape","smape"]
        return {
            "problem": problem,
            "target": target,
            "models": models,
            "metrics": metrics,
            "cv_folds": 5
        }


def _llm_json(prompt: str, openai_key: str = None, anthropic_key: str = None) -> dict:
    """Próbuje zawołać LLM i zwrócić JSON; przy błędzie zwraca pusty dict."""
    try:
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"Return only valid JSON."},
                              {"role":"user","content":prompt}],
                    temperature=0
                )
                content = resp.choices[0].message.content.strip()
                return json.loads(content)
            except Exception:
                pass
        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                msg = client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=1500,
                    temperature=0,
                    messages=[{"role":"user","content":prompt}]
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
