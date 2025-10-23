"""
Integracja z modelami LLM (OpenAI, Anthropic).

Funkcjonalności:
- Opisy kolumn
- Rekomendacje biznesowe
- Interpretacja wyników
- Fallbacki deterministyczne
- Bezpieczne cachowanie
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backend.cache_manager import get_cache_manager
from backend.error_handler import AIIntegrationException, handle_errors
from backend.security_manager import SecurityManager
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
security = SecurityManager()


class AIIntegration:
    """
    Klasa zarządzająca integracją z LLM.
    
    Obsługuje:
    - OpenAI GPT
    - Anthropic Claude
    - Deterministyczne fallbacki
    """

    def __init__(self):
        """Inicjalizacja integracji AI."""
        self.cache_manager = get_cache_manager()
        self.openai_available = False
        self.anthropic_available = False
        self._init_clients()

    def _init_clients(self):
        """Inicjalizuje klientów API."""
        import streamlit as st
        
        # Sprawdź w session_state najpierw
        openai_key = None
        anthropic_key = None
        
        if 'openai_api_key' in st.session_state:
            openai_key = st.session_state['openai_api_key']
        elif settings.openai_api_key:
            openai_key = settings.openai_api_key
        
        if 'anthropic_api_key' in st.session_state:
            anthropic_key = st.session_state['anthropic_api_key']
        elif settings.anthropic_api_key:
            anthropic_key = settings.anthropic_api_key
        
        # OpenAI
        if openai_key:
            try:
                import openai
                openai.api_key = openai_key
                self.openai_available = True
                logger.info("OpenAI client zainicjalizowany")
            except ImportError:
                logger.warning("Biblioteka openai nie jest zainstalowana")
            except Exception as e:
                logger.warning(f"Błąd inicjalizacji OpenAI: {e}")

        # Anthropic
        if anthropic_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(
                    api_key=anthropic_key
                )
                self.anthropic_available = True
                logger.info("Anthropic client zainicjalizowany")
            except ImportError:
                logger.warning("Biblioteka anthropic nie jest zainstalowana")
            except Exception as e:
                logger.warning(f"Błąd inicjalizacji Anthropic: {e}")

    def is_available(self) -> bool:
        """Sprawdza czy jakikolwiek LLM jest dostępny."""
        return self.openai_available or self.anthropic_available

    def get_provider_status(self) -> Dict[str, bool]:
        """Zwraca status providerów."""
        return {
            "openai": self.openai_available,
            "anthropic": self.anthropic_available,
            "any": self.is_available(),
        }

    @handle_errors(show_in_ui=False, default_return={})
    def describe_columns(
        self,
        df: pd.DataFrame,
        df_hash: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, str]:
        """
        Generuje opisy kolumn używając LLM lub fallbacku.

        Args:
            df: DataFrame do analizy
            df_hash: Hash DataFrame dla cache
            use_cache: Czy użyć cache

        Returns:
            Dict[str, str]: Słownik {kolumna: opis}

        Example:
            >>> ai = AIIntegration()
            >>> df = pd.DataFrame({'age': [25, 30], 'income': [50000, 60000]})
            >>> descriptions = ai.describe_columns(df)
            >>> 'age' in descriptions
            True
        """
        # Sprawdź cache
        if use_cache and df_hash:
            cached = self.cache_manager.get_column_descriptions(df_hash)
            if cached:
                logger.info("Opisy kolumn wczytane z cache")
                return cached

        descriptions = {}

        if self.is_available():
            # Użyj LLM
            descriptions = self._describe_columns_llm(df)
        else:
            # Fallback deterministyczny
            logger.info("Brak dostępnego LLM - użycie fallbacku deterministycznego")
            descriptions = self._describe_columns_fallback(df)

        # Zapisz do cache
        if use_cache and df_hash and descriptions:
            self.cache_manager.cache_column_descriptions(df_hash, descriptions)

        return descriptions

    def _describe_columns_llm(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Opisuje kolumny używając LLM.

        Args:
            df: DataFrame do analizy

        Returns:
            Dict[str, str]: Opisy kolumn
        """
        # Przygotuj informacje o kolumnach
        column_info = self._prepare_column_info(df)

        # Przygotuj prompt
        prompt = self._create_column_description_prompt(column_info)

        # Sanityzacja promptu
        prompt = security.sanitize_prompt(prompt)

        # Wywołaj LLM z retry
        descriptions = self._call_llm_with_retry(prompt, max_retries=3)

        if not descriptions:
            logger.warning("LLM nie zwrócił opisów - użycie fallbacku")
            return self._describe_columns_fallback(df)

        return descriptions

    def _prepare_column_info(self, df: pd.DataFrame, max_samples: int = 5) -> List[Dict]:
        """
        Przygotowuje informacje o kolumnach dla LLM.

        Args:
            df: DataFrame
            max_samples: Liczba przykładowych wartości

        Returns:
            List[Dict]: Lista informacji o kolumnach
        """
        column_info = []

        for col in df.columns:
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
                "n_missing": int(df[col].isna().sum()),
                "pct_missing": float(df[col].isna().sum() / len(df) * 100),
            }

            # Przykładowe wartości (nie-null)
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample_values = non_null_values.head(max_samples).tolist()
                info["sample_values"] = [str(v) for v in sample_values]

            # Statystyki dla numerycznych
            if pd.api.types.is_numeric_dtype(df[col]):
                info["min"] = float(df[col].min()) if df[col].notna().any() else None
                info["max"] = float(df[col].max()) if df[col].notna().any() else None
                info["mean"] = float(df[col].mean()) if df[col].notna().any() else None

            column_info.append(info)

        return column_info

    def _create_column_description_prompt(self, column_info: List[Dict]) -> str:
        """
        Tworzy prompt dla LLM do opisania kolumn.

        Args:
            column_info: Informacje o kolumnach

        Returns:
            str: Prompt
        """
        prompt = """Jesteś ekspertem ds. analizy danych. Przeanalizuj poniższe kolumny i wygeneruj krótki, zwięzły opis dla każdej z nich (max 100 znaków).

Opis powinien zawierać:
- Co reprezentuje kolumna
- Typ danych
- Ewentualne zastosowanie biznesowe

Format odpowiedzi: JSON z kluczami jako nazwy kolumn i wartościami jako opisy.

Kolumny do analizy:
"""

        for col_info in column_info:
            prompt += f"\n- {col_info['name']}: typ={col_info['dtype']}, unikalne={col_info['n_unique']}"
            if 'sample_values' in col_info:
                prompt += f", przykłady={col_info['sample_values'][:3]}"

        prompt += "\n\nZwróć TYLKO JSON w formacie: {\"nazwa_kolumny\": \"opis\", ...}"

        return prompt

    def _call_llm_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        timeout: int = 30
    ) -> Dict[str, str]:
        """
        Wywołuje LLM z mechanizmem retry.

        Args:
            prompt: Prompt do wysłania
            max_retries: Liczba prób
            timeout: Timeout w sekundach

        Returns:
            Dict[str, str]: Opisy kolumn
        """
        for attempt in range(max_retries):
            try:
                if self.openai_available:
                    return self._call_openai(prompt, timeout)
                elif self.anthropic_available:
                    return self._call_anthropic(prompt, timeout)
            except Exception as e:
                logger.warning(f"Próba {attempt + 1}/{max_retries} nie powiodła się: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.error("Wszystkie próby wywołania LLM nie powiodły się")
        return {}

    def _call_openai(self, prompt: str, timeout: int) -> Dict[str, str]:
        """Wywołuje OpenAI API."""
        import openai
        import json

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem ds. analizy danych."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=settings.max_llm_tokens,
            temperature=0.3,
            timeout=timeout,
        )

        content = response.choices[0].message.content
        # Parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Próba wyciągnięcia JSON z tekstu
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}

    def _call_anthropic(self, prompt: str, timeout: int) -> Dict[str, str]:
        """Wywołuje Anthropic API."""
        import json

        message = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=settings.max_llm_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )

        content = message.content[0].text

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}

    def _describe_columns_fallback(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Deterministyczny fallback dla opisów kolumn.

        Args:
            df: DataFrame

        Returns:
            Dict[str, str]: Opisy kolumn
        """
        descriptions = {}

        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            n_missing = df[col].isna().sum()

            # Bazowy opis na podstawie typu
            if pd.api.types.is_numeric_dtype(dtype):
                desc = f"Kolumna numeryczna ({dtype})"
                if n_unique < 10:
                    desc += f", {n_unique} unikalnych wartości"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                desc = "Kolumna daty/czasu"
            elif pd.api.types.is_bool_dtype(dtype):
                desc = "Kolumna boolean (prawda/fałsz)"
            else:
                desc = f"Kolumna tekstowa"
                if n_unique < 20:
                    desc += f", {n_unique} kategorii"

            # Dodaj info o brakach
            if n_missing > 0:
                pct = n_missing / len(df) * 100
                desc += f", {pct:.1f}% braków"

            descriptions[col] = desc

        return descriptions

    @handle_errors(show_in_ui=False, default_return="")
    def generate_business_recommendations(
        self,
        eda_summary: Dict,
        ml_results: Dict
    ) -> str:
        """
        Generuje rekomendacje biznesowe na podstawie analizy.

        Args:
            eda_summary: Podsumowanie EDA
            ml_results: Wyniki modelowania

        Returns:
            str: Rekomendacje biznesowe

        Example:
            >>> ai = AIIntegration()
            >>> recommendations = ai.generate_business_recommendations({}, {})
            >>> isinstance(recommendations, str)
            True
        """
        if self.is_available():
            return self._generate_recommendations_llm(eda_summary, ml_results)
        else:
            return self._generate_recommendations_fallback(eda_summary, ml_results)

    def _generate_recommendations_llm(self, eda_summary: Dict, ml_results: Dict) -> str:
        """Generuje rekomendacje używając LLM."""
        prompt = f"""Na podstawie poniższych wyników analizy danych i modelowania ML, wygeneruj 3-5 konkretnych rekomendacji biznesowych.

EDA Summary:
- Liczba wierszy: {eda_summary.get('n_rows', 'N/A')}
- Liczba kolumn: {eda_summary.get('n_columns', 'N/A')}
- Brakujące wartości: {eda_summary.get('total_missing_cells', 0)}

ML Results:
- Najlepszy model: {ml_results.get('best_model_name', 'N/A')}
- Metryka: {ml_results.get('best_metric', 'N/A')}

Rekomendacje powinny być:
1. Konkretne i wykonalne
2. Oparte na danych
3. Zorientowane biznesowo

Format: Lista numerowana, każda rekomendacja w 1-2 zdaniach."""

        prompt = security.sanitize_prompt(prompt)
        response = self._call_llm_with_retry(prompt)

        # Dla rekomendacji oczekujemy tekstu, nie JSON
        if isinstance(response, dict):
            return str(response)
        return str(response) if response else self._generate_recommendations_fallback(eda_summary, ml_results)

    def _generate_recommendations_fallback(self, eda_summary: Dict, ml_results: Dict) -> str:
        """Deterministyczny fallback dla rekomendacji."""
        recommendations = []

        # Rekomendacje na podstawie EDA
        missing_pct = eda_summary.get('total_missing_cells', 0) / (
            eda_summary.get('n_rows', 1) * eda_summary.get('n_columns', 1)
        ) * 100

        if missing_pct > 10:
            recommendations.append(
                f"1. Analiza brakujących danych ({missing_pct:.1f}%): "
                "Rozważ imputację lub usunięcie kolumn z dużą liczbą braków."
            )

        # Rekomendacje na podstawie ML
        best_model = ml_results.get('best_model_name', '')
        if best_model:
            recommendations.append(
                f"2. Model {best_model} osiągnął najlepsze wyniki. "
                "Rozważ jego wdrożenie w środowisku produkcyjnym."
            )

        if not recommendations:
            recommendations.append(
                "1. Dataset wygląda na kompletny - kontynuuj analizę i wdrożenie modelu."
            )

        return "\n".join(recommendations)


# Singleton instance
_ai_integration: Optional[AIIntegration] = None


def get_ai_integration() -> AIIntegration:
    """
    Zwraca singleton instancji AIIntegration.

    Returns:
        AIIntegration: Instancja AI integration
    """
    global _ai_integration
    if _ai_integration is None:
        _ai_integration = AIIntegration()
    return _ai_integration