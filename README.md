# 📊 TMIV – The Most Important Variables

**TMIV** (The Most Important Variables) to uniwersalna aplikacja w **Streamlit**,  
która automatycznie analizuje dane, wybiera najważniejsze cechy i trenuje modele ML w trybie „one-click”.  

Dzięki prostemu interfejsowi i zintegrowanym modułom **EDA + ML** możesz szybko sprawdzić, które zmienne mają największy wpływ na wynik – bez konieczności ręcznego kodowania.

---

## 🚀 Szybki start

```bash
# Utwórz środowisko
mamba env create -f environment.yml
# lub:
conda env create -f environment.yml

# Aktywuj środowisko
mamba activate tmiv_app

# Uruchom aplikację
streamlit run app.py
👉 Jeśli chcesz aktywować funkcje LLM (np. automatyczne opisy kolumn, rekomendacje):
utwórz plik config/.env na bazie config/env.template i uzupełnij OPENAI_API_KEY.

📂 Struktura projektu
bash
Skopiuj kod
hackathon-project/
├── app.py                # Główny plik aplikacji Streamlit
├── environment.yml       # Definicja środowiska (Conda/Mamba)
├── requirements.txt      # Lista dodatkowych zależności (pip)
│
├── config/               # Ustawienia i zmienne środowiskowe
│   ├── .env              # Dane wrażliwe (lokalnie)
│   ├── env.template      # Szablon do uzupełnienia
│   └── settings.py       # Konfiguracja (Pydantic Settings)
│
├── backend/              # Integracje (EDA/ML, upload plików)
├── frontend/             # Komponenty UI
├── ml/                   # Funkcje ML, pipeline’y
├── db/                   # Schema i narzędzia SQLite
├── data/                 # Dane przykładowe (np. avocado.csv)
│
├── eda/                  # Notebooki EDA
├── docs/                 # Dokumentacja projektu
└── qa/tests/             # Testy jednostkowe (pytest)
🏗️ Architektura aplikacji
text
Skopiuj kod
                ┌───────────────────────┐
                │       Frontend        │
                │  (Streamlit UI/UX)    │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │       Backend         │
                │  (EDA + ML logic)     │
                └──────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SQLite    │    │   Models    │    │   OpenAI /  │
│   (db/)     │    │ (ml/)       │    │   Heurystyki│
└─────────────┘    └─────────────┘    └─────────────┘
✨ Funkcje aplikacji
📂 Wczytywanie danych: CSV, JSON, Parquet lub przykładowy dataset avocado

🔎 Analiza kolumn: heurystyki lub LLM (jeśli dostępny klucz API)

🎯 Automatyczny wybór targetu

🤖 Autodetekcja problemu: regresja lub klasyfikacja

🏋️‍♂️ Trening modeli: scikit-learn (lekki, szybki), opcjonalnie PyCaret

📊 Wizualizacje: ranking cech, metryki jakości modelu

💡 Rekomendacje ulepszeń

🗂️ Historia uruchomień zapisywana w SQLite

🖼️ Przykładowy widok aplikacji
yaml
Skopiuj kod
┌───────────────────────────────────────────────┐
│  📊 TMIV – Analiza cech                      │
├───────────────────────────────────────────────┤
│ [Upload CSV]  [Przykładowe dane: avocado.csv] │
│                                               │
│ 🎯 Target:  AveragePrice                      │
│ 🔎 Problem:  Regression                       │
│                                               │
│ 📊 Najważniejsze cechy:                       │
│   1. Total Volume     ████████████  0.87      │
│   2. Year             ████████      0.65      │
│   3. 4046             ██████        0.42      │
│                                               │
│ 🏆 Wyniki: RMSE=0.29, R²=0.91                 │
└───────────────────────────────────────────────┘
⚙️ Uwaga nt. PyCaret
Ze względu na rozmiar zależności PyCaret jest opcjonalny.
Aplikacja działa w pełni w trybie scikit-learn, natomiast dla chętnych dostarczony jest pokazowy notebook:

bash
Skopiuj kod
ml/ml_training.ipynb
