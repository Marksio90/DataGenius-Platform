# TMIV - The Most Important Variables
## Advanced ML Platform v2.0 Pro 🤖

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.37+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Auto Data Scientist** - Automatyczna analiza danych i trenowanie modeli ML z zaawansowaną interpretowalnością.

![TMIV Screenshot](docs/ux_mockups/screenshot_main.png)

---

## ✨ Kluczowe Funkcjonalności

🚀 **Auto ML Pipeline**
- Automatyczna detekcja typu problemu (klasyfikacja/regresja)
- Inteligentny preprocessing i sanityzacja danych
- Trenowanie 10+ modeli równolegle

📊 **Advanced EDA**
- Statystyki opisowe
- Macierz korelacji
- Opisy kolumn powered by AI (OpenAI/Anthropic)

🤖 **Multi-Model Training**
- Sklearn: Logistic, Random Forest, Gradient Boosting
- XGBoost, LightGBM, CatBoost
- Asynchroniczne wykonanie z progress tracking

📈 **Explainability**
- Feature importance (agregowane z wielu modeli)
- ROC/PR curves, Confusion Matrix
- Calibration curves
- Residual plots (regresja)

💡 **AI Recommendations**
- Rekomendacje biznesowe
- Interpretacja wyników
- Fallback deterministyczny (działa bez kluczy API)

📦 **Production Ready**
- Eksport modeli (.joblib)
- Raporty PDF (ReportLab)
- Docker support
- CI/CD (GitHub Actions)

---

## 🚀 Quick Start

### Instalacja (Mamba/Conda)
```bash
# Klonuj repo
git clone https://github.com/your-org/tmiv.git
cd tmiv

# Utwórz środowisko
mamba env create -f environment.yml
mamba activate tmiv

# Uruchom aplikację
streamlit run app.py
```

### Instalacja (pip)
```bash
# Klonuj repo
git clone https://github.com/your-org/tmiv.git
cd tmiv

# Utwórz venv
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom
streamlit run app.py
```

### Docker
```bash
# Build
docker-compose build

# Uruchom
docker-compose up

# Aplikacja dostępna na http://localhost:8501
```

---

## 📖 Użycie

### Krok 1: Wczytaj Dane

Przejdź do zakładki **"📊 Analiza Danych"** i wgraj plik:
- CSV, Excel, Parquet lub JSON
- Przykład: `data/avocado.csv`

### Krok 2: EDA

Kliknij **"Uruchom EDA"** aby zobaczyć:
- Statystyki opisowe
- Rozkłady i korelacje
- Opisy kolumn (AI)

### Krok 3: Trening

W zakładce **"🤖 Trening Modelu"**:
1. Wybierz kolumnę target (auto-wykrywana)
2. Wybierz strategię (`balanced` zalecane)
3. Kliknij **"Rozpocznij Trening"**

### Krok 4: Wyniki

Zakładka **"📈 Wyniki i Wizualizacje"**:
- Ranking modeli
- Porównanie metryk (tabela + radar)
- Feature importance
- Wykresy (ROC, CM, etc.)

### Krok 5: Eksport

Zakładka **"💡 Rekomendacje"**:
- 📥 Pobierz ZIP (modele + artefakty)
- 📄 Pobierz PDF (raport)

---

## 🏗️ Architektura
```
┌─────────────┐
│   app.py    │  Streamlit UI
└──────┬──────┘
       │
       ├─► frontend/      UI Components
       │   ├─ ui_components.py
       │   ├─ ui_panels.py
       │   └─ ui_compare.py
       │
       └─► backend/       Business Logic
           ├─ file_upload.py
           ├─ dtype_sanitizer.py
           ├─ eda_integration.py
           ├─ ml_integration.py
           ├─ async_ml_trainer.py
           ├─ plots.py
           ├─ export_everything.py
           └─ ai_integration.py
```

---

## 🧪 Testy
```bash
# Uruchom wszystkie testy
pytest qa/tests/ -v

# Z coverage
pytest qa/tests/ --cov=backend --cov-report=html

# Smoke testy
pytest qa/tests/test_pipeline_smoke.py -v

# Testy kontraktów
pytest qa/tests/test_api_contracts.py -v
```

---

## 📚 Dokumentacja

Pełna dokumentacja: [docs/README.md](docs/README.md)

- [Instalacja](docs/README.md#instalacja)
- [Użycie](docs/README.md#użycie)
- [Architektura](docs/README.md#architektura)
- [Moduły](docs/README.md#moduły)
- [Rozwój](docs/README.md#rozwój)
- [FAQ](docs/README.md#faq)

---

## 🤝 Wkład

Contributions welcome! Zobacz [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork repo
2. Utwórz branch (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Otwórz Pull Request

---

## 📄 Licencja

MIT License - zobacz [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Streamlit** - Amazing UI framework
- **Scikit-learn** - ML foundation
- **XGBoost, LightGBM, CatBoost** - Gradient boosting
- **ReportLab** - PDF generation
- Open Source Community ❤️

---

## 📞 Kontakt

- **Email:** support@tmiv.ai
- **GitHub:** https://github.com/your-org/tmiv
- **Issues:** https://github.com/your-org/tmiv/issues

---

**Made with ❤️ for Data Scientists**

⭐ Star us on GitHub if you find this useful!