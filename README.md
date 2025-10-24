
# THE MOST IMPORTANT VARIABLES - SETUP INSTRUCTIONS
# Marksio AI Solutions - Advanced ML Platform v2.0 Pro


# 🚀 OPCJA 1: SZYBKI START (LOKALNIE)


# 1. Klonuj repozytorium
git clone https://github.com/Marksio90/The-Most-Important-Variable.git
cd The-Most-Important-Variable

# 2. Utwórz środowisko wirtualne
python -m venv venv

# Aktywuj środowisko:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Zainstaluj zależności
pip install --upgrade pip
pip install -r requirements.txt

# 4. [OPCJONALNIE] Zainstaluj zaawansowane ML biblioteki
pip install xgboost lightgbm catboost optuna

# 5. Konfiguracja (opcjonalna)
cp config/env.template config/.env
# Edytuj config/.env jeśli potrzebujesz AI integration

# 6. Uruchom aplikację
streamlit run app.py

# Aplikacja będzie dostępna na: http://localhost:8501


# 🐍 OPCJA 2: CONDA/MAMBA (ZALECANE DLA ML)


# 1. Zainstaluj Mamba (szybsza alternatywa dla Conda)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# 2. Klonuj repo
git clone https://github.com/Marksio90/The-Most-Important-Variable.git
cd The-Most-Important-Variable

# 3. Utwórz środowisko z environment.yml
mamba env create -f environment.yml

# 4. Aktywuj środowisko
mamba activate tmiv-ml-platform

# 5. Uruchom aplikację
streamlit run app.py


# 🐳 OPCJA 3: DOCKER (PRODUKCJA)


# 1. Klonuj repozytorium
git clone https://github.com/Marksio90/The-Most-Important-Variable.git
cd The-Most-Important-Variable

# 2. Zbuduj Docker image
docker build -t tmiv-ml-platform:2.0 .

# 3. Uruchom kontener
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-api-key" \
  -e ANTHROPIC_API_KEY="your-api-key" \
  -v $(pwd)/data:/app/data \
  tmiv-ml-platform:2.0

# Lub użyj Docker Compose dla pełnego stacku:
docker-compose up -d


# ☁️ OPCJA 4: CLOUD DEPLOYMENT


# === RAILWAY DEPLOYMENT ===
# 1. Zaloguj się na railway.app
# 2. Stwórz nowy projekt z GitHub repo
# 3. Ustaw zmienne środowiskowe:
#    - OPENAI_API_KEY (opcjonalnie)
#    - ANTHROPIC_API_KEY (opcjonalnie)
# 4. Deploy automatycznie po push do main

# === STREAMLIT CLOUD ===
# 1. Połącz z GitHub na share.streamlit.io
# 2. Wybierz repo i branch main
# 3. Main file: app.py
# 4. Deploy!

# === HEROKU DEPLOYMENT ===
# 1. Zainstaluj Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# 2. Zaloguj się
heroku login

# 3. Stwórz aplikację
heroku create your-tmiv-app-name

# 4. Ustaw zmienne
heroku config:set OPENAI_API_KEY=your-key
heroku config:set ANTHROPIC_API_KEY=your-key

# 5. Deploy
git push heroku main

# === AWS/GCP/AZURE ===
# Użyj Kubernetes manifest z plików konfiguracyjnych
kubectl apply -f k8s/


# ⚙️ KONFIGURACJA ZAAWANSOWANA


# === BAZA DANYCH (OPCJONALNIE) ===
# 1. PostgreSQL lokalnie
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createdb tmiv_db

# 2. Ustaw w config/.env:
DATABASE_URL=postgresql://postgres:password@localhost:5432/tmiv_db

# === AI INTEGRATION ===
# Dodaj do config/.env:
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# === MONITORING (PRODUKCJA) ===
# 1. Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# 2. Dostęp:
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090


# 🧪 ROZWÓJ I TESTOWANIE


# === SETUP DEV ENVIRONMENT ===
# 1. Zainstaluj dev dependencies
pip install -r requirements-dev.txt

# 2. Setup pre-commit hooks
pre-commit install

# 3. Uruchom testy
pytest tests/ -v --cov=backend --cov=frontend

# 4. Formatowanie kodu
black .
isort .
flake8 .

# === HOT RELOAD DEVELOPMENT ===
# Uruchom z auto-reload:
streamlit run app.py --server.fileWatcherType poll


# 🔍 TROUBLESHOOTING


# === Problem: ImportError dla zaawansowanych ML ===
# Rozwiązanie:
pip install xgboost lightgbm catboost
# Lub pomijn zaawansowane algorytmy - aplikacja działa bez nich

# === Problem: Memory Error dla dużych plików ===
# Rozwiązania:
# 1. Zwiększ limit w config/.env:
MAX_FILE_SIZE=500  # MB

# 2. Użyj sampling:
# W kodzie automatycznie sample dla >15k wierszy

# === Problem: Streamlit port już zajęty ===
# Rozwiązanie:
streamlit run app.py --server.port 8502

# === Problem: Powolne działanie ===
# Rozwiązania:
# 1. Użyj "fast" algorithm strategy
# 2. Wyłącz hyperparameter tuning  
# 3. Użyj sampling dla dużych zbiorów

# === Problem: Docker build fails ===
# Rozwiązanie:
# Usuń problematyczne biblioteki z requirements.txt:
# tensorflow, torch (jeśli nieużywane)


# 📊 PRZYKŁADY UŻYCIA


# === BASIC USAGE ===
# 1. Otwórz http://localhost:8501
# 2. Wczytaj dane (CSV/JSON lub przykładowy zbiór)
# 3. Przejdź do "Trening Modelu"
# 4. Wybierz target column i kliknij "Rozpocznij trening"
# 5. Zobacz wyniki w "Wyniki i Wizualizacje"
# 6. Przeczytaj rekomendacje w sekcji "Rekomendacje"

# === ADVANCED USAGE ===
# 1. Skonfiguruj AI API keys dla lepszych insights
# 2. Użyj "advanced" algorithm strategy
# 3. Włącz hyperparameter tuning
# 4. Eksperymentuj z ensemble mode
# 5. Eksportuj wszystkie wyniki dla dokumentacji

# === PRODUCTION USAGE ===
# 1. Deploy na cloud (Railway/Streamlit Cloud/Heroku)
# 2. Setup monitoring (Prometheus/Grafana)
# 3. Configure database dla persistencji
# 4. Setup automated backups
# 5. Monitor performance i retrain modele


# 📞 SUPPORT I COMMUNITY


# === DOKUMENTACJA ===
# - README.md - podstawowe info
# - docs/ - szczegółowa dokumentacja
# - W aplikacji: sekcja "Dokumentacja" - kompletny guide

# === ISSUES I BUGS ===
# GitHub Issues: https://github.com/Marksio90/The-Most-Important-Variable/issues

# === KONTAKT ===
# Email: marks.mateusz@wp.pl
# LinkedIn: linkedin.com/in/marksio (przykład)
# Website: marksio.ai (przykład)

# === CONTRIBUTING ===
# 1. Fork the repo
# 2. Create feature branch: git checkout -b feature/amazing-feature
# 3. Commit changes: git commit -m 'Add amazing feature'  
# 4. Push branch: git push origin feature/amazing-feature
# 5. Open Pull Request
