FROM python:3.11-slim

WORKDIR /app

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Kopiuj pliki requirements
COPY requirements.txt .

# Instalacja pakietów Python
RUN pip install --no-cache-dir -r requirements.txt

# Kopiuj aplikację
COPY . .

# Ekspozycja portu Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Uruchomienie aplikacji
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]