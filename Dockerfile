# =============================================================================
# DOCKER DEPLOYMENT CONFIGURATION
# =============================================================================

# Dockerfile
FROM python:3.11-slim

LABEL maintainer="Marksio AI Solutions <info@marksio.ai>"
LABEL version="2.0.0"
LABEL description="The Most Important Variables - Advanced ML Platform"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

---
# =============================================================================
# DOCKER COMPOSE - FULL STACK
# =============================================================================

version: '3.8'

services:
  # Main ML Application
  ml-app:
    build: .
    container_name: tmiv-ml-app
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/tmiv_db
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    networks:
      - tmiv-network
    restart: unless-stopped

  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: tmiv-postgres
    environment:
      - POSTGRES_DB=tmiv_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - tmiv-network
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: tmiv-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - tmiv-network
    restart: unless-stopped

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: tmiv-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - ml-app
    networks:
      - tmiv-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  tmiv-network:
    driver: bridge

---
# =============================================================================
# KUBERNETES DEPLOYMENT
# =============================================================================

apiVersion: apps/v1
kind: Deployment
metadata:
  name: tmiv-ml-app
  labels:
    app: tmiv-ml-app
    version: v2.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tmiv-ml-app
  template:
    metadata:
      labels:
        app: tmiv-ml-app
        version: v2.0.0
    spec:
      containers:
      - name: ml-app
        image: marksio/tmiv-ml-app:2.0.0
        ports:
        - containerPort: 8501
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tmiv-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: tmiv-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 30

---
apiVersion: v1
kind: Service
metadata:
  name: tmiv-ml-service
spec:
  selector:
    app: tmiv-ml-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer

---
# =============================================================================
# GITHUB ACTIONS CI/CD PIPELINE
# =============================================================================

name: Deploy TMIV ML Platform

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=backend --cov=frontend --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          marksio/tmiv-ml-app:latest
          marksio/tmiv-ml-app:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Deploy to production
      uses: appleboy/ssh-action@v1.0.0
      with:
        host: ${{ secrets.PRODUCTION_HOST }}
        username: ${{ secrets.PRODUCTION_USER }}
        key: ${{ secrets.PRODUCTION_SSH_KEY }}
        script: |
          cd /opt/tmiv-ml-platform
          docker-compose pull
          docker-compose up -d --remove-orphans
          docker system prune -af

---
# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

# Prometheus monitoring config
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "tmiv_alerts.yml"

scrape_configs:
  - job_name: 'tmiv-ml-app'
    static_configs:
      - targets: ['localhost:8501']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

---
# Grafana Dashboard Config (JSON)
{
  "dashboard": {
    "id": null,
    "title": "TMIV ML Platform Monitoring",
    "tags": ["ml", "streamlit", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Model Training Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(tmiv_model_training_success_total[5m])",
            "legendFormat": "Success Rate"
          }
        ]
      },
      {
        "id": 2,
        "title": "Active Users",
        "type": "graph",
        "targets": [
          {
            "expr": "tmiv_active_sessions",
            "legendFormat": "Active Sessions"
          }
        ]
      },
      {
        "id": 3,
        "title": "Model Performance Metrics",
        "type": "table",
        "targets": [
          {
            "expr": "tmiv_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}