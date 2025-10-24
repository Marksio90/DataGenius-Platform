"""
TMIV v3.0 ULTRA PRO - FastAPI Application

Complete REST API with:
- v1.0: Basic ML Pipeline
- v2.0: Advanced ML & Ensemble
- v3.0: Neural Networks, Optimization, Time Series, MLOps, Explainability
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import psutil
import platform

# Configuration
try:
    from config.settings import get_settings
    settings = get_settings()
except:
    # Fallback settings
    class Settings:
        environment = "development"
        debug = True
        app_version = "3.0.0"
        upload_dir = Path("./uploads")
        artifacts_dir = Path("./artifacts")
        logs_dir = Path("./logs")
        allowed_origins = ["http://localhost:3000", "http://localhost:3001"]
    settings = Settings()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("=" * 80)
    logger.info("🚀 TMIV v3.0 ULTRA PRO - Starting Application")
    logger.info("=" * 80)
    
    # Create directories
    settings.upload_dir.mkdir(exist_ok=True, parents=True)
    settings.artifacts_dir.mkdir(exist_ok=True, parents=True)
    settings.logs_dir.mkdir(exist_ok=True, parents=True)
    (settings.artifacts_dir / "feature_store").mkdir(exist_ok=True)
    (settings.artifacts_dir / "registry").mkdir(exist_ok=True)
    (settings.artifacts_dir / "mlruns").mkdir(exist_ok=True)
    
    logger.info(f"📁 Upload directory: {settings.upload_dir}")
    logger.info(f"📁 Artifacts directory: {settings.artifacts_dir}")
    logger.info(f"🔧 Environment: {settings.environment}")
    
    # Log features
    logger.info("=" * 80)
    logger.info("✨ ENABLED FEATURES:")
    logger.info("  ✅ v1.0 - Basic ML Pipeline")
    logger.info("  ✅ v2.0 - Advanced ML & Ensemble")
    logger.info("  ✅ v3.0 - Neural Networks")
    logger.info("  ✅ v3.0 - Hyperparameter Optimization")
    logger.info("  ✅ v3.0 - Time Series Forecasting")
    logger.info("  ✅ v3.0 - MLOps (MLflow, Drift, Registry)")
    logger.info("  ✅ v3.0 - Feature Engineering")
    logger.info("  ✅ v3.0 - Explainability (SHAP, LIME)")
    logger.info("=" * 80)
    
    # Check dependencies
    optional_deps = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'prophet': 'Prophet',
        'shap': 'SHAP',
        'optuna': 'Optuna',
        'mlflow': 'MLflow',
    }
    
    logger.info("🔍 Checking dependencies...")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            logger.info(f"  ✅ {name}")
        except ImportError:
            logger.warning(f"  ⚠️  {name} - NOT INSTALLED")
    
    logger.info("=" * 80)
    logger.info("✨ Application ready!")
    logger.info("🌐 API Docs: http://localhost:8000/docs")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="TMIV v3.0 ULTRA PRO",
    description="""
    **The Most Intelligent Visualizer** - Production-Grade AutoML Platform
    
    ## Features
    
    ### v3.0 - Enterprise Features
    - **Neural Networks**: PyTorch, TensorFlow, AutoML
    - **Optimization**: Optuna, Genetic, Bayesian
    - **Time Series**: Prophet, ARIMA, LSTM
    - **MLOps**: MLflow, Drift Detection, Model Registry
    - **Feature Engineering**: Auto Features, Selection, Store
    - **Explainability**: SHAP, LIME, What-If Analysis
    
    ## Documentation
    - API Docs: `/docs`
    - ReDoc: `/redoc`
    - Health: `/health`
    """,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.debug else "An error occurred"
        }
    )


# Import and include routers (with error handling)
logger.info("Loading API routers...")

# v3.0 endpoints (always try to load)
try:
    from backend.api.v3_endpoints import router as v3_router
    app.include_router(v3_router, tags=["v3"])
    logger.info("  ✅ v3.0 endpoints loaded")
except Exception as e:
    logger.warning(f"  ⚠️  v3.0 endpoints not available: {e}")

# v2.0 endpoints (optional)
try:
    from backend.api.v2_endpoints import router as v2_router
    app.include_router(v2_router, prefix="/api/v2", tags=["v2"])
    logger.info("  ✅ v2.0 endpoints loaded")
except Exception as e:
    logger.warning(f"  ⚠️  v2.0 endpoints not available: {e}")

# v1.0 endpoints (optional)
try:
    from backend.api.endpoints import router as v1_router
    app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
    logger.info("  ✅ v1.0 endpoints loaded")
except Exception as e:
    logger.warning(f"  ⚠️  v1.0 endpoints not available: {e}")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API information."""
    return {
        "name": "TMIV v3.0 ULTRA PRO",
        "version": "3.0.0",
        "status": "operational",
        "description": "Production-Grade AutoML Platform",
        "features": {
            "neural_networks": "PyTorch, TensorFlow, AutoML",
            "optimization": "Optuna, Genetic, Bayesian",
            "time_series": "Prophet, ARIMA, LSTM",
            "mlops": "MLflow, Drift Detection, Registry",
            "feature_engineering": "Auto Features, Selection",
            "explainability": "SHAP, LIME, What-If"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "/health",
            "version": "/version"
        }
    }


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """System health check."""
    
    # Check dependencies
    dependencies = {}
    for module in ['torch', 'tensorflow', 'prophet', 'shap', 'optuna', 'mlflow']:
        try:
            __import__(module)
            dependencies[module] = "installed"
        except ImportError:
            dependencies[module] = "not_installed"
    
    # System info
    system_info = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "memory_percent": psutil.virtual_memory().percent
    }
    
    return {
        "status": "healthy",
        "version": "3.0.0",
        "environment": settings.environment,
        "system": system_info,
        "dependencies": dependencies
    }


# Version endpoint
@app.get("/version", tags=["Version"])
async def version():
    """Version information."""
    return {
        "version": "3.0.0",
        "name": "TMIV ULTRA PRO",
        "release_date": "2024-01-20",
        "features": [
            "Neural Networks (PyTorch, TensorFlow)",
            "Hyperparameter Optimization (Optuna, Genetic, Bayesian)",
            "Time Series (Prophet, ARIMA, LSTM)",
            "MLOps (MLflow, Drift, Registry)",
            "Feature Engineering (Auto, Selection)",
            "Explainability (SHAP, LIME, What-If)"
        ]
    }


# Documentation endpoint
@app.get("/api/v3/info", tags=["Documentation"])
async def v3_info():
    """v3.0 features information."""
    return {
        "version": "3.0.0",
        "modules": {
            "neural_networks": {
                "description": "Deep learning with PyTorch and TensorFlow",
                "features": ["AutoML", "Early Stopping", "TensorBoard"]
            },
            "optimization": {
                "description": "Advanced hyperparameter tuning",
                "methods": ["Optuna (TPE)", "Genetic Algorithm", "Bayesian"]
            },
            "time_series": {
                "description": "Time series forecasting",
                "methods": ["Prophet", "ARIMA/SARIMA", "LSTM"]
            },
            "mlops": {
                "description": "Production ML operations",
                "features": ["MLflow", "Drift Detection", "Model Registry", "Auto Retraining"]
            },
            "feature_engineering": {
                "description": "Automated feature engineering",
                "features": ["Auto Generation", "Selection", "Feature Store"]
            },
            "explainability": {
                "description": "Model interpretability",
                "methods": ["SHAP", "LIME", "What-If Analysis", "Fairness"]
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)