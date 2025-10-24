"""
TMIV v3.0 ULTRA PRO - Main Entry Point
FastAPI Backend Application

Run with:
    python backend/main.py
    
Or with uvicorn:
    uvicorn backend.app:app --reload --port 8000
"""

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🚀 Starting TMIV v3.0 ULTRA PRO")
    logger.info("=" * 80)
    
    # Run FastAPI with uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )