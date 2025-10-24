"""
TMIV v3.0 ULTRA PRO - Main Entry Point
FastAPI Backend Application

Run with:
    python backend/main.py
    python -m backend.main
    
Or with uvicorn:
    uvicorn backend.app:app --reload --port 8000
"""

import sys
from pathlib import Path

# FIX dla Windows: Add project root to Python path
# To pozwala na import backend.app
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"🐍 Python path: {sys.path[0]}")
    
    # Run FastAPI with uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )