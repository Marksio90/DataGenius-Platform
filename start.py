#!/usr/bin/env python3
"""
TMIV v3.0 ULTRA PRO - Complete Startup Script
Cross-platform Python version

Usage:
    python start.py
    python start.py --backend-only
    python start.py --no-streamlit
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import argparse
import webbrowser

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    
    @staticmethod
    def strip_if_windows():
        """Remove colors on Windows if not supported"""
        if sys.platform == 'win32':
            Colors.GREEN = Colors.BLUE = Colors.YELLOW = Colors.RED = Colors.NC = ''

Colors.strip_if_windows()

# Store process PIDs
processes = []

def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print(f"{Colors.GREEN}🚀 Starting TMIV v3.0 ULTRA PRO - Complete Stack{Colors.NC}")
    print("=" * 80)
    print()

def check_directory():
    """Check if we're in the right directory"""
    if not Path("backend/main.py").exists():
        print(f"{Colors.RED}❌ Error: Please run this script from the project root directory{Colors.NC}")
        print("   (where backend/ folder is located)")
        sys.exit(1)

def check_health(url="http://localhost:8000/health", timeout=30):
    """Check if backend is healthy"""
    import urllib.request
    import json
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=2)
            data = json.loads(response.read().decode())
            if data.get('status') == 'healthy':
                return True
        except:
            time.sleep(1)
    return False

def start_backend():
    """Start FastAPI backend"""
    print(f"{Colors.GREEN}✅ Starting Backend (FastAPI){Colors.NC}")
    print("   Port: 8000")
    print("   Docs: http://localhost:8000/docs")
    
    if sys.platform == 'win32':
        # Windows: Start in new console window
        process = subprocess.Popen(
            ['python', 'backend/main.py'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/Mac: Start in background
        process = subprocess.Popen(
            ['python', 'backend/main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    processes.append(('Backend', process))
    print(f"   PID: {process.pid}")
    print()
    
    # Wait for backend to start
    print(f"{Colors.YELLOW}⏳ Waiting for backend to start...{Colors.NC}")
    if check_health():
        print(f"{Colors.GREEN}✅ Backend is healthy!{Colors.NC}")
    else:
        print(f"{Colors.YELLOW}⚠️  Backend might still be starting...{Colors.NC}")
    print()
    
    return process

def start_frontend():
    """Start React frontend"""
    if not Path("frontend/package.json").exists():
        print(f"{Colors.YELLOW}⚠️  Frontend not found - skipping{Colors.NC}")
        print()
        return None
    
    print(f"{Colors.GREEN}✅ Starting Frontend (React){Colors.NC}")
    print("   Port: 3000")
    print("   URL: http://localhost:3000")
    
    if sys.platform == 'win32':
        # Windows
        process = subprocess.Popen(
            ['cmd', '/c', 'cd frontend && npm start'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/Mac
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd='frontend',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    processes.append(('Frontend', process))
    print(f"   PID: {process.pid}")
    print()
    
    return process

def start_streamlit():
    """Start Streamlit legacy UI"""
    streamlit_file = None
    if Path("streamlit_app.py").exists():
        streamlit_file = "streamlit_app.py"
    elif Path("app.py").exists():
        streamlit_file = "app.py"
    
    if not streamlit_file:
        print(f"{Colors.YELLOW}⚠️  Streamlit app not found - skipping{Colors.NC}")
        print()
        return None
    
    print(f"{Colors.GREEN}✅ Starting Streamlit (Legacy UI){Colors.NC}")
    print("   Port: 8501")
    print("   URL: http://localhost:8501")
    
    if sys.platform == 'win32':
        # Windows
        process = subprocess.Popen(
            ['streamlit', 'run', streamlit_file, '--server.port', '8501'],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux/Mac
        process = subprocess.Popen(
            ['streamlit', 'run', streamlit_file, '--server.port', '8501'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    processes.append(('Streamlit', process))
    print(f"   PID: {process.pid}")
    print()
    
    return process

def print_summary():
    """Print summary of running services"""
    print("=" * 80)
    print(f"{Colors.GREEN}🎉 TMIV v3.0 ULTRA PRO - ALL SERVICES STARTED!{Colors.NC}")
    print("=" * 80)
    print()
    print(f"{Colors.BLUE}📊 RUNNING SERVICES:{Colors.NC}")
    print("   🔹 Backend (FastAPI):    http://localhost:8000/docs")
    print("   🔹 API Health Check:     http://localhost:8000/health")
    
    if Path("frontend/package.json").exists():
        print("   🔹 Frontend (React):     http://localhost:3000")
    
    if Path("streamlit_app.py").exists() or Path("app.py").exists():
        print("   🔹 Streamlit (Legacy):   http://localhost:8501")
    
    print()
    print(f"{Colors.YELLOW}💡 TIPS:{Colors.NC}")
    print("   • Press Ctrl+C to stop all services")
    print("   • View logs in terminal/console windows")
    print("   • Running processes:")
    for name, process in processes:
        print(f"     - {name}: PID {process.pid}")
    print()
    print(f"{Colors.GREEN}🚀 Ready to build amazing ML applications!{Colors.NC}")
    print("=" * 80)
    print()

def cleanup(signum=None, frame=None):
    """Stop all services"""
    print()
    print(f"{Colors.YELLOW}🛑 Stopping all services...{Colors.NC}")
    
    for name, process in processes:
        try:
            process.terminate()
            print(f"   ✅ Stopped {name} (PID {process.pid})")
        except:
            pass
    
    print(f"{Colors.GREEN}✅ All services stopped{Colors.NC}")
    sys.exit(0)

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Start TMIV v3.0 ULTRA PRO')
    parser.add_argument('--backend-only', action='store_true', help='Start only backend')
    parser.add_argument('--no-frontend', action='store_true', help='Skip frontend')
    parser.add_argument('--no-streamlit', action='store_true', help='Skip Streamlit')
    parser.add_argument('--open-browser', action='store_true', help='Open browser automatically')
    args = parser.parse_args()
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, cleanup)
    
    # Print banner
    print_banner()
    
    # Check directory
    check_directory()
    
    # Start services
    start_backend()
    
    if not args.backend_only:
        if not args.no_frontend:
            start_frontend()
        
        if not args.no_streamlit:
            start_streamlit()
    
    # Print summary
    print_summary()
    
    # Open browser
    if args.open_browser:
        print("🌐 Opening browser...")
        webbrowser.open('http://localhost:8000/docs')
        print()
    
    # Keep running
    print(f"{Colors.BLUE}📝 Services running... (Press Ctrl+C to stop all){Colors.NC}")
    print()
    
    try:
        # Wait forever
        while True:
            time.sleep(1)
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"{Colors.RED}⚠️  {name} stopped unexpectedly!{Colors.NC}")
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()