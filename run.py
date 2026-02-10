#!/usr/bin/env python3
"""
Launch script: starts FastAPI backend and opens frontend in browser.
"""
import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).parent
    frontend_dir = repo_root / "frontend"
    
    print("Starting FastAPI backend...")
    # Start backend in background
    backend_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--reload",
            "--reload-dir",
            str(repo_root / "backend"),
        ],
        cwd=repo_root
    )
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    print("Starting frontend HTTP server...")
    # Start frontend in background
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "5500"],
        cwd=frontend_dir
    )
    
    # Wait a moment for frontend to start
    time.sleep(2)
    
    print("Opening browser...")
    webbrowser.open("http://localhost:5500")
    
    print("\n" + "="*60)
    print("Backend running at: http://127.0.0.1:8000")
    print("Frontend running at: http://localhost:5500")
    print("Press Ctrl+C to stop both servers")
    print("="*60 + "\n")
    
    try:
        # Keep processes running
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("Done.")

if __name__ == "__main__":
    main()
