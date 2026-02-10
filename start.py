#!/usr/bin/env python
"""
Startup script to launch the Heart Disease Prediction app.
Starts both backend and frontend servers, then opens the app in the browser.
"""

import subprocess
import time
import webbrowser
import sys
from pathlib import Path

def main():
    root = Path(__file__).parent
    
    print("🚀 Starting Heart Disease Prediction App...")
    
    # Start backend server
    print("📡 Starting backend API on http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--reload",
            "--reload-dir",
            str(root / "backend"),
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start frontend server
    print("🌐 Starting frontend server on http://localhost:5500...")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "5500"],
        cwd=root / "frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a moment for servers to start
    print("⏳ Waiting for servers to initialize...")
    time.sleep(3)
    
    # Open app in browser
    app_url = "http://localhost:5500"
    print(f"🌍 Opening app in browser: {app_url}")
    webbrowser.open(app_url)
    
    print("\n✅ App is running!")
    print(f"   Frontend: {app_url}")
    print(f"   API: http://127.0.0.1:8000")
    print("\nPress Ctrl+C to stop all servers...")
    
    try:
        # Keep processes running
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("✅ Servers stopped.")

if __name__ == "__main__":
    main()
