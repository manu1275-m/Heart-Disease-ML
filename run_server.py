"""
FastAPI Server Launcher with Auto Browser Opening
Starts both backend API server and frontend static server, then opens the frontend in browser
"""
import subprocess
import webbrowser
import time
import sys
import os
from pathlib import Path
import http.server
import socketserver
import threading

def start_frontend_server(port=5500):
    """Start a simple HTTP server for the frontend"""
    frontend_dir = Path(__file__).parent / "frontend"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_dir), **kwargs)
        
        def log_message(self, format, *args):
            # Suppress request logs to keep output clean
            pass
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"🌐 Frontend server running on http://127.0.0.1:{port}")
        httpd.serve_forever()

def _env_int(var_name: str, default: int) -> int:
    try:
        return int(os.getenv(var_name, default))
    except ValueError:
        return default


def _env_bool(var_name: str, default: bool) -> bool:
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    project_root = Path(__file__).parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    # Server configuration from env with safe fallbacks
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = _env_int("BACKEND_PORT", 8000)
    frontend_port = _env_int("FRONTEND_PORT", 5500)
    open_browser = _env_bool("OPEN_BROWSER", True)
    
    backend_url = f"http://{backend_host}:{backend_port}"
    frontend_url = f"http://127.0.0.1:{frontend_port}"
    
    print(f"🚀 Starting Heart Disease Prediction System...")
    print(f"📍 Backend API: {backend_url}")
    print(f"📚 API Docs: {backend_url}/docs")
    print(f"🌐 Frontend: {frontend_url}")
    print(f"⏳ Starting servers...\n")
    
    # Start the backend API server in a subprocess
    backend_process = subprocess.Popen(
        [
            python_exe,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--reload",
            "--reload-dir",
            str(project_root / "backend"),
            "--host",
            backend_host,
            "--port",
            str(backend_port)
        ],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Start the frontend server in a separate thread
    frontend_thread = threading.Thread(
        target=start_frontend_server,
        args=(frontend_port,),
        daemon=True
    )
    frontend_thread.start()
    time.sleep(0.5)  # Give frontend server time to start
    
    # Wait for backend to be ready and open browser
    browser_opened = False
    try:
        if backend_process.stdout:
            for line in backend_process.stdout:
                print(line, end='')
                
                # Open browser when backend server is ready
                if not browser_opened and "Application startup complete" in line:
                    time.sleep(0.5)  # Small delay to ensure server is fully ready
                    print(f"\n✅ Both servers are ready!")
                    if open_browser:
                        print(f"🎉 Opening Heart Disease Prediction App in browser...\n")
                        webbrowser.open(frontend_url)
                    else:
                        print(f"🔗 Open manually: {frontend_url}\n")
                    browser_opened = True
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Servers stopped successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
