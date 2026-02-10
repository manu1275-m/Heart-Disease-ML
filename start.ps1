# Heart Disease Prediction API Server Launcher
# Automatically starts the server and opens the browser

Write-Host "🚀 Starting Heart Disease Prediction Server..." -ForegroundColor Green
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptPath ".venv\Scripts\python.exe"
$runScript = Join-Path $scriptPath "run_server.py"

# Check if virtual environment exists
if (-not (Test-Path $pythonExe)) {
    Write-Host "❌ Error: Virtual environment not found at .venv" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Check if run_server.py exists
if (-not (Test-Path $runScript)) {
    Write-Host "❌ Error: run_server.py not found" -ForegroundColor Red
    exit 1
}

# Run the server launcher
& $pythonExe $runScript
