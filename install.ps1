# ==============================================================================
# CAD-PREPROCESS WINDOWS INSTALLER (PowerShell)
# ==============================================================================
# A one-line installer for Windows that sets up the Python environment,
# dependencies, and CLI wrappers.
# ==============================================================================

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/Harshil-Anuwadia/cad-preprocess.git"
$INSTALL_DIR = "$HOME\.local\share\cad-preprocess"
$BIN_DIR = "$HOME\.local\bin"

function Print-Step { param($msg) Write-Host "[*] $msg..." -ForegroundColor Cyan }
function Print-Success { param($msg) Write-Host "[✓] $msg" -ForegroundColor Green }
function Print-Error { param($msg) Write-Host "[✗] ERROR: $msg" -ForegroundColor Red; exit 1 }

Write-Host "================================================================================" -ForegroundColor Magenta
Write-Host "          CAD-PREPROCESS — Windows Installation Utility" -ForegroundColor Magenta
Write-Host "================================================================================" -ForegroundColor Magenta

# 1. Check for Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Print-Error "Python not found. Please install Python 3.9+ from python.org and add it to your PATH."
}

# 2. Check for Git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Print-Error "Git not found. Please install Git for Windows."
}

# 3. Setup Directories
Print-Step "Preparing installation path: $INSTALL_DIR"
if (!(Test-Path $INSTALL_DIR)) { New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null }
if (!(Test-Path $BIN_DIR)) { New-Item -ItemType Directory -Path $BIN_DIR -Force | Out-Null }

# 4. Clone or Update
if (Test-Path "$INSTALL_DIR\.git") {
    Print-Step "Updating source code"
    Set-Location $INSTALL_DIR
    git pull -q
} else {
    Print-Step "Downloading source code"
    git clone -q $REPO_URL $INSTALL_DIR
}

# 5. Virtual Environment
Print-Step "Setting up Python virtual environment"
Set-Location $INSTALL_DIR
python -m venv venv
$VENV_PYTHON = "$INSTALL_DIR\venv\Scripts\python.exe"
$VENV_PIP = "$INSTALL_DIR\venv\Scripts\pip.exe"

# 6. Install Package
Print-Step "Installing cad-preprocess [Full Feature Set]"
& $VENV_PIP install -q --upgrade pip
& $VENV_PIP install -q ".[explorer,performance]"
Print-Success "Package installed successfully"

# 7. Create Wrappers
Print-Step "Creating command wrappers"
function Create-Wrapper {
    param($name, $module)
    $path = "$BIN_DIR\$name.bat"
    "@echo off`nset PYTHONPATH=$INSTALL_DIR\src;%PYTHONPATH%`n`"$VENV_PYTHON`" -m $module %*" | Out-File -FilePath $path -Encoding ascii
}

Create-Wrapper "cad-preprocess" "cad_preprocess.cli"
Create-Wrapper "cad-preprocess-benchmark" "cad_preprocess.benchmark"
Create-Wrapper "cad-preprocess-diagnose" "cad_preprocess.diagnose_cli"
Create-Wrapper "cad-preprocess-explorer" "cad_preprocess.explorer"

# 8. Update PATH
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -notlike "*$BIN_DIR*") {
    Print-Step "Adding $BIN_DIR to User PATH"
    $NewPath = "$CurrentPath;$BIN_DIR"
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    $env:Path = "$env:Path;$BIN_DIR" # Update current session
    Print-Success "PATH updated. You may need to restart your terminal."
}

Write-Host "`n================================================================================" -ForegroundColor Green
Print-Success "INSTALLATION COMPLETE!"
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  Available Commands:"
Write-Host "    - cad-preprocess"
Write-Host "    - cad-preprocess-benchmark"
Write-Host "    - cad-preprocess-explorer"
Write-Host ""
Write-Host "  Try: cad-preprocess --help"
Write-Host "================================================================================" -ForegroundColor Green
