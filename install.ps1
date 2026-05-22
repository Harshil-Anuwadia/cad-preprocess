# ==============================================================================
# CAD-PREPROCESS WINDOWS INSTALLER (PRO)
# ==============================================================================
# A professional, high-feedback installer for Windows.
# Handles Python environment setup, dependencies, and PATH natively.
# ==============================================================================

$ErrorActionPreference = "Stop"

# --- Configuration ---
$REPO_URL = "https://github.com/Harshil-Anuwadia/cad-preprocess.git"
$INSTALL_DIR = "$HOME\.local\share\cad-preprocess"
$BIN_DIR = "$HOME\.local\bin"

# --- Professional UI Components ---
function Print-Header {
    Clear-Host
    Write-Host "  +----------------------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "  |                                                                      |" -ForegroundColor Magenta
    Write-Host "  |    CAD-PREPROCESS -- Medical Imaging Preprocessing Pipeline          |" -ForegroundColor Magenta
    Write-Host "  |    Standardizing DICOM Workflows for Production & Research           |" -ForegroundColor Magenta
    Write-Host "  |                                                                      |" -ForegroundColor Magenta
    Write-Host "  +----------------------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host ""
}

function Print-Section { param($msg) Write-Host "`n==> " -NoNewline -ForegroundColor White; Write-Host $msg -ForegroundColor White -FontWeight Bold }
function Print-Step { param($msg) Write-Host "  ➜ $msg..." -ForegroundColor Cyan }
function Print-Success { param($msg) Write-Host "  [V] $msg" -ForegroundColor Green }
function Print-Error { param($msg) Write-Host "`n  [X] ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Installation Logic ---
function Main {
    Print-Header

    # 1. System Inspection
    Print-Section "System Inspection"
    if (!(Get-Command python -ErrorAction SilentlyContinue)) {
        Print-Error "Python 3.9+ not found. Please install from python.org"
    }
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        Print-Error "Git not found. Please install Git for Windows"
    }
    $osVersion = (Get-CimInstance Win32_OperatingSystem).Caption
    Write-Host "      Platform:  $osVersion" -ForegroundColor Gray
    Write-Host "      Arch:      $env:PROCESSOR_ARCHITECTURE" -ForegroundColor Gray

    # 2. Repository Management
    Print-Section "Repository Management"
    if (!(Test-Path $INSTALL_DIR)) { New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null }
    if (!(Test-Path $BIN_DIR)) { New-Item -ItemType Directory -Path $BIN_DIR -Force | Out-Null }

    if (Test-Path "$INSTALL_DIR\.git") {
        Print-Step "Pulling latest updates"
        Set-Location $INSTALL_DIR
        git pull -q
    } else {
        Print-Step "Cloning from GitHub"
        git clone -q $REPO_URL $INSTALL_DIR
    }
    Print-Success "Source code synchronized"

    # 3. Python Environment
    Print-Section "Python Environment"
    Print-Step "Initializing isolated venv"
    Set-Location $INSTALL_DIR
    python -m venv venv
    $VENV_PYTHON = "$INSTALL_DIR\venv\Scripts\python.exe"

    Print-Step "Installing package [Full Feature Set]"
    & $VENV_PYTHON -m pip install -q ".[explorer,performance]"
    Print-Success "Runtime environment ready"

    # 4. CLI Configuration
    Print-Section "CLI Configuration"
    function Create-Wrapper {
        param($name, $module)
        $path = "$BIN_DIR\$name.bat"
        "@echo off`nset PYTHONPATH=$INSTALL_DIR\src;%PYTHONPATH%`n`"$VENV_PYTHON`" -m $module %*" | Out-File -FilePath $path -Encoding ascii
    }

    Create-Wrapper "cad-preprocess" "cad_preprocess.cli"
    Create-Wrapper "cad-preprocess-benchmark" "cad_preprocess.benchmark"
    Create-Wrapper "cad-preprocess-diagnose" "cad_preprocess.diagnose_cli"
    Create-Wrapper "cad-preprocess-explorer" "cad_preprocess.explorer"

    # Uninstaller
    Copy-Item "uninstall.ps1" "$INSTALL_DIR\uninstall.ps1" -Force -ErrorAction SilentlyContinue
    "@echo off`npowershell -ExecutionPolicy Bypass -File `"$INSTALL_DIR\uninstall.ps1`"" | Out-File -FilePath "$BIN_DIR\cad-preprocess-uninstall.bat" -Encoding ascii
    Print-Success "CLI commands registered in $BIN_DIR"

    # 5. PATH Integration
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -split ';' -notcontains $BIN_DIR) {
        Print-Step "Updating User PATH"
        $NewPath = "$UserPath;$BIN_DIR"
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
        $env:Path = "$env:Path;$BIN_DIR"
        Print-Success "Global PATH updated"
    }

    Write-Host "`n  🎉 INSTALLATION SUCCESSFUL!" -ForegroundColor Green
    Write-Host "  ----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "  Try these commands:" -ForegroundColor White
    Write-Host "    cad-preprocess           (Processing)" -ForegroundColor Cyan
    Write-Host "    cad-preprocess-benchmark (Stress Test)" -ForegroundColor Cyan
    Write-Host "    cad-preprocess-explorer  (GUI Viewer)" -ForegroundColor Cyan
    Write-Host "  ----------------------------------------------------------------------" -ForegroundColor Gray
    Write-Host "  NOTE: Please RESTART your terminal to use commands globally.`n" -ForegroundColor Yellow
}

Main
