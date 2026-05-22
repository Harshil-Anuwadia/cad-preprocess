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
# Install the package directly. Pip handles its own logic, and skipping the explicit 
# upgrade avoids common Windows file-locking issues.
& $VENV_PYTHON -m pip install -q ".[explorer,performance]"
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

# Create uninstaller wrapper
Print-Step "Configuring uninstaller"
try {
    # Check if the source uninstall.ps1 exists in the current directory (from git clone)
    if (Test-Path "uninstall.ps1") {
        # Only copy if source and destination are different
        $DestPath = "$INSTALL_DIR\uninstall.ps1"
        if ((Get-Item "uninstall.ps1").FullName -ne (Get-Item $DestPath -ErrorAction SilentlyContinue).FullName) {
            Copy-Item "uninstall.ps1" $DestPath -Force -ErrorAction SilentlyContinue
        }
    }
} catch {
    # Non-critical failure, continue to PATH update
}

$UninstallPath = "$BIN_DIR\cad-preprocess-uninstall.bat"
"@echo off`npowershell -ExecutionPolicy Bypass -File `"$INSTALL_DIR\uninstall.ps1`"" | Out-File -FilePath $UninstallPath -Encoding ascii

# 8. Update PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$PathUpdated = $false

if ($UserPath -split ';' -notcontains $BIN_DIR) {
    Print-Step "Adding $BIN_DIR to User PATH"
    $NewPath = "$UserPath;$BIN_DIR"
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    $env:Path = "$env:Path;$BIN_DIR" # Update current session
    $PathUpdated = $true
    Print-Success "Global PATH updated."
} else {
    $env:Path = "$env:Path;$BIN_DIR" # Ensure current session is always updated
    Print-Success "Binary directory already in PATH."
}

Write-Host "`n================================================================================" -ForegroundColor Green
Print-Success "INSTALLATION COMPLETE!"
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  Available Commands:"
Write-Host "    - cad-preprocess"
Write-Host "    - cad-preprocess-benchmark"
Write-Host "    - cad-preprocess-explorer"
Write-Host ""
if ($PathUpdated) {
    Write-Host "  NOTE: Please RESTART your terminal (close and open again) to use the commands." -ForegroundColor Yellow
}
Write-Host "  Or run this to use them immediately in this window:"
Write-Host "  `$env:Path += ';$BIN_DIR'" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Green
