# ==============================================================================
# CAD-PREPROCESS WINDOWS UNINSTALLER (PowerShell)
# ==============================================================================
# Completely removes cad-preprocess and its environment from Windows.
# ==============================================================================

$INSTALL_DIR = "$HOME\.local\share\cad-preprocess"
$BIN_DIR = "$HOME\.local\bin"

Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "          CAD-PREPROCESS -- Windows Uninstallation Utility" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

# 1. Remove Binary Wrappers
Write-Host "[*] Removing command wrappers from $BIN_DIR..." -ForegroundColor Cyan
if (Test-Path $BIN_DIR) {
    Get-ChildItem -Path $BIN_DIR -Filter "cad-preprocess*" | Remove-Item -Force -ErrorAction SilentlyContinue
}

# 2. Remove Installation Directory
if (Test-Path $INSTALL_DIR) {
    Write-Host "[*] Removing installation directory: $INSTALL_DIR..." -ForegroundColor Cyan
    Remove-Item -Path $INSTALL_DIR -Recurse -Force -ErrorAction SilentlyContinue
}

# 3. Update PATH
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -like "*$BIN_DIR*") {
    Write-Host "[*] Cleaning up User PATH..." -ForegroundColor Cyan
    $PathArray = $CurrentPath.Split(";")
    $NewPathArray = $PathArray | Where-Object { $_ -ne $BIN_DIR -and $_ -ne "" }
    $NewPath = $NewPathArray -join ";"
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    Write-Host "[V] PATH updated." -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "          CAD-PREPROCESS HAS BEEN REMOVED FROM YOUR SYSTEM" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "  Note: Global Python and Git installations were kept."
Write-Host "================================================================================" -ForegroundColor Green
