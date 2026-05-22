#!/bin/bash

# ==============================================================================
# CAD-PREPROCESS UNINSTALLER
# ==============================================================================
# Completely removes cad-preprocess, its environment, and binary wrappers.
# ==============================================================================

INSTALL_DIR="$HOME/.local/share/cad-preprocess"
BIN_DIR="$HOME/.local/bin"

# UI Helpers
BOLD="\033[1m"
GREEN="\033[92m"
BLUE="\033[94m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"

echo -e "${BOLD}${YELLOW}================================================================================"
echo "          CAD-PREPROCESS — System Uninstallation Utility"
echo -e "================================================================================${RESET}"

# 1. Native Package Removal
if [ -f /etc/debian_version ] && command -v dpkg &> /dev/null; then
    if dpkg -s cad-preprocess &> /dev/null; then
        echo -e "${BLUE}[*]${RESET} Detected native Debian package. Removing via apt..."
        sudo apt-get remove -y cad-preprocess
    fi
fi

if command -v pacman &> /dev/null; then
    if pacman -Qs cad-preprocess &> /dev/null; then
        echo -e "${BLUE}[*]${RESET} Detected native Arch package. Removing via pacman..."
        sudo pacman -Rs --noconfirm cad-preprocess
    fi
fi

# 2. Remove Binary Wrappers
echo -e "${BLUE}[*]${RESET} Removing CLI wrappers from $BIN_DIR..."
rm -f "$BIN_DIR/cad-preprocess"
rm -f "$BIN_DIR/cad-preprocess-benchmark"
rm -f "$BIN_DIR/cad-preprocess-diagnose"
rm -f "$BIN_DIR/cad-preprocess-explorer"
rm -f "$BIN_DIR/cad-preprocess-uninstall"

# 3. Remove Installation Directory
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${BLUE}[*]${RESET} Removing installation directory: $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"
fi

# 4. Cleanup Shell Config
echo -e "${BLUE}[*]${RESET} Cleaning up shell configuration PATH entries..."
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rc" ]; then
        # Remove the specific block and the PATH export
        sed -i '/# CAD-Preprocess Binary Path/d' "$rc"
        sed -i "s|export PATH=\"\$PATH:$BIN_DIR\"||g" "$rc"
        # Clean up empty lines if any
    fi
done

echo -e "\n${BOLD}${GREEN}================================================================================"
echo "          ✨ CAD-PREPROCESS HAS BEEN COMPLETELY REMOVED ✨"
echo -e "================================================================================${RESET}"
echo "  Note: System-level dependencies (like git, python3-venv) were kept."
echo "  To fully refresh your current shell, run: exec $SHELL"
echo "================================================================================"
