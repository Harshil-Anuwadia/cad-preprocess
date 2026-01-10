#!/bin/bash
#
# Quick installer for CAD Preprocess (without .deb)
# Installs directly to /usr/local
#
# Usage: sudo ./install.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CAD Preprocess Quick Installer       ${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root: sudo ./install.sh${NC}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/usr/local/lib/cad-preprocess"
BIN_DIR="/usr/local/bin"
CONFIG_DIR="/etc/cad-preprocess"

echo -e "\n${YELLOW}Step 1: Installing Python dependencies...${NC}"
pip3 install pydicom>=2.3.0 numpy>=1.21.0 Pillow>=9.0.0 PyYAML>=6.0 scikit-image>=0.19.0

echo -e "\n${YELLOW}Step 2: Creating directories...${NC}"
mkdir -p "${INSTALL_DIR}"
mkdir -p "${CONFIG_DIR}"

echo -e "\n${YELLOW}Step 3: Copying files...${NC}"
cp -r "${SCRIPT_DIR}/src/cad_preprocess" "${INSTALL_DIR}/"

# Copy config if exists
if [ -f "${SCRIPT_DIR}/config.example.yaml" ]; then
    cp "${SCRIPT_DIR}/config.example.yaml" "${CONFIG_DIR}/"
fi

echo -e "\n${YELLOW}Step 4: Creating executable...${NC}"

cat > "${BIN_DIR}/cad-preprocess" << 'EOF'
#!/usr/bin/env python3
"""CAD Preprocess CLI."""
import sys
sys.path.insert(0, '/usr/local/lib/cad-preprocess')

from cad_preprocess.cli import main

if __name__ == '__main__':
    sys.exit(main())
EOF

chmod +x "${BIN_DIR}/cad-preprocess"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Run: ${YELLOW}cad-preprocess --help${NC}"
echo ""
echo -e "To uninstall: ${YELLOW}sudo ./uninstall.sh${NC}"
echo ""
