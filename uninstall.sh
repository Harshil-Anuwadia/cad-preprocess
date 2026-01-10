#!/bin/bash
#
# Uninstaller for CAD Preprocess
#
# Usage: sudo ./uninstall.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Uninstalling CAD Preprocess...${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root: sudo ./uninstall.sh${NC}"
    exit 1
fi

# Remove files
rm -rf /usr/local/lib/cad-preprocess 2>/dev/null || true
rm -rf /usr/lib/cad-preprocess 2>/dev/null || true
rm -f /usr/local/bin/cad-preprocess 2>/dev/null || true
rm -f /usr/bin/cad-preprocess 2>/dev/null || true
rm -rf /etc/cad-preprocess 2>/dev/null || true

echo -e "${GREEN}CAD Preprocess uninstalled successfully!${NC}"
