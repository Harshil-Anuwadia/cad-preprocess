#!/bin/bash

# ==============================================================================
# CAD-PREPROCESS UNIVERSAL INSTALLER (V2)
# ==============================================================================
# A robust, one-line installer that handles system dependencies, 
# python environments, and package installation across Linux distributions.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../install.sh | bash
#
# For testing:
#   INSTALL_DIR=/tmp/cad-test BIN_DIR=/tmp/bin bash install.sh
# ==============================================================================

set -e

# --- Configuration (with overrides for testing) ---
REPO_URL="https://github.com/Harshil-Anuwadia/cad-preprocess.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/cad-preprocess}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

# --- UI Helpers ---
BOLD="\033[1m"
GREEN="\033[92m"
BLUE="\033[94m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"

print_header() {
    echo -e "${BOLD}${MAGENTA}"
    echo "================================================================================"
    echo "          CAD-PREPROCESS — Unified Medical Imaging Setup"
    echo "================================================================================"
    echo -e "${RESET}"
}

print_step() { echo -e "${BLUE}[*]${RESET} $1..."; }
print_success() { echo -e "${GREEN}[✓]${RESET} $1"; }
print_error() { echo -e "${RED}[✗] ERROR:${RESET} $1" >&2; exit 1; }

# --- System Detection ---
detect_os() {
    print_step "Detecting system environment"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        else
            OS="unknown-linux"
        fi
    else
        print_error "This installer currently only supports Linux-based systems."
    fi
    print_success "Platform: $OS ($(uname -m))"
}

# --- Dependency Installation ---
install_dependencies() {
    print_step "Validating system dependencies"
    
    # Check for sudo
    SUDO_CMD=""
    if command -v sudo &> /dev/null; then
        SUDO_CMD="sudo"
    fi

    case $OS in
        ubuntu|debian|kali|pop|linuxmint)
            if [ -n "$SUDO_CMD" ]; then
                $SUDO_CMD apt-get update -qq
                # Added dpkg-dev and build-essential for building .deb
                $SUDO_CMD apt-get install -y -qq git python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 \
                    build-essential dpkg-dev \
                    libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
                    libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 libxcb-shape0
            else
                print_step "Sudo not found, skipping system package installation."
            fi
            ;;
        arch|manjaro|endeavouros)
            if [ -n "$SUDO_CMD" ]; then
                $SUDO_CMD pacman -Sy --noconfirm --needed git python-pip python-virtualenv mesa libglvnd \
                    libxkbcommon-x11 base-devel
            else
                print_step "Sudo not found, skipping system package installation."
            fi
            ;;
        fedora|rhel|centos|almalinux|rocky)
            if [ -n "$SUDO_CMD" ]; then
                $SUDO_CMD dnf install -y -q git python3-pip mesa-libGL libxkbcommon-x11 glib2 libxcb gcc
            else
                print_step "Sudo not found, skipping system package installation."
            fi
            ;;
        *)
            print_step "Distribution $OS not explicitly supported for auto-dep install."
            ;;
    esac
    print_success "System environment verified"
}

# --- Native Build and Install ---
install_native() {
    local SUDO_CMD=""
    if command -v sudo &> /dev/null; then SUDO_CMD="sudo"; fi

    case $OS in
        ubuntu|debian|kali|pop|linuxmint)
            print_step "Building native Debian package (.deb)"
            bash scripts/build_deb.sh
            local deb_file=$(ls cad-preprocess_*.deb | head -n 1)
            if [ -n "$deb_file" ]; then
                print_step "Installing $deb_file via apt"
                if [ -n "$SUDO_CMD" ]; then
                    $SUDO_CMD apt install -y ./"$deb_file"
                    print_success "Native installation complete"
                    return 0
                fi
            fi
            ;;
        arch|manjaro|endeavouros)
            print_step "Building native Arch Linux package (.pkg.tar.zst)"
            bash scripts/build_arch.sh
            local pkg_file=$(ls cad-preprocess-bundled-*.pkg.tar.zst | head -n 1)
            if [ -n "$pkg_file" ]; then
                print_step "Installing $pkg_file via pacman"
                if [ -n "$SUDO_CMD" ]; then
                    $SUDO_CMD pacman -U --noconfirm "$pkg_file"
                    print_success "Native installation complete"
                    return 0
                fi
            fi
            ;;
    esac
    return 1 # Fallback to venv
}

# --- Installation Logic ---
main_install() {
    print_header
    detect_os
    install_dependencies

    # Setup Directory
    print_step "Preparing installation path: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"

    # Clone or Update
    if [ -d "$INSTALL_DIR/.git" ]; then
        print_step "Updating existing source code"
        cd "$INSTALL_DIR"
        git pull -q
    else
        print_step "Downloading source code from GitHub"
        git clone -q "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # Try Native Install first if running as root/sudo available
    if [[ "$INSTALL_DIR" != "/tmp/"* ]]; then
        if install_native; then
            echo -e "\n${BOLD}${GREEN}================================================================================"
            echo "          ✨ NATIVE SYSTEM INSTALLATION SUCCESSFUL ✨"
            echo "================================================================================${RESET}"
            echo "  Commands are now globally available in /usr/bin"
            exit 0
        fi
    fi

    # Fallback to Virtual Environment (Venv)
    print_step "Falling back to isolated Python environment (venv)"
    python3 -m venv venv
    source venv/bin/activate

    # Install Package with progress feel
    print_step "Installing cad-preprocess with full feature set (GUI + Performance)"
    pip install -q --upgrade pip
    
    # Simulate progress for better UX since pip is quiet
    echo -n "  [ Progress: "
    pip install -q ".[explorer,performance]" &
    PID=$!
    while kill -0 $PID 2>/dev/null; do
        echo -n "■"
        sleep 0.5
    done
    echo " ] Done!"
    
    print_success "Package and dependencies installed successfully"

    # Create Wrapper Scripts
    print_step "Registering CLI commands"
    
    # We use a template for the wrappers
    create_wrapper() {
        local cmd_name=$1
        local module=$2
        cat > "$BIN_DIR/$cmd_name" <<EOF
#!/bin/bash
# Auto-generated wrapper for $cmd_name
export PYTHONPATH="$INSTALL_DIR/src:\$PYTHONPATH"
source "$INSTALL_DIR/venv/bin/activate"
exec python3 -m $module "\$@"
EOF
        chmod +x "$BIN_DIR/$cmd_name"
    }

    create_wrapper "cad-preprocess" "cad_preprocess.cli"
    create_wrapper "cad-preprocess-benchmark" "cad_preprocess.benchmark"
    create_wrapper "cad-preprocess-diagnose" "cad_preprocess.diagnose_cli"
    create_wrapper "cad-preprocess-explorer" "cad_preprocess.explorer"

    # Create uninstaller wrapper
    print_step "Configuring uninstaller"
    cp uninstall.sh "$INSTALL_DIR/uninstall.sh"
    chmod +x "$INSTALL_DIR/uninstall.sh"
    cat > "$BIN_DIR/cad-preprocess-uninstall" <<EOF
#!/bin/bash
exec bash "$INSTALL_DIR/uninstall.sh"
EOF
    chmod +x "$BIN_DIR/cad-preprocess-uninstall"

    print_success "Binary wrappers created in $BIN_DIR"

    # Setup Shell Path
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        print_step "Adding installation binaries to your PATH"
        SHELL_RC=""
        case $SHELL in
            */zsh) SHELL_RC="$HOME/.zshrc" ;;
            */bash) SHELL_RC="$HOME/.bashrc" ;;
        esac

        if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
            if ! grep -q "$BIN_DIR" "$SHELL_RC"; then
                echo "" >> "$SHELL_RC"
                echo "# CAD-Preprocess Binary Path" >> "$SHELL_RC"
                echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$SHELL_RC"
                print_success "Modified $SHELL_RC. Run 'source $SHELL_RC' to update current session."
            fi
        fi
    fi

    echo -e "\n${BOLD}${GREEN}================================================================================"
    echo "          ✨ INSTALLATION SUCCESSFUL — WELCOME TO CAD-PREPROCESS ✨"
    echo "================================================================================${RESET}"
    echo -e "  ${BOLD}Commands Available Now:${RESET}"
    echo -e "    ${CYAN}• cad-preprocess${RESET}           (Main preprocessing tool)"
    echo -e "    ${CYAN}• cad-preprocess-benchmark${RESET} (Performance testing suite)"
    echo -e "    ${CYAN}• cad-preprocess-diagnose${RESET}  (DICOM health checker)"
    echo -e "    ${CYAN}• cad-preprocess-explorer${RESET}  (GUI Results Viewer)"
    echo ""
    echo -e "  ${YELLOW}Quick Start:${RESET}"
    echo -e "    ${BOLD}cad-preprocess -i ./input_dicoms -o ./output_results${RESET}"
    echo "================================================================================"
}

main_install
