#!/bin/bash

# ==============================================================================
# CAD-PREPROCESS UNIVERSAL INSTALLER (PRO)
# ==============================================================================
# A professional, high-feedback installer for Linux distributions.
# Handles system deps, native builds, and venv isolation with modern UI.
# ==============================================================================

set -e

# --- Configuration ---
REPO_URL="https://github.com/Harshil-Anuwadia/cad-preprocess.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/cad-preprocess}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

# --- Professional UI Styling ---
BOLD="\033[1m"
DIM="\033[2m"
ITALIC="\033[3m"
UNDERLINE="\033[4m"

RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"
RESET="\033[0m"

# Status Icons
TICK="${GREEN}✔${RESET}"
CROSS="${RED}✘${RESET}"
INFO="${BLUE}ℹ${RESET}"
STEP="${CYAN}➜${RESET}"
WAIT="${YELLOW}⏳${RESET}"

# --- UI Components ---
print_banner() {
    clear
    echo -e "${BOLD}${MAGENTA}"
    echo "  ╔══════════════════════════════════════════════════════════════════════╗"
    echo "  ║                                                                      ║"
    echo "  ║    CAD-PREPROCESS — Medical Imaging Preprocessing Pipeline           ║"
    echo "  ║    Standardizing DICOM Workflows for Production & Research           ║"
    echo "  ║                                                                      ║"
    echo "  ╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

print_section() {
    echo -e "\n${BOLD}${WHITE}==>${RESET} ${BOLD}$1${RESET}"
}

print_step() {
    echo -e "  ${STEP} $1..."
}

print_success() {
    echo -e "  ${TICK} $1"
}

print_error() {
    echo -e "\n  ${CROSS} ${RED}${BOLD}ERROR:${RESET} $1" >&2
    exit 1
}

show_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "  ${WAIT}  [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# --- System Logic ---
detect_os() {
    print_section "System Inspection"
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
    echo -e "      ${DIM}Platform:${RESET}  $OS ($(uname -m))"
    echo -e "      ${DIM}Kernel:${RESET}    $(uname -r)"
}

install_dependencies() {
    print_section "Environment Preparation"
    print_step "Validating system dependencies"
    
    SUDO_CMD=""
    if command -v sudo &> /dev/null; then SUDO_CMD="sudo"; fi

    case $OS in
        ubuntu|debian|kali|pop|linuxmint)
            if [ -n "$SUDO_CMD" ]; then
                $SUDO_CMD apt-get update -qq
                $SUDO_CMD apt-get install -y -qq git python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 \
                    build-essential dpkg-dev libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
                    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 \
                    libxcb-xinput0 libxcb-xfixes0 libxcb-shape0 > /dev/null 2>&1 &
                show_spinner $!
            fi
            ;;
        arch|manjaro|endeavouros)
            if [ -n "$SUDO_CMD" ]; then
                $SUDO_CMD pacman -Sy --noconfirm --needed git python-pip python-virtualenv mesa \
                    libglvnd libxkbcommon-x11 base-devel > /dev/null 2>&1 &
                show_spinner $!
            fi
            ;;
        fedora|rhel|centos|almalinux|rocky)
            if [ -n "$SUDO_CMD" ]; then
                $SUDO_CMD dnf install -y -q git python3-pip mesa-libGL libxkbcommon-x11 glib2 libxcb gcc > /dev/null 2>&1 &
                show_spinner $!
            fi
            ;;
    esac
    print_success "System environment hardened"
}

install_native() {
    print_section "Native Package Build"
    local SUDO_CMD=""
    if command -v sudo &> /dev/null; then SUDO_CMD="sudo"; fi

    case $OS in
        ubuntu|debian|kali|pop|linuxmint)
            print_step "Assembling Debian package (.deb)"
            bash scripts/build_deb.sh > /dev/null 2>&1 &
            show_spinner $!
            local deb_file=$(ls cad-preprocess_*.deb | head -n 1)
            if [ -n "$deb_file" ]; then
                print_step "Installing system-wide via apt"
                if [ -n "$SUDO_CMD" ]; then
                    $SUDO_CMD apt install -y -qq ./"$deb_file" > /dev/null 2>&1
                    return 0
                fi
            fi
            ;;
        arch|manjaro|endeavouros)
            print_step "Assembling Arch package (.pkg.tar.zst)"
            bash scripts/build_arch.sh > /dev/null 2>&1 &
            show_spinner $!
            local pkg_file=$(ls cad-preprocess-bundled-*.pkg.tar.zst | head -n 1)
            if [ -n "$pkg_file" ]; then
                print_step "Installing system-wide via pacman"
                if [ -n "$SUDO_CMD" ]; then
                    $SUDO_CMD pacman -U --noconfirm "$pkg_file" > /dev/null 2>&1
                    return 0
                fi
            fi
            ;;
    esac
    return 1
}

# --- Main Flow ---
main() {
    print_banner
    detect_os
    
    if [[ "$INSTALL_DIR" != "/tmp/"* ]]; then
        install_dependencies
    fi

    print_section "Repository Management"
    mkdir -p "$INSTALL_DIR" "$BIN_DIR"
    
    if [ -d "$INSTALL_DIR/.git" ]; then
        print_step "Pulling latest updates"
        cd "$INSTALL_DIR" && git pull -q
    else
        print_step "Cloning from GitHub"
        git clone -q "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    print_success "Source code synchronized"

    if [[ "$INSTALL_DIR" != "/tmp/"* ]]; then
        if install_native; then
            echo -e "\n${BOLD}${GREEN}  ${TICK} NATIVE SYSTEM INSTALLATION SUCCESSFUL${RESET}"
            echo -e "      ${DIM}Type 'cad-preprocess --help' to get started.${RESET}\n"
            exit 0
        fi
    fi

    print_section "Python Environment"
    print_step "Initializing isolated venv"
    python3 -m venv venv
    source venv/bin/activate
    
    print_step "Installing package [Full Feature Set]"
    pip install -q --upgrade pip
    pip install -q ".[explorer,performance]" &
    show_spinner $!
    print_success "Runtime environment ready"

    print_section "CLI Configuration"
    create_wrapper() {
        cat > "$BIN_DIR/$1" <<EOF
#!/bin/bash
export PYTHONPATH="$INSTALL_DIR/src:\$PYTHONPATH"
source "$INSTALL_DIR/venv/bin/activate"
exec python3 -m $2 "\$@"
EOF
        chmod +x "$BIN_DIR/$1"
    }

    create_wrapper "cad-preprocess" "cad_preprocess.cli"
    create_wrapper "cad-preprocess-benchmark" "cad_preprocess.benchmark"
    create_wrapper "cad-preprocess-diagnose" "cad_preprocess.diagnose_cli"
    create_wrapper "cad-preprocess-explorer" "cad_preprocess.explorer"
    
    cp uninstall.sh "$INSTALL_DIR/uninstall.sh"
    chmod +x "$INSTALL_DIR/uninstall.sh"
    create_wrapper "cad-preprocess-uninstall" "cad_preprocess.cli" # dummy for bash template
    cat > "$BIN_DIR/cad-preprocess-uninstall" <<EOF
#!/bin/bash
exec bash "$INSTALL_DIR/uninstall.sh"
EOF
    chmod +x "$BIN_DIR/cad-preprocess-uninstall"
    print_success "CLI commands registered in $BIN_DIR"

    # Shell Integration
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        SHELL_RC=""
        case $SHELL in
            */zsh) SHELL_RC="$HOME/.zshrc" ;;
            */bash) SHELL_RC="$HOME/.bashrc" ;;
        esac
        if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
            if ! grep -q "$BIN_DIR" "$SHELL_RC"; then
                echo -e "\n# CAD-Preprocess Binary Path\nexport PATH=\"\$PATH:$BIN_DIR\"" >> "$SHELL_RC"
                print_success "Updated $SHELL_RC"
            fi
        fi
    fi

    echo -e "\n${BOLD}${GREEN}  🎉 INSTALLATION COMPLETE!${RESET}"
    echo -e "  ${DIM}──────────────────────────────────────────────────────────────────────${RESET}"
    echo -e "  ${BOLD}Try these commands:${RESET}"
    echo -e "    ${CYAN}cad-preprocess${RESET}           ${DIM}(Processing)${RESET}"
    echo -e "    ${CYAN}cad-preprocess-benchmark${RESET} (Stress Test)"
    echo -e "    ${CYAN}cad-preprocess-explorer${RESET}  ${DIM}(GUI Viewer)${RESET}"
    echo -e "  ${DIM}──────────────────────────────────────────────────────────────────────${RESET}\n"
}

main
