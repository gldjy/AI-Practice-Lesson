#!/usr/bin/env bash
# check_gpu.sh
# Inspects the NVIDIA hardware and driver state on a Linux host without relying
# on nvidia-smi, which only exists once the driver is installed and loaded.

set -u

section() { printf '\n== %s\n' "$*"; }
note() { printf '  -> %s\n' "$*"; }
empty() { printf '  (none)\n'; }

if [ -r /etc/nv_tegra_release ]; then
    section "jetson board"
    printf '  model   : %s\n' "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
    printf '  release : %s\n' "$(head -1 /etc/nv_tegra_release)"

    section "memory, unified and shared with the gpu"
    free -h | sed 's/^/  /'

    section "nvidia-smi"
    note "Tegra has no discrete GPU and no nvidia-smi, nothing is missing"

    section "tegrastats"
    if command -v tegrastats >/dev/null; then
        timeout 5 tegrastats 2>/dev/null | head -n 2 | sed 's/^/  /'
    else
        note "tegrastats not found, install jetson-stats for a readable view: pip install -U jetson-stats"
    fi

    printf '\n'
    note "the gpu is on-chip, lspci only lists the pcie bridges"
    note "paddle has no arm64 gpu wheel, tools/setup_env.sh installs the cpu build"
    exit 0
fi

section "display controllers"
if command -v lspci >/dev/null; then
    lspci -nn 2>/dev/null | grep -iE 'vga|3d|display' || empty
else
    note "lspci is missing, install it with: sudo apt install pciutils"
fi

nvidia_slots=$(lspci -D 2>/dev/null | grep -i nvidia | cut -d' ' -f1)
if [ -n "$nvidia_slots" ]; then
    section "kernel driver in use"
    for slot in $nvidia_slots; do
        lspci -k -s "$slot" 2>/dev/null | sed -n '1,4p' | sed 's/^/  /'
        printf '\n'
    done
fi

section "loaded modules"
lsmod 2>/dev/null | grep -E '^(nvidia|nouveau)' || empty

section "driver version"
if [ -r /proc/driver/nvidia/version ]; then
    sed 's/^/  /' /proc/driver/nvidia/version
else
    note "/proc/driver/nvidia/version does not exist, the driver is not loaded"
fi

section "device nodes"
ls /dev/nvidia* 2>/dev/null || empty

section "nvidia-smi"
if command -v nvidia-smi >/dev/null; then
    note "found at $(command -v nvidia-smi)"
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    note "found at /usr/lib/wsl/lib/nvidia-smi (WSL, add that directory to PATH)"
else
    note "not found on PATH"
fi

section "installed packages"
if command -v dpkg >/dev/null; then
    packages=$(dpkg -l 2>/dev/null | grep -iE 'nvidia|cuda' | awk '{print "  " $2 " " $3}')
    printf '%s\n' "${packages:-(none)}"
else
    note "dpkg not available"
fi

section "kernel messages"
if dmesg 2>/dev/null | grep -qiE 'nvidia|nouveau'; then
    dmesg 2>/dev/null | grep -iE 'nvidia|nouveau|NVRM' | tail -n 15 | sed 's/^/  /'
else
    note "no nvidia/nouveau lines, or dmesg needs root: sudo dmesg | grep -i nvidia"
fi

section "recommended driver"
if command -v ubuntu-drivers >/dev/null; then
    devices=$(ubuntu-drivers devices 2>/dev/null | sed 's/^/  /')
    printf '%s\n' "${devices:-(none)}"
else
    note "ubuntu-drivers is missing, install it with: sudo apt install ubuntu-drivers-common"
fi

if command -v mokutil >/dev/null; then
    section "secure boot"
    mokutil --sb-state 2>/dev/null | sed 's/^/  /' || empty
fi

printf '\n'
note "no driver loaded yet? install one and reboot: sudo ubuntu-drivers install"
note "vram is only reported by the driver, read it from nvidia-smi after the reboot"
